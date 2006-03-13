/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "CalibMuon/DTCalibration/interface/DTTTrigCalibration.h"
#include "CalibMuon/DTCalibration/interface/DTTimeBoxFitter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/IOVService/interface/IOV.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"

#include "TFile.h"
#include "TH1F.h"

using namespace std;
using namespace edm;
using namespace cond;

DTTTrigCalibration::DTTTrigCalibration(const edm::ParameterSet& pset) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");

  // Get the label to retrieve digis from the event
  digiLabel = pset.getParameter<string>("digiLabel");

  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
  theFitter = new DTTimeBoxFitter();
  if(debug)
    theFitter->setVerbosity(1);

  // DB related parameters
  theConnect = pset.getParameter<string>("connect");
  theCatalog =  pset.getParameter<string>("catalog");
  theTag = pset.getParameter<string>("tag");
  theMessageLevel = pset.getUntrackedParameter<unsigned int>("messageLevel",0);
  theAuthMethod =  pset.getParameter<unsigned int>("authenticationMethod");
  theCoralUser =  string("CORAL_AUTH_USER=")
    + pset.getUntrackedParameter<string>("coralUser","me");
  theCoralPasswd = string("CORAL_AUTH_PASSWORD=")
    + pset.getUntrackedParameter<string>("coralPasswd","mypass");
  //   theMaxRun = pset.getParameter<int>("runMax");
  //   theMinRun = pset.getParameter<int>("runMin");

  
  if(debug) 
    cout << "[DTTTrigCalibration]Constructor called!" << endl;
}

DTTTrigCalibration::~DTTTrigCalibration(){
  if(debug) 
    cout << "[DTTTrigCalibration]Destructor called!" << endl;

  // Write all time boxes to file
  theFile->cd();
  for(map<DTSuperLayerId, TH1F*>::const_iterator slHisto = theHistoMap.begin();
      slHisto != theHistoMap.end();
      slHisto++) {
    (*slHisto).second->Write();
    delete (*slHisto).second;
  }
  theFile->Close();
  delete theFitter;
}



/// Perform the real analysis
void DTTTrigCalibration::analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {
  // Get the digis from the event
  Handle<DTDigiCollection> digis; 
  event.getByLabel(digiLabel, digis);

  // Iterate through all digi collections ordered by LayerId   
  DTDigiCollection::DigiRangeIterator dtLayerIt;
  for (dtLayerIt = digis->begin();
       dtLayerIt != digis->end();
       ++dtLayerIt){
    // The layerId
    const DTLayerId layerId = (*dtLayerIt).first;
    const DTSuperLayerId slId = layerId.superlayerId();

    // Get the histo from the map
    TH1F *hTBox = theHistoMap[slId];
    if(hTBox == 0) {
      // Book the histogram
      theFile->cd();
      hTBox = new TH1F(getTBoxName(slId).c_str(), "Time box (ns)", 10000, -1000, 9000);
      if(debug)
	cout << "  New Time Box: " << hTBox->GetName() << endl;
      theHistoMap[slId] = hTBox;
    }

    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& digiRange = (*dtLayerIt).second;

    // Loop over all digis in the given range
    for (DTDigiCollection::const_iterator digi = digiRange.first;
	 digi != digiRange.second;
	 digi++) {
      theFile->cd();
      hTBox->Fill((*digi).time());
      if(debug) {
 	cout << "   Filling Time Box: " << hTBox->GetName() << endl;
 	cout << "           time(ns): " << (*digi).time() << endl;
      }
    }
  }
}


void DTTTrigCalibration::endJob() {
  // Connect to DB to write the results
  ServiceLoader* loader = new ServiceLoader;
  // Set the coral password
  ::putenv(const_cast<char*>(theCoralUser.c_str()));
  ::putenv(const_cast<char*>(theCoralPasswd.c_str()));

  // Set the authentication method  
  if(theAuthMethod == 1) {
    loader->loadAuthenticationService(cond::XML);
  }else{
    loader->loadAuthenticationService(cond::Env);
  }

  // Set the message level
  switch (theMessageLevel) {
  case 0 :
    loader->loadMessageService(cond::Error);
    break;    
  case 1:
    loader->loadMessageService(cond::Warning);
    break;
  case 2:
    loader->loadMessageService( cond::Info );
    break;
  case 3:
    loader->loadMessageService( cond::Debug );
    break;  
  default:
    loader->loadMessageService();
  }
  try{
    DBSession* session = new DBSession(theConnect);
    session->setCatalog(theCatalog);
    session->connect(cond::ReadWriteCreate);
    DBWriter pwriter(*session, "DTTTrigs");
    DBWriter iovwriter(*session, "IOV");

    session->startUpdateTransaction();

    IOV* ttrigIOV= new IOV; 
   
    // Create the object to be written to DB
    DTTtrig* tTrig = new DTTtrig(theTag);


    // Loop over the map, fit the histos and write the resulting values to the DB
    for(map<DTSuperLayerId, TH1F*>::const_iterator slHisto = theHistoMap.begin();
	slHisto != theHistoMap.end();
	slHisto++) {
      pair<double, double> meanAndSigma = theFitter->fitTimeBox((*slHisto).second);
      tTrig->setSLTtrig((*slHisto).first.wheel(),
			(*slHisto).first.station(),
			(*slHisto).first.sector(),
			(*slHisto).first.superlayer(),
			meanAndSigma.first); //FIXME: should use tdc counts and sigma???
      if(debug) {
	cout << " SL: " << (*slHisto).first
	     << " mean = " << meanAndSigma.first
	     << " sigma = " << meanAndSigma.second << endl;
      }
    }

    string mytok = pwriter.markWrite<DTTtrig>(tTrig);
    // Set the IOV
    ttrigIOV->iov.insert(make_pair(edm::IOVSyncValue::endOfTime().eventID().run(), mytok ));
    if(debug)
      cout << "  iov size " << ttrigIOV->iov.size() << endl;

    if(debug)
      cout << "  markWrite IOV..." << endl;
    string tTrigiovToken = iovwriter.markWrite<cond::IOV>(ttrigIOV);
    if(debug)
      cout << "   Commit..." << endl;
    session->commit();//commit all in one
    if(debug)
      cout << "  iov size " << ttrigIOV->iov.size() << endl;
    session->disconnect();
    delete session;
    if(debug)
      cout << "  Add MetaData... " << endl;
    cond::MetaData metadata_svc(theConnect, *loader );
    metadata_svc.connect();
    metadata_svc.addMapping(theTag, tTrigiovToken );
    metadata_svc.disconnect();
    if(debug)
      cout << "   Done." << endl;
  } catch( const cond::Exception& er ) {
    std::cout << er.what() << std::endl;
  } catch( ... ) {
    std::cout << "Unknown excpetion while writeing to DB!" << std::endl;
  }


  delete loader;
}


string DTTTrigCalibration::getTBoxName(const DTSuperLayerId& slId) const {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" << slId.wheel() << "_" << slId.station() << "_" << slId.sector()
	    << "_SL" << slId.superlayer() << "_hTimeBox";
  theStream >> histoName;
  return histoName;
}
