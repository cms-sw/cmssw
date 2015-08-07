/*
 * \file AlcaBeamMonitor.cc
 * \author Lorenzo Uplegger/FNAL
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
//#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include "RecoVertex/BeamSpotProducer/interface/PVFitter.h"
#include "DQM/BeamMonitor/plugins/AlcaBeamMonitor.h"
#include "DQMServices/Core/interface/QReport.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <numeric>
//#include <iostream>

using namespace std;
using namespace edm;
using namespace reco;

//----------------------------------------------------------------------------------------------------------------------
AlcaBeamMonitor::AlcaBeamMonitor( const ParameterSet& ps ) :
  parameters_         	(ps),
  monitorName_        	(parameters_.getUntrackedParameter<string>("MonitorName","YourSubsystemName")),
  primaryVertexLabel_ 	(consumes<VertexCollection>(
      parameters_.getUntrackedParameter<InputTag>("PrimaryVertexLabel"))),
  trackLabel_         	(consumes<reco::TrackCollection>(
      parameters_.getUntrackedParameter<InputTag>("TrackLabel"))),
  scalerLabel_        	(consumes<BeamSpot>(
      parameters_.getUntrackedParameter<InputTag>("ScalerLabel"))),
  beamSpotLabel_      	(parameters_.getUntrackedParameter<InputTag>("BeamSpotLabel")),
  numberOfValuesToSave_ (0)
{
  dbe_ = Service<DQMStore>().operator->();

  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;

  theBeamFitter_ = new BeamFitter(parameters_, consumesCollector() );
  theBeamFitter_->resetTrkVector();
  theBeamFitter_->resetLSRange();
  theBeamFitter_->resetRefTime();
  theBeamFitter_->resetPVFitter();

  thePVFitter_ = new PVFitter(parameters_, consumesCollector());


  varNamesV_.push_back("x");
  varNamesV_.push_back("y");
  varNamesV_.push_back("z");
  varNamesV_.push_back("sigmaX");
  varNamesV_.push_back("sigmaY");
  varNamesV_.push_back("sigmaZ");

  histoByCategoryNames_.insert( pair<string,string>("run",        "Coordinate"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-BeamFit"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-Scalers"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-BeamFit"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-Scalers"));

  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased BeamSpotFit"));  
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased PrimaryVertex"));
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased DataBase"));     
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased Scalers"));      
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased PrimaryVertex-DataBase fit"));
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased PrimaryVertex-Scalers fit"));
  histoByCategoryNames_.insert( pair<string,string>("validation", "Lumibased Scalers-DataBase fit"));
  histoByCategoryNames_.insert( pair<string,string>("validation", "Lumibased PrimaryVertex-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("validation", "Lumibased PrimaryVertex-Scalers"));


  for(vector<string>::iterator itV=varNamesV_.begin(); itV!=varNamesV_.end(); itV++){
    for(multimap<string,string>::iterator itM=histoByCategoryNames_.begin(); itM!=histoByCategoryNames_.end(); itM++){
      if(itM->first=="run"){
        histosMap_[*itV][itM->first][itM->second] = 0;
      }
      else{
        positionsMap_[*itV][itM->first][itM->second] = 3*numberOfValuesToSave_;//value, error, ok 
        ++numberOfValuesToSave_;
      }
    }
  }
  
//  beamSpotsMap_["BF"] = map<LuminosityBlockNumber_t,BeamSpot>();//For each lumi the beamfitter will have a result
//  beamSpotsMap_["PV"] = map<LuminosityBlockNumber_t,BeamSpot>();//For each lumi the PVfitter will have a result
//  beamSpotsMap_["DB"] = map<LuminosityBlockNumber_t,BeamSpot>();//For each lumi we take the values that are stored in the database, already collapsed then
//  beamSpotsMap_["SC"] = map<LuminosityBlockNumber_t,BeamSpot>();//For each lumi we take the beamspot value in the file that is the same as the scaler for the alca reco stream
//  beamSpotsMap_["BF"] = 0;//For each lumi the beamfitter will have a result
//  beamSpotsMap_["PV"] = 0;//For each lumi the PVfitter will have a result
//  beamSpotsMap_["DB"] = 0;//For each lumi we take the values that are stored in the database, already collapsed then
//  beamSpotsMap_["SC"] = 0;//For each lumi we take the beamspot value in the file that is the same as the scaler for the alca reco stream
}


AlcaBeamMonitor::~AlcaBeamMonitor() {
  if(theBeamFitter_ != 0){
    delete theBeamFitter_;
  }
  
  if(thePVFitter_ != 0){
    delete thePVFitter_;
  }
}


//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::beginJob() {
  string name;
  string title;
  dbe_->setCurrentFolder(monitorName_+"Debug");
  for(HistosContainer::iterator itM=histosMap_.begin(); itM!=histosMap_.end(); itM++){
    for(map<string,MonitorElement*>::iterator itMM=itM->second["run"].begin(); itMM!=itM->second["run"].end(); itMM++){
      name = string("h") + itM->first + itMM->first;
      title = itM->first + "_{0} " + itMM->first;
      if(itM->first == "x" || itM->first == "y"){
        if(itMM->first == "Coordinate"){
          itMM->second = dbe_->book1D(name,title,1001,-0.2525,0.2525);
	}
	else if(itMM->first == "PrimaryVertex fit-DataBase" || itMM->first == "PrimaryVertex fit-BeamFit" || itMM->first == "PrimaryVertex fit-Scalers"
	     || itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
          itMM->second = dbe_->book1D(name,title,1001,-0.02525,0.02525);
	}
	else{
	  //assert(0);
	}
      }
      else if(itM->first == "z"){
        if(itMM->first == "Coordinate"){
          itMM->second = dbe_->book1D(name,title,101,-5.05,5.05);
	}
	else if(itMM->first == "PrimaryVertex fit-DataBase" || itMM->first == "PrimaryVertex fit-BeamFit" || itMM->first == "PrimaryVertex fit-Scalers"){
          itMM->second = dbe_->book1D(name,title,101,-0.505,0.505);
	}
	else if(itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
          itMM->second = dbe_->book1D(name,title,1001,-5.005,5.005);
	}
	else{
	  //assert(0);
	}
      }
      else if(itM->first == "sigmaX" || itM->first == "sigmaY"){
        if(itMM->first == "Coordinate"){
          itMM->second = dbe_->book1D(name,title,100,0,0.015);
	}
	else if(itMM->first == "PrimaryVertex fit-DataBase" || itMM->first == "PrimaryVertex fit-BeamFit" || itMM->first == "PrimaryVertex fit-Scalers"
	     || itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
          itMM->second = 0;
	}
	else{
	  //assert(0);
	}
      }
      else if(itM->first == "sigmaZ"){
        if(itMM->first == "Coordinate"){
          itMM->second = dbe_->book1D(name,title,110,0,11);
	}
	else if(itMM->first == "PrimaryVertex fit-DataBase" || itMM->first == "PrimaryVertex fit-BeamFit" || itMM->first == "PrimaryVertex fit-Scalers"
	     || itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
          itMM->second = dbe_->book1D(name,title,101,-5.05,5.05);
	}
	else{
	  //assert(0);
	}
      }
      else{
        //assert(0);
      }
      if(itMM->second != 0){
      	if(itMM->first == "Coordinate"){				
      	  itMM->second->setAxisTitle(itM->first + "_{0} (cm)",1);  
      	}
      	else if(itMM->first == "PrimaryVertex fit-DataBase" || itMM->first == "PrimaryVertex fit-BeamFit" || itMM->first == "PrimaryVertex fit-Scalers"
	     || itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
      	  itMM->second->setAxisTitle(itMM->first + " " + itM->first + "_{0} (cm)",1);  
      	}
      	itMM->second->setAxisTitle("Entries",2);
      }		
    }
  }
  dbe_->setCurrentFolder(monitorName_+"Service");
  theValuesContainer_ = dbe_->bookProfile("hHistoLumiValues","Histo Lumi Values", 3*numberOfValuesToSave_, 0., 3*numberOfValuesToSave_, 100., -100., 9000., " ");
  theValuesContainer_->setLumiFlag();

}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::beginRun(const edm::Run& r, const EventSetup& context) {
  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Validation");
  //Book histograms
  hD0Phi0_ = dbe_->bookProfile("hD0Phi0","d_{0} vs. #phi_{0} (All Tracks)",63,-3.15,3.15,100,-0.1,0.1,"");
  hD0Phi0_->setAxisTitle("#phi_{0} (rad)",1);
  hD0Phi0_->setAxisTitle("d_{0} (cm)",2);

  dbe_->setCurrentFolder(monitorName_+"Debug");
  hDxyBS_ = dbe_->book1D("hDxyBS","dxy_{0} w.r.t. Beam spot (All Tracks)",100,-0.1,0.1);
  hDxyBS_->setAxisTitle("dxy_{0} w.r.t. Beam spot (cm)",1);
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::beginLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  // Always create a beamspot group for each lumi weather we have results or not! Each Beamspot will be of unknown type!
  
  vertices_.clear();
  theValuesContainer_->Reset();
  beamSpotsMap_.clear();
  
  //Read BeamSpot from DB
  ESHandle<BeamSpotObjects> bsDBHandle;
  try{
    iSetup.get<BeamSpotObjectsRcd>().get(bsDBHandle);
  }
  catch( cms::Exception& exception ){				      
    LogInfo("AlcaBeamMonitor") 
      << exception.what(); 
    return;	      
  }				      
  if(bsDBHandle.isValid()) { // check the product
    const BeamSpotObjects *spotDB = bsDBHandle.product();

    // translate from BeamSpotObjects to reco::BeamSpot
    BeamSpot::Point apoint( spotDB->GetX(), spotDB->GetY(), spotDB->GetZ() );
  
    BeamSpot::CovarianceMatrix matrix;
    for ( int i=0; i<7; ++i ) {
      for ( int j=0; j<7; ++j ) {
  	matrix(i,j) = spotDB->GetCovariance(i,j);
      }
    }
  
    beamSpotsMap_["DB"] = BeamSpot( apoint,
  				    spotDB->GetSigmaZ(),
  				    spotDB->Getdxdz(),
  				    spotDB->Getdydz(),
  				    spotDB->GetBeamWidthX(),
  				    matrix );

    BeamSpot* aSpot = &(beamSpotsMap_["DB"]);

    aSpot->setBeamWidthY( spotDB->GetBeamWidthY() );
    aSpot->setEmittanceX( spotDB->GetEmittanceX() );
    aSpot->setEmittanceY( spotDB->GetEmittanceY() );
    aSpot->setbetaStar( spotDB->GetBetaStar() );

    if ( spotDB->GetBeamType() == 2 ) {
      aSpot->setType( reco::BeamSpot::Tracker );
    } else{
      aSpot->setType( reco::BeamSpot::Fake );
    }
    //LogInfo("AlcaBeamMonitor")
    //  << *aSpot << std::endl;
  }
  else {
    LogInfo("AlcaBeamMonitor") 
      << "Database BeamSpot is not valid at lumi: " << iLumi.id().luminosityBlock(); 
  }
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::analyze(const Event& iEvent, const EventSetup& iSetup ){
  
  //------ BeamFitter 
  theBeamFitter_->readEvent(iEvent);
  //------ PVFitter 
  thePVFitter_->readEvent(iEvent);
  
  if(beamSpotsMap_.find("DB") != beamSpotsMap_.end()){
    //------ Tracks
    Handle<reco::TrackCollection> TrackCollection;
    iEvent.getByToken(trackLabel_, TrackCollection);
    const reco::TrackCollection *tracks = TrackCollection.product();
    for ( reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); ++track ) {    
      hD0Phi0_->Fill(track->phi(), -1*track->dxy());
      hDxyBS_->Fill(-1*track->dxy(beamSpotsMap_["DB"].position()));
    }
  }
  
  //------ Primary Vertices
  Handle<VertexCollection > PVCollection;
  if (iEvent.getByToken(primaryVertexLabel_, PVCollection )) {
    vertices_.push_back(*PVCollection.product());
  }

  if(beamSpotsMap_.find("SC") == beamSpotsMap_.end()){
    //BeamSpot from file for this stream is = to the scalar BeamSpot
    Handle<BeamSpot> recoBeamSpotHandle;
    try{
      iEvent.getByToken(scalerLabel_,recoBeamSpotHandle);
    }
    catch( cms::Exception& exception ){ 			      
      LogInfo("AlcaBeamMonitor") 
      << exception.what(); 
      return;	      
    }				      
    beamSpotsMap_["SC"] = *recoBeamSpotHandle;
    if ( beamSpotsMap_["SC"].BeamWidthX() != 0 ) {
      beamSpotsMap_["SC"].setType( reco::BeamSpot::Tracker );
    } else{
      beamSpotsMap_["SC"].setType( reco::BeamSpot::Fake );
    }
  }
}


//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::endLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  if (theBeamFitter_->runPVandTrkFitter()) {
    beamSpotsMap_["BF"] = theBeamFitter_->getBeamSpot();
  }
  theBeamFitter_->resetTrkVector();
  theBeamFitter_->resetLSRange();
  theBeamFitter_->resetRefTime();
  theBeamFitter_->resetPVFitter();

  if ( thePVFitter_->runFitter() ) {
    beamSpotsMap_["PV"] = thePVFitter_->getBeamSpot();
  }
  thePVFitter_->resetAll();

  //    "PV,BF..."      Value,Error
  map<std::string,pair<double,double> >   resultsMap;
  vector<pair<double,double> >  vertexResults;
  MonitorElement* histo=0;
  int position = 0;
  for(vector<string>::iterator itV=varNamesV_.begin(); itV!=varNamesV_.end(); itV++){
    resultsMap.clear();
    for(BeamSpotContainer::iterator itBS = beamSpotsMap_.begin(); itBS != beamSpotsMap_.end(); itBS++){
      if(itBS->second.type() == BeamSpot::Tracker){
    	if(*itV == "x"){
    	  resultsMap[itBS->first] = pair<double,double>(itBS->second.x0(),itBS->second.x0Error());
    	}
    	else if(*itV == "y"){
    	  resultsMap[itBS->first] = pair<double,double>(itBS->second.y0(),itBS->second.y0Error());
    	}
    	else if(*itV == "z"){
    	  resultsMap[itBS->first] = pair<double,double>(itBS->second.z0(),itBS->second.z0Error());
    	}
    	else if(*itV == "sigmaX"){
    	  resultsMap[itBS->first] = pair<double,double>(itBS->second.BeamWidthX(),itBS->second.BeamWidthXError());
    	}
    	else if(*itV == "sigmaY"){
    	  resultsMap[itBS->first] = pair<double,double>(itBS->second.BeamWidthY(),itBS->second.BeamWidthYError());
    	}
    	else if(*itV == "sigmaZ"){
    	  resultsMap[itBS->first] = pair<double,double>(itBS->second.sigmaZ(),itBS->second.sigmaZ0Error());
    	}
    	else{
    	  LogInfo("AlcaBeamMonitor")
    	    << "The histosMap_ has been built with the name " << *itV << " that I can't recognize!";
    	  //assert(0);
    	}
      }
    }
    vertexResults.clear();
    for(vector<VertexCollection>::iterator itPV = vertices_.begin(); itPV != vertices_.end(); itPV++){
      if(itPV->size() != 0){
    	for(VertexCollection::const_iterator pv = itPV->begin(); pv != itPV->end(); pv++) {
    	  if (pv->isFake() || pv->tracksSize()<10)  continue;
    	  if(*itV == "x"){										      
    	    vertexResults.push_back(pair<double,double>(pv->x(),pv->xError()));       
    	  }													      
    	  else if(*itV == "y"){ 									      
    	    vertexResults.push_back(pair<double,double>(pv->y(),pv->yError()));       
    	  }													      
    	  else if(*itV == "z"){ 									      
    	    vertexResults.push_back(pair<double,double>(pv->z(),pv->zError()));       
    	  }													      
    	  else if(*itV != "sigmaX" && *itV != "sigmaY" && *itV != "sigmaZ"){		      
    	    LogInfo("AlcaBeamMonitor")  									      
    	      << "The histosMap_ has been built with the name " << *itV << " that I can't recognize!";
    	    //assert(0);  											      
    	  }													      
    	}
      }
    }
/*
  histoByCategoryNames_.insert( pair<string,string>("run",        "Coordinate"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-BeamFit"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-Scalers"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-BeamFit"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-Scalers"));

  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased BeamSpotFit"));  
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased PrimaryVertex"));
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased DataBase"));     
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased Scalers"));      
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased PrimaryVertex-DataBase fit"));
  histoByCategoryNames_.insert( pair<string,string>("lumi",       "Lumibased PrimaryVertex-Scalers fit"));
  histoByCategoryNames_.insert( pair<string,string>("validation", "Lumibased Scalers-DataBase fit"));
  histoByCategoryNames_.insert( pair<string,string>("validation", "Lumibased PrimaryVertex-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("validation", "Lumibased PrimaryVertex-Scalers"));
*/
    for(multimap<string,string>::iterator itM=histoByCategoryNames_.begin(); itM!=histoByCategoryNames_.end(); itM++){
      if(itM->first == "run" && (histo = histosMap_[*itV][itM->first][itM->second]) == 0){
        continue;
      }
      else if(itM->first != "run"){
        position = positionsMap_[*itV][itM->first][itM->second];
      }
      if(itM->second == "Coordinate"){
        if(beamSpotsMap_.find("DB") != beamSpotsMap_.end()){
          histo->Fill(resultsMap["DB"].first);
        }
      }
      else if(itM->second == "PrimaryVertex fit-DataBase"){
        if(resultsMap.find("PV") != resultsMap.end() && resultsMap.find("DB") != resultsMap.end()){
          histo->Fill(resultsMap["PV"].first-resultsMap["DB"].first);
        }
      }
      else if(itM->second == "PrimaryVertex fit-BeamFit"){
        if(resultsMap.find("PV") != resultsMap.end() && resultsMap.find("BF") != resultsMap.end()){
          histo->Fill(resultsMap["PV"].first-resultsMap["BF"].first);
        }
      }
      else if(itM->second == "PrimaryVertex fit-Scalers"){
        if(resultsMap.find("PV") != resultsMap.end() && resultsMap.find("SC") != resultsMap.end()){
          histo->Fill(resultsMap["PV"].first-resultsMap["SC"].first);
        }
      }
      else if(itM->second == "PrimaryVertex-DataBase"){
        if(resultsMap.find("PV") != resultsMap.end() && resultsMap.find("DB") != resultsMap.end()){
          for(vector<pair<double,double> >::iterator itPV=vertexResults.begin(); itPV!=vertexResults.end(); itPV++){
            histo->Fill(itPV->first-resultsMap["DB"].first);
          }
        }
      }
      else if(itM->second == "PrimaryVertex-BeamFit"){
        if(resultsMap.find("PV") != resultsMap.end() && resultsMap.find("BF") != resultsMap.end()){
          for(vector<pair<double,double> >::iterator itPV=vertexResults.begin(); itPV!=vertexResults.end(); itPV++){
            histo->Fill(itPV->first-resultsMap["BF"].first);
          }
        }
      }
      else if(itM->second == "PrimaryVertex-Scalers"){
        if(resultsMap.find("PV") != resultsMap.end() && resultsMap.find("SC") != resultsMap.end()){
          for(vector<pair<double,double> >::iterator itPV=vertexResults.begin(); itPV!=vertexResults.end(); itPV++){
            histo->Fill(itPV->first-resultsMap["SC"].first);
          }
        }
      }
      else if(itM->second == "Lumibased BeamSpotFit"){
        if(resultsMap.find("BF") != resultsMap.end()){
          theValuesContainer_->Fill(position  ,resultsMap["BF"].first);//Value
          theValuesContainer_->Fill(position+1,resultsMap["BF"].second);//Error
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
      else if(itM->second == "Lumibased PrimaryVertex"){
        if(resultsMap.find("PV") != resultsMap.end()){
          theValuesContainer_->Fill(position  ,resultsMap["PV"].first);//Value
          theValuesContainer_->Fill(position+1,resultsMap["PV"].second);//Error
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
      else if(itM->second == "Lumibased DataBase"){
        if(resultsMap.find("DB") != resultsMap.end()){
          theValuesContainer_->Fill(position  ,resultsMap["DB"].first);//Value
          theValuesContainer_->Fill(position+1,resultsMap["DB"].second);//Error		  
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
      else if(itM->second == "Lumibased Scalers"){
        if(resultsMap.find("SC") != resultsMap.end()){
          theValuesContainer_->Fill(position  ,resultsMap["SC"].first);//Value
          theValuesContainer_->Fill(position+1,resultsMap["SC"].second);//Error		  
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
      else if(itM->second == "Lumibased PrimaryVertex-DataBase fit"){
        if(resultsMap.find("PV") != resultsMap.end() && resultsMap.find("DB") != resultsMap.end()){
          theValuesContainer_->Fill(position  ,resultsMap["PV"].first-resultsMap["DB"].first);//Value
          theValuesContainer_->Fill(position+1,std::sqrt(std::pow(resultsMap["PV"].second,2)+std::pow(resultsMap["DB"].second,2)));//Error	  
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
      else if(itM->second == "Lumibased PrimaryVertex-Scalers fit"){
        if(resultsMap.find("PV") != resultsMap.end() && resultsMap.find("SC") != resultsMap.end()){
          theValuesContainer_->Fill(position  ,resultsMap["PV"].first-resultsMap["SC"].first);//Value
          theValuesContainer_->Fill(position+1,std::sqrt(std::pow(resultsMap["PV"].second,2)+std::pow(resultsMap["SC"].second,2)));//Error	  
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
      else if(itM->second == "Lumibased Scalers-DataBase fit"){
        if(resultsMap.find("SC") != resultsMap.end() && resultsMap.find("DB") != resultsMap.end()){
          theValuesContainer_->Fill(position  ,resultsMap["SC"].first-resultsMap["DB"].first);//Value
          theValuesContainer_->Fill(position+1,std::sqrt(std::pow(resultsMap["SC"].second,2)+std::pow(resultsMap["DB"].second,2)));//Error	  
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
      else if(itM->second == "Lumibased PrimaryVertex-DataBase"){
        if(resultsMap.find("DB") != resultsMap.end() && vertexResults.size() != 0){
	  for(vector<pair<double,double> >::iterator itPV=vertexResults.begin(); itPV!=vertexResults.end(); itPV++){
            theValuesContainer_->Fill(position  ,(*itPV).first-resultsMap["DB"].first);//Value
          }
/*
          double error = 0;
	  if(vertexResults.size() != 0){
	    for(vector<pair<double,double> >::iterator itPV=vertexResults.begin(); itPV!=vertexResults.end(); itPV++){
              error += std::pow((*itPV).first-resultsMap["DB"].first-theValuesContainer_->getTProfile()->GetBinContent(position+1),2.);
            }
	    error = std::sqrt(error)/vertexResults.size();
	  }
//          theValuesContainer_->Fill(position+1,std::sqrt(std::pow((*itPV).second,2)+std::pow(resultsMap["DB"].second,2)));//Error	  
          theValuesContainer_->Fill(position+1,error);//Error	  
*/
          theValuesContainer_->Fill(position+1,theValuesContainer_->getTProfile()->GetBinError(position+1));//Error	  
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
      else if(itM->second == "Lumibased PrimaryVertex-Scalers"){
        if(resultsMap.find("SC") != resultsMap.end() && vertexResults.size() != 0){
          for(vector<pair<double,double> >::iterator itPV=vertexResults.begin(); itPV!=vertexResults.end(); itPV++){
            theValuesContainer_->Fill(position  ,(*itPV).first-resultsMap["SC"].first);//Value
          }
/*
          double error = 0;
	  if(vertexResults.size() != 0){
	    for(vector<pair<double,double> >::iterator itPV=vertexResults.begin(); itPV!=vertexResults.end(); itPV++){
              error += std::pow((*itPV).first-resultsMap["SC"].first-theValuesContainer_->getTProfile()->GetBinContent(position+1),2.);
            }
	    error = std::sqrt(error)/vertexResults.size();
	  }
//          theValuesContainer_->Fill(position+1,std::sqrt(std::pow((*itPV).second,2)+std::pow(resultsMap["SC"].second,2)));//Error	  
          theValuesContainer_->Fill(position+1,error);//Error	  
*/
          theValuesContainer_->Fill(position+1,theValuesContainer_->getTProfile()->GetBinError(position+1));//Error	  
          theValuesContainer_->Fill(position+2,1);//ok
        }
      }
//      else if(itM->second == "Lumibased Scalers-DataBase"){
//      if(resultsMap.find("SC") != resultsMap.end() && resultsMap.find("DB") != resultsMap.end()){
//        itHHH->second->Fill(bin,resultsMap["SC"].first-resultsMap["DB"].first);
//      }
//    }
      else{
        LogInfo("AlcaBeamMonitor")
          << "The histosMap_ have a histogram named " << itM->second << " that I can't recognize in this loop!";
        //assert(0);

      }
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::endRun(const Run& iRun, const EventSetup& context){
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::endJob(const LuminosityBlock& iLumi, const EventSetup& iSetup){
}


DEFINE_FWK_MODULE(AlcaBeamMonitor);

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
