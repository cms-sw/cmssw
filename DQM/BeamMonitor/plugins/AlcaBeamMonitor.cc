/*
 * \file AlcaBeamMonitor.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 * $Date: 2010/08/11 21:58:52 $
 * $Revision: 1.1 $
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include "RecoVertex/BeamSpotProducer/interface/PVFitter.h"
#include "DQM/BeamMonitor/plugins/AlcaBeamMonitor.h"
#include "DQMServices/Core/interface/QReport.h"
#include <numeric>
//#include <iostream>

using namespace std;
using namespace edm;
using namespace reco;

//----------------------------------------------------------------------------------------------------------------------
AlcaBeamMonitor::AlcaBeamMonitor( const ParameterSet& ps ){

  parameters_         = ps;
  monitorName_        = parameters_.getUntrackedParameter<string>("MonitorName","YourSubsystemName");
  primaryVertexLabel_ = parameters_.getUntrackedParameter<InputTag>("PrimaryVertexLabel");
  beamSpotLabel_      = parameters_.getUntrackedParameter<InputTag>("BeamSpotLabel");
  trackLabel_         = parameters_.getUntrackedParameter<InputTag>("TrackLabel");
  scalerLabel_        = parameters_.getUntrackedParameter<InputTag>("ScalerLabel");
//  min_Ntrks_      = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumInputTracks");
//  maxZ_           = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumZ");
//  minNrVertices_  = parameters_.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<unsigned int>("minNrVerticesForFit");
//  minVtxNdf_      = parameters_.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexNdf");
//  minVtxWgt_      = parameters_.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexMeanWeight");

  dbe_            = Service<DQMStore>().operator->();
  
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  
  theBeamFitter_ = new BeamFitter(parameters_);
  theBeamFitter_->resetTrkVector();
  theBeamFitter_->resetLSRange();
  theBeamFitter_->resetRefTime();
  theBeamFitter_->resetPVFitter();

  thePVFitter_ = new PVFitter(parameters_);


  varNamesV_.push_back("x");
  varNamesV_.push_back("y");
  varNamesV_.push_back("z");
  varNamesV_.push_back("sigmaX");
  varNamesV_.push_back("sigmaY");
  varNamesV_.push_back("sigmaZ");

  histoByCategoryNames_.insert( pair<string,string>("run",     "Coordinate"));
  histoByCategoryNames_.insert( pair<string,string>("run",     "PrimaryVertex-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("run",     "PrimaryVertex-BeamFit"));
  histoByCategoryNames_.insert( pair<string,string>("run",     "PrimaryVertex-Scalers"));
  histoByCategoryNames_.insert( pair<string,string>("lumi",    "Lumibased BeamSpotFit"));  
  histoByCategoryNames_.insert( pair<string,string>("lumi",    "Lumibased PrimaryVertex"));
  histoByCategoryNames_.insert( pair<string,string>("lumi",    "Lumibased DataBase"));     
  histoByCategoryNames_.insert( pair<string,string>("lumi",    "Lumibased Scalers"));	   
  histoByCategoryNames_.insert( pair<string,string>("lumi",    "Lumibased PrimaryVertex-DataBase fit"));
  histoByCategoryNames_.insert( pair<string,string>("lumi",    "Lumibased PrimaryVertex-Scalers fit"));
  histoByCategoryNames_.insert( pair<string,string>("lumi",    "Lumibased Scalers-DataBase fit"));
  histoByCategoryNames_.insert( pair<string,string>("profile", "Lumibased PrimaryVertex-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("profile", "Lumibased PrimaryVertex-Scalers"));
  histoByCategoryNames_.insert( pair<string,string>("profile", "Lumibased Scalers-DataBase"));


  for(vector<string>::iterator itV=varNamesV_.begin(); itV!=varNamesV_.end(); itV++){
    for(multimap<string,string>::iterator itM=histoByCategoryNames_.begin(); itM!=histoByCategoryNames_.end(); itM++){
      histosMap_[*itV][itM->first][itM->second] = 0;
    }
  }
  
  beamSpotsMap_["BF"] = map<LuminosityBlockNumber_t,BeamSpot>();//For each lumi the beamfitter will have a result
  beamSpotsMap_["PV"] = map<LuminosityBlockNumber_t,BeamSpot>();//For each lumi the PVfitter will have a result
  beamSpotsMap_["DB"] = map<LuminosityBlockNumber_t,BeamSpot>();//For each lumi we take the values that are stored in the database, already collapsed then
  beamSpotsMap_["SC"] = map<LuminosityBlockNumber_t,BeamSpot>();//For each lumi we take the beamspot value in the file that is the same as the scaler for the alca reco stream
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
	else if(itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
          itMM->second = dbe_->book1D(name,title,1001,-0.02525,0.02525);
	}
	else{
	  assert(0);
	}
      }
      else if(itM->first == "z"){
        if(itMM->first == "Coordinate"){
          itMM->second = dbe_->book1D(name,title,101,-5.05,5.05);
	}
	else if(itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
          itMM->second = dbe_->book1D(name,title,101,-0.505,0.505);
	}
	else{
	  assert(0);
	}
      }
      else if(itM->first == "sigmaX" || itM->first == "sigmaY"){
        if(itMM->first == "Coordinate"){
          itMM->second = dbe_->book1D(name,title,100,0,0.015);
	}
	else if(itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
          itMM->second = 0;
	}
	else{
	  assert(0);
	}
      }
      else if(itM->first == "sigmaZ"){
        if(itMM->first == "Coordinate"){
          itMM->second = dbe_->book1D(name,title,110,0,11);
	}
	else if(itMM->first == "PrimaryVertex-DataBase" || itMM->first == "PrimaryVertex-BeamFit" || itMM->first == "PrimaryVertex-Scalers"){
          itMM->second = dbe_->book1D(name,title,101,-5.05,5.05);
	}
	else{
	  assert(0);
	}
      }
      else{
        assert(0);
      }
      if(itMM->second != 0){
      	if(itMM->first == "Coordinate"){				
      	  itMM->second->setAxisTitle(itM->first + "_{0} (cm)",1);  
      	}
      	else if(itMM->first == "PrimaryVertex-DataBase"){
      	  itMM->second->setAxisTitle(string("PrimaryVertex-Database") + itM->first + "_{0} (cm)",1);  
      	}
      	else if(itMM->first == "PrimaryVertex-BeamFit"){
      	  itMM->second->setAxisTitle(string("PrimaryVertex-BeamFit") + itM->first + "_{0} (cm)",1);  
      	}
      	else if(itMM->first == "PrimaryVertex-Scalers"){
      	  itMM->second->setAxisTitle(string("PrimaryVertex-Scalers") + itM->first + "_{0} (cm)",1);  
      	}
      	itMM->second->setAxisTitle("Entries",2);
      }		
    }
  }
  /*
  h_x0=dbe_->book1D("h_x0","X0 position",100,-0.1,0.1);
  h_x0->setAxisTitle("x_{0} (cm)",1);
  h_x0->setAxisTitle("Entries",2);
  
  h_x0_delta_DB_PV=dbe_->book1D("h_x0_delta_DB_PV","DB-PV Beamspot X0 position",100,-0.0025,0.0025);
  h_x0_delta_DB_PV->setAxisTitle("Database-PrimaryVertex x_{0} (cm)",1);
  h_x0_delta_DB_PV->setAxisTitle("Entries",2);
  */
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::beginRun(const edm::Run& r, const EventSetup& context) {
  //Resetting variables
  firstLumi_ = -1;
  lastLumi_  = -1;
  numberOfLumis_ = 0;
  verticesMap_.clear();
  for(BeamSpotContainer::iterator it = beamSpotsMap_.begin(); it != beamSpotsMap_.end(); it++){
    it->second.clear(); 
  }
  for(HistosContainer::iterator itM=histosMap_.begin(); itM!=histosMap_.end(); itM++){
    for(map<string,map<string,MonitorElement*> >::iterator itMM=itM->second.begin(); itMM!=itM->second.end(); itMM++){
      if(itMM->first != "run"){
      	for(map<string,MonitorElement*>::iterator itMMM=itMM->second.begin(); itMMM!=itMM->second.end(); itMMM++){
      	  if( itMMM->second != 0){
      	    dbe_->removeElement(monitorName_+"Validation",itMMM->second->getName());
      	    itMMM->second = 0;
      	  }
      	}
      }
    }
  }

  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Validation");
  //Book histograms
  h_d0_phi0 = dbe_->bookProfile("d0_phi0","d_{0} vs. #phi_{0} (Selected Tracks)",63,-3.15,3.15,100,-0.1,0.1,"");
  h_d0_phi0->setAxisTitle("#phi_{0} (rad)",1);
  h_d0_phi0->setAxisTitle("d_{0} (cm)",2);
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::beginLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  // Always create a beamspot group for each lumi weather we have results or not! Each Beamspot will be of unknown type!
  
  //Read BeamSpot from DB
  ESHandle<BeamSpotObjects> bsDBHandle;
  try{
    iSetup.get<BeamSpotObjectsRcd>().get(bsDBHandle);
  }
  catch( cms::Exception& exception ){				      
    LogInfo("AlcaBeamMonitor") 
      << exception.what() << endl; 
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
  
    beamSpotsMap_["DB"][iLumi.luminosityBlock()] = BeamSpot( apoint,
  						  	     spotDB->GetSigmaZ(),
  							     spotDB->Getdxdz(),
  							     spotDB->Getdydz(),
  							     spotDB->GetBeamWidthX(),
  							     matrix );

    BeamSpot* aSpot = &(beamSpotsMap_["DB"][iLumi.luminosityBlock()]);

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
      << "Database BeamSpot is not valid at lumi: " << iLumi.id().luminosityBlock() 
      << endl;
  }
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::analyze(const Event& iEvent, const EventSetup& iSetup ){
  
  //------ BeamFitter 
  theBeamFitter_->readEvent(iEvent);
  //------ PVFitter 
  thePVFitter_->readEvent(iEvent);
  
  //------ Tracks
  Handle<reco::TrackCollection> TrackCollection;
  iEvent.getByLabel(trackLabel_, TrackCollection);
  const reco::TrackCollection *tracks = TrackCollection.product();
  for ( reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); ++track ) {    
    h_d0_phi0->Fill(track->phi(), -1*track->dxy(beamSpotsMap_["DB"][iEvent.luminosityBlock()].position()));
  }
  
  //------ Primary Vertices
  Handle<VertexCollection > PVCollection;
  if (iEvent.getByLabel(primaryVertexLabel_, PVCollection )) {
    if(verticesMap_.find(iEvent.luminosityBlock()) == verticesMap_.end()){
      verticesMap_[iEvent.luminosityBlock()] = vector<VertexCollection>();
    }
    verticesMap_[iEvent.luminosityBlock()].push_back(*PVCollection.product());
  }

  if(beamSpotsMap_["SC"].find(iEvent.luminosityBlock()) == beamSpotsMap_["SC"].end()){
    //BeamSpot from file for this stream is = to the scalar BeamSpot
    Handle<BeamSpot> recoBeamSpotHandle;
    try{
      iEvent.getByLabel(beamSpotLabel_,recoBeamSpotHandle);
    }
    catch( cms::Exception& exception ){ 			      
      LogInfo("AlcaBeamMonitor") 
      << exception.what() << endl; 
      return;	      
    }				      
    beamSpotsMap_["SC"][iEvent.luminosityBlock()] = *recoBeamSpotHandle;
    if ( beamSpotsMap_["SC"][iEvent.luminosityBlock()].BeamWidthX() != 0 ) {
      beamSpotsMap_["SC"][iEvent.luminosityBlock()].setType( reco::BeamSpot::Tracker );
    } else{
      beamSpotsMap_["SC"][iEvent.luminosityBlock()].setType( reco::BeamSpot::Fake );
    }
//    LogInfo("AlcaBeamMonitor")
//      << beamSpotsMap_["SC"][iEvent.luminosityBlock()];
/*
    double theSetSigmaZ = 10;
    double theMaxZ = 40;
    double theMaxR2 = 10;
    
    
    // get scalar collection
    Handle<BeamSpotOnlineCollection> handleScaler;
    try{
      iEvent.getByLabel( scalerLabel_, handleScaler);
    }
    catch( cms::Exception& exception ){ 			      
      LogInfo("AlcaBeamMonitor") 
      << exception.what() << endl; 
      return;	      
    }				      

    // beam spot scalar object
    BeamSpotOnline spotOnline;
    
    BeamSpot* aSpot = 0;

    if (handleScaler->size()!=0){
      // get one element
      spotOnline = * ( handleScaler->begin() );
      
      // in case we need to switch to LHC reference frame
      // ignore for the moment rotations, and translations
      double f = 1.;
//      if (changeFrame_) f = -1.;
      
      BeamSpot::Point apoint( f* spotOnline.x(), spotOnline.y(), f* spotOnline.z() );
      
      BeamSpot::CovarianceMatrix matrix;
      matrix(0,0) = spotOnline.err_x()*spotOnline.err_x();
      matrix(1,1) = spotOnline.err_y()*spotOnline.err_y();
      matrix(2,2) = spotOnline.err_z()*spotOnline.err_z();
      matrix(3,3) = spotOnline.err_sigma_z()*spotOnline.err_sigma_z();
      
      double sigmaZ = spotOnline.sigma_z();
      if (theSetSigmaZ>0)
    	sigmaZ = theSetSigmaZ;
      
      beamSpotsMap_["SC"][iEvent.luminosityBlock()] = BeamSpot( apoint,
    			      					sigmaZ,
    			      					spotOnline.dxdz(),
    			      					f* spotOnline.dydz(),
    			      					spotOnline.width_x(),
    			      					matrix);
      
      aSpot = &beamSpotsMap_["SC"][iEvent.luminosityBlock()];
      aSpot->setBeamWidthY( spotOnline.width_y() );
      aSpot->setEmittanceX( 0. );
      aSpot->setEmittanceY( 0. );
      aSpot->setbetaStar( 0.) ;
      aSpot->setType( reco::BeamSpot::LHC ); // flag value from scalars
      
      // check if we have a valid beam spot fit result from online DQM
      if ( spotOnline.x() == 0 &&
    	   spotOnline.y() == 0 &&
    	   spotOnline.z() == 0 &&
    	   spotOnline.width_x() == 0 &&
    	   spotOnline.width_y() == 0 ) 
    	{
          LogInfo("AlcaBeamMonitor") 
    	    << "Online Beam Spot producer falls back to DB value because the scaler values are zero ";
          aSpot->setType( reco::BeamSpot::Unknown ); // flag value from scalars
    	}
      double r2=spotOnline.x()*spotOnline.x() + spotOnline.y()*spotOnline.y();
      if (fabs(spotOnline.z())>=theMaxZ || r2>=theMaxR2){
        LogInfo("AlcaBeamMonitor") 
    	  << "Online Beam Spot producer falls back to DB value because the scaler values are too big to be true :"
    	  <<spotOnline.x()<<" "<<spotOnline.y()<<" "<<spotOnline.z();
        aSpot->setType( reco::BeamSpot::Unknown ); // flag value from scalars
      }
    }
    else{
      //empty online beamspot collection: FED data was empty
      //the error should probably have been send at unpacker level
      aSpot->setType( reco::BeamSpot::Unknown ); // flag value from scalars
    }
*/
  }
}


//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::endLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  ++numberOfLumis_;
  if(firstLumi_== -1 || iLumi.id().luminosityBlock() < (unsigned int)firstLumi_){
    firstLumi_ = iLumi.id().luminosityBlock();
  }
  if(lastLumi_ == -1 || iLumi.id().luminosityBlock() > (unsigned int)lastLumi_){
    lastLumi_ = iLumi.id().luminosityBlock();
  }
  
  if (theBeamFitter_->runPVandTrkFitter()) {
    beamSpotsMap_["BF"][iLumi.luminosityBlock()] = theBeamFitter_->getBeamSpot();
  }
  theBeamFitter_ = new BeamFitter(parameters_);
  theBeamFitter_->resetTrkVector();
  theBeamFitter_->resetLSRange();
  theBeamFitter_->resetRefTime();
  theBeamFitter_->resetPVFitter();

  if ( thePVFitter_->runFitter() ) {
    beamSpotsMap_["PV"][iLumi.luminosityBlock()] = thePVFitter_->getBeamSpot();
  }
  thePVFitter_->resetAll();
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::endRun(const Run& iRun, const EventSetup& context){
  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Validation");

  LogInfo("AlcaBeamMonitor") 
    << "End of run " << iRun.id().run() << "(" << firstLumi_ << "-" << lastLumi_ << ")" << endl;

  string name;
  string title;
  //x,y,z,sigmaX,sigmaY,sigmaZ
  for(HistosContainer::iterator itM=histosMap_.begin(); itM!=histosMap_.end(); itM++){
    for(map<string,map<string,MonitorElement*> >::iterator itMM=itM->second.begin(); itMM!=itM->second.end(); itMM++){
      if(itMM->first != "run"){
      	for(map<string,MonitorElement*>::iterator itMMM=itMM->second.begin(); itMMM!=itMM->second.end(); itMMM++){
      	  name = string("h") + itM->first + itMMM->first;
      	  title = itM->first + "_{0} " + itMMM->first;
      	  if(itMM->first == "lumi"){  
	    dbe_->setCurrentFolder(monitorName_+"Debug");
	    itMMM->second = dbe_->book1D(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5);
      	  }
	  else if(itMM->first == "profile" && (itM->first == "x" || itM->first == "y") ){
	    dbe_->setCurrentFolder(monitorName_+"Validation");
	    itMMM->second = dbe_->bookProfile(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5,-0.1,0.1,"");
	  }
	  else if(itMM->first == "profile" && itM->first == "z" ){
	    dbe_->setCurrentFolder(monitorName_+"Validation");
	    itMMM->second = dbe_->bookProfile(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5,-1,1,"");
	  }
	  else if(itMM->first == "profile" && (itM->first == "sigmaX" || itM->first == "sigmaY" || itM->first == "sigmaZ") ){
	    itMMM->second = 0;
	  }
	  else{
 	    LogInfo("AlcaBeamMonitor") 
 	      << "Unrecognized category " << itMM->first << endl;
	      assert(0);	    
	  }
	  if(itMMM->second != 0){
	    if(itMMM->first.find('-') != string::npos){ 				    
      	      itMMM->second->setAxisTitle(string("#Delta ") + itM->first + "_{0} (cm)",2);  
      	    }
      	    else{
      	      itMMM->second->setAxisTitle(itM->first + "_{0} (cm)",2);  
      	    }
      	    itMMM->second->setAxisTitle("Lumisection",1);
	  }
	}
      }		  
    }
  }
  
  //   x,y,z,sigmaX...      lumi                    "PV,BF..."      Value,Error
  map<std::string,map<LuminosityBlockNumber_t,map<std::string,pair<double,double> > > > resultsMap;
  //x,y,z,sigmaX,sigmaY,sigmaZ
  for(HistosContainer::iterator itH=histosMap_.begin(); itH!=histosMap_.end(); itH++){
    //BF,PV,DB,SC
    for(BeamSpotContainer::iterator itBS = beamSpotsMap_.begin(); itBS != beamSpotsMap_.end(); itBS++){
      for(map<LuminosityBlockNumber_t,BeamSpot>::iterator itBSBS = itBS->second.begin(); itBSBS != itBS->second.end(); itBSBS++){
        if(itBSBS->second.type() == BeamSpot::Tracker){
	  if(itH->first == "x"){
  	    //  	   "x"         "lumi"	    "PV"
	    resultsMap[itH->first][itBSBS->first][itBS->first] = pair<double,double>(itBSBS->second.x0(),itBSBS->second.x0Error());
          }
          else if(itH->first == "y"){
  	    resultsMap[itH->first][itBSBS->first][itBS->first] = pair<double,double>(itBSBS->second.y0(),itBSBS->second.y0Error());
          }
          else if(itH->first == "z"){
  	    resultsMap[itH->first][itBSBS->first][itBS->first] = pair<double,double>(itBSBS->second.z0(),itBSBS->second.z0Error());
          }
          else if(itH->first == "sigmaX"){
  	    resultsMap[itH->first][itBSBS->first][itBS->first] = pair<double,double>(itBSBS->second.BeamWidthX(),itBSBS->second.BeamWidthXError());
          }
          else if(itH->first == "sigmaY"){
  	    resultsMap[itH->first][itBSBS->first][itBS->first] = pair<double,double>(itBSBS->second.BeamWidthY(),itBSBS->second.BeamWidthYError());
          }
          else if(itH->first == "sigmaZ"){
  	    resultsMap[itH->first][itBSBS->first][itBS->first] = pair<double,double>(itBSBS->second.sigmaZ(),itBSBS->second.sigmaZ0Error());
          }
	  else{
            LogInfo("AlcaBeamMonitor")
	      << "The histosMap_ has been built with the name " << itH->first << " that I can't recognize!" << endl;
	    assert(0);
	  }
	}
      }
    }
  }  
  
 //   x,y,z,sigmaX...      lumi                               Value,Error
  map<string,map<LuminosityBlockNumber_t,vector<pair<double,double> > > > vertexResultsMap;
  //x,y,z,sigmaX,sigmaY,sigmaZ
  for(HistosContainer::iterator itH=histosMap_.begin(); itH!=histosMap_.end(); itH++){
    for(map<LuminosityBlockNumber_t,vector<VertexCollection> >::iterator itPV = verticesMap_.begin(); itPV != verticesMap_.end(); itPV++){
      if(itPV->second.size() != 0){
      	vertexResultsMap[itH->first][itPV->first] = vector<pair<double,double> >();
      	for(vector<VertexCollection>::iterator itPVV = itPV->second.begin(); itPVV != itPV->second.end(); itPVV++){
      	  for (VertexCollection::const_iterator pv = itPVV->begin(); pv != itPVV->end(); pv++) {
            if (pv->isFake() || pv->tracksSize()<10)  continue;
	    if(itH->first == "x"){											
  	      //	          "x"	     "lumi"     								
	      vertexResultsMap[itH->first][itPV->first].push_back(pair<double,double>(pv->x(),pv->xError()));		
      	    }														
      	    else if(itH->first == "y"){ 										
  	      vertexResultsMap[itH->first][itPV->first].push_back(pair<double,double>(pv->y(),pv->yError()));		
      	    }														
      	    else if(itH->first == "z"){ 										
  	      vertexResultsMap[itH->first][itPV->first].push_back(pair<double,double>(pv->z(),pv->zError()));		
      	    }														
	    else if(itH->first != "sigmaX" && itH->first != "sigmaY" && itH->first != "sigmaZ"){			
      	      LogInfo("AlcaBeamMonitor")										
		<< "The histosMap_ has been built with the name " << itH->first << " that I can't recognize!" << endl;
	      assert(0);												
	    }														
	  }
      	}
      }
    }
  }  

  unsigned int bin=0;
  for(HistosContainer::iterator itH=histosMap_.begin(); itH!=histosMap_.end(); itH++){
    for(map<string, map<string,MonitorElement*> >::iterator itHH=itH->second.begin(); itHH!=itH->second.end(); itHH++){
      for(map<string,MonitorElement*>::iterator itHHH=itHH->second.begin(); itHHH!=itHH->second.end(); itHHH++){
    	for(map<LuminosityBlockNumber_t,map<std::string,pair<double,double> > >::iterator itVal = resultsMap[itH->first].begin(); itVal != resultsMap[itH->first].end(); itVal++){
	  if(itHHH->second != 0){
	    bin = itHHH->second->getTH1()->FindBin(itVal->first);
	    if(itHHH->first == "Coordinate"){
	      if(itVal->second.find("DB") != itVal->second.end()){
	        itHHH->second->Fill(itVal->second["DB"].first);
              }
	    }
            else if(itHHH->first == "PrimaryVertex-DataBase"){
	      if(itVal->second.find("PV") != itVal->second.end() && itVal->second.find("DB") != itVal->second.end()){
	        itHHH->second->Fill(itVal->second["PV"].first-itVal->second["DB"].first);
	      }
	    }
            else if(itHHH->first == "PrimaryVertex-BeamFit"){
	      if(itVal->second.find("PV") != itVal->second.end() && itVal->second.find("BF") != itVal->second.end()){
	        itHHH->second->Fill(itVal->second["PV"].first-itVal->second["BF"].first);
	      }
	    }
            else if(itHHH->first == "PrimaryVertex-Scalers"){
	      if(itVal->second.find("PV") != itVal->second.end() && itVal->second.find("SC") != itVal->second.end()){
	        itHHH->second->Fill(itVal->second["PV"].first-itVal->second["SC"].first);
	      }
	    }
            else if(itHHH->first == "Lumibased BeamSpotFit"){
	      if(itVal->second.find("BF") != itVal->second.end()){
      	        itHHH->second->setBinContent(bin,itVal->second["BF"].first);
      	        itHHH->second->setBinError  (bin,itVal->second["BF"].second);
	      }
	    }
            else if(itHHH->first == "Lumibased PrimaryVertex"){
	      if(itVal->second.find("PV") != itVal->second.end()){
      	        itHHH->second->setBinContent(bin,itVal->second["PV"].first);
      	        itHHH->second->setBinError  (bin,itVal->second["PV"].second);
	      }
	    }
            else if(itHHH->first == "Lumibased DataBase"){
	      if(itVal->second.find("DB") != itVal->second.end()){
      	        itHHH->second->setBinContent(bin,itVal->second["DB"].first);
      	        itHHH->second->setBinError  (bin,itVal->second["DB"].second);
	      }
	    }
            else if(itHHH->first == "Lumibased Scalers"){
	      if(itVal->second.find("SC") != itVal->second.end()){
      	        itHHH->second->setBinContent(bin,itVal->second["SC"].first);
      	        itHHH->second->setBinError  (bin,itVal->second["SC"].second);
	      }
	    }
            else if(itHHH->first == "Lumibased PrimaryVertex-DataBase fit"){
	      if(itVal->second.find("PV") != itVal->second.end() && itVal->second.find("DB") != itVal->second.end()){
      	        itHHH->second->setBinContent(bin,itVal->second["PV"].first-itVal->second["DB"].first);
      	        itHHH->second->setBinError  (bin,sqrt(pow(itVal->second["PV"].second,2)+pow(itVal->second["DB"].second,2)));
	      }
	    }
            else if(itHHH->first == "Lumibased PrimaryVertex-Scalers fit"){
	      if(itVal->second.find("PV") != itVal->second.end() && itVal->second.find("SC") != itVal->second.end()){
      	        itHHH->second->setBinContent(bin,itVal->second["PV"].first-itVal->second["SC"].first);
      	        itHHH->second->setBinError  (bin,sqrt(pow(itVal->second["PV"].second,2)+pow(itVal->second["SC"].second,2)));
	      }
	    }
            else if(itHHH->first == "Lumibased Scalers-DataBase fit"){
	      if(itVal->second.find("SC") != itVal->second.end() && itVal->second.find("DB") != itVal->second.end()){
      	        itHHH->second->setBinContent(bin,itVal->second["SC"].first-itVal->second["DB"].first);
      	        itHHH->second->setBinError  (bin,sqrt(pow(itVal->second["SC"].second,2)+pow(itVal->second["DB"].second,2)));
	      }
	    }
            else if(itHHH->first == "Lumibased PrimaryVertex-DataBase"){
  	      if(itVal->second.find("DB") != itVal->second.end()){
        	if(vertexResultsMap.find(itH->first) != vertexResultsMap.end() && vertexResultsMap[itH->first].find(itVal->first) != vertexResultsMap[itH->first].end()){
		  for(vector<pair<double,double> >::iterator itV=vertexResultsMap[itH->first][itVal->first].begin(); itV!=vertexResultsMap[itH->first][itVal->first].end(); itV++){
		    itHHH->second->Fill(bin,itV->first-itVal->second["DB"].first);
		  }
		}
  	      }
	    }
            else if(itHHH->first == "Lumibased PrimaryVertex-Scalers"){
  	      if(itVal->second.find("SC") != itVal->second.end()){
        	if(vertexResultsMap.find(itH->first) != vertexResultsMap.end() && vertexResultsMap[itH->first].find(itVal->first) != vertexResultsMap[itH->first].end()){
		  for(vector<pair<double,double> >::iterator itV=vertexResultsMap[itH->first][itVal->first].begin(); itV!=vertexResultsMap[itH->first][itVal->first].end(); itV++){
		    itHHH->second->Fill(bin,itV->first-itVal->second["SC"].first);
		  }
		}
  	      }
	    }
            else if(itHHH->first == "Lumibased Scalers-DataBase"){
  	      if(itVal->second.find("SC") != itVal->second.end() && itVal->second.find("DB") != itVal->second.end()){
		itHHH->second->Fill(bin,itVal->second["SC"].first-itVal->second["DB"].first);
  	      }
//	      if(itVal->second.find("SC") != itVal->second.end() && itVal->second.find("DB") != itVal->second.end()){
//    	        itHHH->second->setBinContent(bin,itVal->second["SC"].first-itVal->second["DB"].first);
//    	        itHHH->second->setBinError  (bin,sqrt(pow(itVal->second["SC"].second,2)+pow(itVal->second["DB"].second,2))/sqrt(2));
//	      }
	    }
            else{
	      LogInfo("AlcaBeamMonitor")
	    	<< "The histosMap_ have a histogram named " << itHHH->first << " that I can't recognize in this loop!" << endl;
	      assert(0);

	    }
	  }
	}
      }
    }
  }

  for(HistosContainer::iterator itH=histosMap_.begin(); itH!=histosMap_.end(); itH++){
    for(map<string, map<string,MonitorElement*> >::iterator itHH=itH->second.begin(); itHH!=itH->second.end(); itHH++){
      double min = +1000000.;
      double max = -1000000.;
      double minDelta = +1000000.;
      double maxDelta = -1000000.;
      if(itHH->first != "run"){
      	for(map<string,MonitorElement*>::iterator itHHH=itHH->second.begin(); itHHH!=itHH->second.end(); itHHH++){
	  if(itHHH->second != 0){
	    for(int bin=1; bin<=itHHH->second->getTH1()->GetNbinsX(); bin++){
	      if(itHHH->second->getTH1()->GetBinError(bin) != 0){
	    	if(itHHH->first == "Lumibased BeamSpotFit" 
	    	|| itHHH->first == "Lumibased PrimaryVertex" 
	    	|| itHHH->first == "Lumibased DataBase" 
	    	|| itHHH->first == "Lumibased Scalers"){
		  if(min >= itHHH->second->getTH1()->GetBinContent(bin)){					           
	    	    min = itHHH->second->getTH1()->GetBinContent(bin);
	    	  }												           
	    	  if(max < itHHH->second->getTH1()->GetBinContent(bin)){					           
	    	    max = itHHH->second->getTH1()->GetBinContent(bin);
	    	  }												           
	    	}
	    	else if(itHHH->first == "Lumibased PrimaryVertex-DataBase fit" 
	    	|| itHHH->first == "Lumibased PrimaryVertex-Scalers fit"
	    	|| itHHH->first == "Lumibased Scalers-DataBase fit"){
	    	  if(minDelta > itHHH->second->getTH1()->GetBinContent(bin)){
	    	    minDelta = itHHH->second->getTH1()->GetBinContent(bin);
	    	  }
	    	  if(maxDelta < itHHH->second->getTH1()->GetBinContent(bin)){
	    	    maxDelta = itHHH->second->getTH1()->GetBinContent(bin);
	    	  }
	    	}
	    	else if(itHHH->first == "Lumibased PrimaryVertex-DataBase" 
	    	|| itHHH->first == "Lumibased PrimaryVertex-Scalers"
	    	|| itHHH->first == "Lumibased Scalers-DataBase"){
	    	  if(minDelta > itHHH->second->getTProfile()->GetBinContent(bin)){
	    	    minDelta = itHHH->second->getTProfile()->GetBinContent(bin);
	    	  }
	    	  if(maxDelta < itHHH->second->getTProfile()->GetBinContent(bin)){
	    	    maxDelta = itHHH->second->getTProfile()->GetBinContent(bin);
	    	  }
	    	}
      	    	else{
	    	  LogInfo("AlcaBeamMonitor")
		    << "The histosMap_ have a histogram named " << itHHH->first << " that I can't recognize in this loop!" << endl;
	    	  assert(0);

	    	}
	      }
	    }
	  }
      	}
      	for(map<string,MonitorElement*>::iterator itHHH=itHH->second.begin(); itHHH!=itHH->second.end(); itHHH++){
	  if(itHHH->second != 0){
	    if(itHHH->first == "Lumibased BeamSpotFit" 
	    || itHHH->first == "Lumibased PrimaryVertex" 
	    || itHHH->first == "Lumibased DataBase" 
	    || itHHH->first == "Lumibased Scalers"){
	      itHHH->second->getTH1()->SetMinimum(min-0.1*(max-min));
	      itHHH->second->getTH1()->SetMaximum(max+0.1*(max-min));
	    }
	    else if(itHHH->first == "Lumibased PrimaryVertex-DataBase fit" 
	    || itHHH->first == "Lumibased PrimaryVertex-Scalers fit"
	    || itHHH->first == "Lumibased Scalers-DataBase fit"){
	      itHHH->second->getTH1()->SetMinimum(minDelta-0.1*(maxDelta-minDelta));
	      itHHH->second->getTH1()->SetMaximum(maxDelta+0.1*(maxDelta-minDelta));
	    }
	    else if(itHHH->first == "Lumibased PrimaryVertex-DataBase" 
	    || itHHH->first == "Lumibased PrimaryVertex-Scalers"
	    || itHHH->first == "Lumibased Scalers-DataBase"){
	      itHHH->second->getTProfile()->SetMinimum(minDelta-5*(maxDelta-minDelta));
	      itHHH->second->getTProfile()->SetMaximum(maxDelta+5*(maxDelta-minDelta));
	    }
      	    else{
	      LogInfo("AlcaBeamMonitor")
		<< "The histosMap_ have a histogram named " << itHHH->first << " that I can't recognize in this loop!" << endl;
	      assert(0);

	    }
	  }
      	}
      }
    }
  }


/*
  double BFVal,PVVal,DBVal;
  double BFValError,PVValError,DBValError;
  typedef struct {
   int type;
   double val;
   double valError;    
  } ValStruct;
  map<string,ValStruct> tmpValues;
  for(map<string,ValueHistos>::iterator itM=hValMap_.begin(); itM!=hValMap_.end(); itM++){
    for(map<string,MonitorElement**>::iterator itMM=itM->second.histoMapLumi.begin(); itMM!=itM->second.histoMapLumi.end(); itMM++){
      tmpValues[itMM->first] = ValStruct();
    }
    int    bin = 0;
    bool   first = true;
    double min = 0, max = 0;
    bool   firstDelta = true;
    double minDelta = 0,maxDelta = 0;
    for(bsMap_iterator it=beamSpotMap_.begin(); it!=beamSpotMap_.end();it++){
      tmpGroup = &(it->second);
      if(itM->first == "x"){
        BFVal = tmpGroup->BFBS.x0();
	PVVal = tmpGroup->PVBS.x0();
	DBVal = tmpGroup->DBBS.x0();
        BFValError = tmpGroup->BFBS.x0Error();
	PVValError = tmpGroup->PVBS.x0Error();
	DBValError = tmpGroup->DBBS.x0Error();
      }
      else if(itM->first == "y"){
        BFVal = tmpGroup->BFBS.y0();
	PVVal = tmpGroup->PVBS.y0();
	DBVal = tmpGroup->DBBS.y0();
        BFValError = tmpGroup->BFBS.y0Error();
	PVValError = tmpGroup->PVBS.y0Error();
	DBValError = tmpGroup->DBBS.y0Error();
      }
      else if(itM->first == "z"){
        BFVal = tmpGroup->BFBS.z0();
	PVVal = tmpGroup->PVBS.z0();
	DBVal = tmpGroup->DBBS.z0();
        BFValError = tmpGroup->BFBS.z0Error();
	PVValError = tmpGroup->PVBS.z0Error();
	DBValError = tmpGroup->DBBS.z0Error();
      }
      else if(itM->first == "sigmaX"){
        BFVal = tmpGroup->BFBS.BeamWidthX();
	PVVal = tmpGroup->PVBS.BeamWidthX();
	DBVal = tmpGroup->DBBS.BeamWidthX();
        BFValError = tmpGroup->BFBS.BeamWidthXError();
	PVValError = tmpGroup->PVBS.BeamWidthXError();
	DBValError = tmpGroup->DBBS.BeamWidthXError();
      }
      else if(itM->first == "sigmaY"){
        BFVal = tmpGroup->BFBS.BeamWidthY();
	PVVal = tmpGroup->PVBS.BeamWidthY();
	DBVal = tmpGroup->DBBS.BeamWidthY();
        BFValError = tmpGroup->BFBS.BeamWidthYError();
	PVValError = tmpGroup->PVBS.BeamWidthYError();
	DBValError = tmpGroup->DBBS.BeamWidthYError();
      }
      else if(itM->first == "sigmaZ"){
        BFVal = tmpGroup->BFBS.sigmaZ();
	PVVal = tmpGroup->PVBS.sigmaZ();
	DBVal = tmpGroup->DBBS.sigmaZ();
        BFValError = tmpGroup->BFBS.sigmaZ0Error();
	PVValError = tmpGroup->PVBS.sigmaZ0Error();
	DBValError = tmpGroup->DBBS.sigmaZ0Error();
      }
      for(map<string,MonitorElement**>::iterator itMM=itM->second.histoMapLumi.begin(); itMM!=itM->second.histoMapLumi.end(); itMM++){
      	if(itMM->first == "Lumibased BeamSpot"){
	  tmpValues[itMM->first].type     = tmpGroup->BFBS.type();
	  tmpValues[itMM->first].val      = BFVal;
	  tmpValues[itMM->first].valError = BFValError;
	}
	else if(itMM->first == "Lumibased PrimaryVertex"){
	  tmpValues[itMM->first].type     = tmpGroup->PVBS.type();
	  tmpValues[itMM->first].val      = PVVal;
	  tmpValues[itMM->first].valError = PVValError;
	}
	else if(itMM->first == "Lumibased Payload"){
	  tmpValues[itMM->first].type     = tmpGroup->DBBS.type();
	  tmpValues[itMM->first].val      = DBVal;
	  tmpValues[itMM->first].valError = DBValError;
	}
	else if(itMM->first == "Lumibased Payload-PrimaryVertex"){
	  if(tmpGroup->DBBS.type() == BeamSpot::Tracker || tmpGroup->PVBS.type() == BeamSpot::Tracker){
	    tmpValues[itMM->first].type = BeamSpot::Tracker;
	  }
	  else{
	    tmpValues[itMM->first].type = BeamSpot::Unknown;
	  }
	  tmpValues[itMM->first].val      = DBVal-PVVal;
	  tmpValues[itMM->first].valError = sqrt(DBValError*DBValError+PVValError*PVValError)/sqrt(2);
	}
      }				    
      for(map<string,MonitorElement**>::iterator itMM=itM->second.histoMapLumi.begin(); itMM!=itM->second.histoMapLumi.end(); itMM++){
      	if(itMM->first == "Lumibased BeamSpot" || itMM->first == "Lumibased PrimaryVertex" || itMM->first == "Lumibased Payload"){  				    
      	  if(tmpValues[itMM->first].type == BeamSpot::Tracker){
	    if(first){
      	      min = tmpValues[itMM->first].val;
      	      max = tmpValues[itMM->first].val;
      	      first = false;
      	    }
      	    else{
      	      if(tmpValues[itMM->first].val < min){
      	  	min = tmpValues[itMM->first].val;
      	      }
      	      if(tmpValues[itMM->first].val > max){
      	  	max = tmpValues[itMM->first].val;
      	      }
      	    }
	  }
        }
        else{
      	  if(tmpValues[itMM->first].type == BeamSpot::Tracker){
	    if(firstDelta){
      	      minDelta = tmpValues[itMM->first].val;
      	      maxDelta = tmpValues[itMM->first].val;
      	      firstDelta = false;
      	    }
      	    else{
      	      if(tmpValues[itMM->first].val < min){
      	  	minDelta = tmpValues[itMM->first].val;
      	      }
      	      if(tmpValues[itMM->first].val > max){
      	  	maxDelta = tmpValues[itMM->first].val;
      	      }
      	    }
	  }
        }
      	bin = (*(itMM->second))->getTH1()->FindBin(it->first);
      	(*(itMM->second))->setBinContent(bin,tmpValues[itMM->first].val);
      	(*(itMM->second))->setBinError(bin,tmpValues[itMM->first].valError);
      	/////////////////////////////////////////////////////
	//h_x0->Fill(tmpGroup->BFBS.x0());
	////////////////////////////////////////////////
      }
    }
  }

*/

/*  
  for(map<string,ValueHistos>::iterator itM=hValMap_.begin(); itM!=hValMap_.end(); itM++){
    for(vector<MonitorElement*>::iterator itV=itM->second.histoVLumi.begin(); itV!=itM->second.histoVLumi.end(); itV++){
      if( (**itV) != 0){
        dbe_->removeElement(monitorName_+"Validation",(*itV)->getName());
	(**itV) = 0;
      }
    }
  }

  h_x0_BF_lumi=dbe_->book1D("h_x0_BF_lumi","X0 vs lumi from BF file",lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5);
  h_x0_BF_lumi->setAxisTitle("Lumisection",1);
  h_x0_BF_lumi->setAxisTitle("x_{0} (cm)",2);

  h_x0_PV_lumi=dbe_->book1D("h_x0_PV_lumi","X0 vs lumi from PV file",lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5);
  h_x0_PV_lumi->setAxisTitle("Lumisection",1);
  h_x0_PV_lumi->setAxisTitle("x_{0} (cm)",2);

  h_x0_DB_lumi=dbe_->book1D("h_x0_DB_lumi","X0 vs lumi from database",lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5);
  h_x0_DB_lumi->setAxisTitle("Lumisection",1);
  h_x0_DB_lumi->setAxisTitle("x_{0} (cm)",2);

  h_x0_delta_DB_PV_lumi=dbe_->book1D("h_x0_delta_DB_PV_lumi","DB-PV Beamspot X0 vs lumi",lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5);
  h_x0_delta_DB_PV_lumi->setAxisTitle("Lumisection",1);
  h_x0_delta_DB_PV_lumi->setAxisTitle("Database-PrimaryVertex x_{0} (cm)",2);

  
  int bin = 0;
  bool first = true;
  double min = 0,max = 0;
  bool first_x0_delta_DB_PV_lumi = true;
  double min_x0_delta_DB_PV_lumi = 0,max_x0_delta_DB_PV_lumi = 0;
  GroupOfBeamSpots *tmpGroup = 0;
  for(bsMap_iterator it=beamSpotMap_.begin(); it!=beamSpotMap_.end();it++){
    tmpGroup = &(it->second);
    if(tmpGroup->BFBS.type() == BeamSpot::Tracker){
      if(first){
        min = tmpGroup->BFBS.x0();
        max = tmpGroup->BFBS.x0();
        first = false;
      }
      else{
        if(tmpGroup->BFBS.x0() < min){
	  min = tmpGroup->BFBS.x0();
	}
        if(tmpGroup->BFBS.x0() > max){
	  max = tmpGroup->BFBS.x0();
	}
      }
      bin = h_x0_BF_lumi->getTH1()->FindBin(it->first);
      h_x0_BF_lumi->setBinContent(bin,tmpGroup->BFBS.x0());
      h_x0_BF_lumi->setBinError(bin,tmpGroup->BFBS.x0Error());
      h_x0->Fill(tmpGroup->BFBS.x0());
    }
    if(tmpGroup->PVBS.type() == BeamSpot::Tracker){
      if(first){
        min = tmpGroup->PVBS.x0();
        max = tmpGroup->PVBS.x0();
        first = false;
      }
      else{
        if(tmpGroup->PVBS.x0() < min){
	  min = tmpGroup->PVBS.x0();
	}
        if(tmpGroup->PVBS.x0() > max){
	  max = tmpGroup->PVBS.x0();
	}
      }
      bin = h_x0_PV_lumi->getTH1()->FindBin(it->first);
      h_x0_PV_lumi->setBinContent(bin,tmpGroup->PVBS.x0());
      h_x0_PV_lumi->setBinError(bin,tmpGroup->PVBS.x0Error());
      h_x0->Fill(tmpGroup->PVBS.x0());
    }
    if(tmpGroup->DBBS.type() == BeamSpot::Tracker){
      if(first){
        min = tmpGroup->DBBS.x0();
        max = tmpGroup->DBBS.x0();
        first = false;
      }
      else{
        if(tmpGroup->DBBS.x0() < min){
	  min = tmpGroup->DBBS.x0();
	}
        if(tmpGroup->DBBS.x0() > max){
	  max = tmpGroup->DBBS.x0();
	}
      }
      bin = h_x0_DB_lumi->getTH1()->FindBin(it->first);
      h_x0_DB_lumi->setBinContent(bin,tmpGroup->DBBS.x0());
      h_x0_DB_lumi->setBinError(bin,tmpGroup->DBBS.x0Error());
      h_x0->Fill(tmpGroup->DBBS.x0());
    }
    if(tmpGroup->DBBS.type() == BeamSpot::Tracker && tmpGroup->PVBS.type() == BeamSpot::Tracker){
      if(first_x0_delta_DB_PV_lumi){
        min_x0_delta_DB_PV_lumi = tmpGroup->DBBS.x0()-tmpGroup->PVBS.x0();
        max_x0_delta_DB_PV_lumi = tmpGroup->DBBS.x0()-tmpGroup->PVBS.x0();
        first_x0_delta_DB_PV_lumi = false;
      }
      else{
        if(tmpGroup->DBBS.x0()-tmpGroup->PVBS.x0() < min_x0_delta_DB_PV_lumi){
	  min_x0_delta_DB_PV_lumi = tmpGroup->DBBS.x0()-tmpGroup->PVBS.x0();
	}
        if(tmpGroup->DBBS.x0()-tmpGroup->PVBS.x0() > max_x0_delta_DB_PV_lumi){
	  max_x0_delta_DB_PV_lumi = tmpGroup->DBBS.x0()-tmpGroup->PVBS.x0();
	}
      }
      bin = h_x0_delta_DB_PV_lumi->getTH1()->FindBin(it->first);
      h_x0_delta_DB_PV_lumi->setBinContent(bin,tmpGroup->DBBS.x0()-tmpGroup->PVBS.x0());
      h_x0_delta_DB_PV_lumi->setBinError(bin,sqrt(tmpGroup->DBBS.x0Error()*tmpGroup->DBBS.x0Error()+tmpGroup->PVBS.x0Error()*tmpGroup->PVBS.x0Error())/sqrt(2));
      h_x0_delta_DB_PV->Fill(tmpGroup->DBBS.x0()-tmpGroup->PVBS.x0());
    }
  }
  
  h_x0_BF_lumi         ->getTH1()->SetMinimum(min-0.002);
  h_x0_BF_lumi         ->getTH1()->SetMaximum(max+0.002);
  h_x0_PV_lumi         ->getTH1()->SetMinimum(min-0.002);
  h_x0_PV_lumi         ->getTH1()->SetMaximum(max+0.002);
  h_x0_DB_lumi	       ->getTH1()->SetMinimum(min-0.002);
  h_x0_DB_lumi	       ->getTH1()->SetMaximum(max+0.002);
  h_x0_delta_DB_PV_lumi->getTH1()->SetMinimum(min_x0_delta_DB_PV_lumi-0.002);
  h_x0_delta_DB_PV_lumi->getTH1()->SetMaximum(max_x0_delta_DB_PV_lumi+0.002);
*/
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitor::endJob(const LuminosityBlock& iLumi, const EventSetup& iSetup){
}


DEFINE_FWK_MODULE(AlcaBeamMonitor);
