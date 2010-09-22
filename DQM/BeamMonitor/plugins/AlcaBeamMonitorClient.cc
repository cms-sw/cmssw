/*
 * \file AlcaBeamMonitorClient.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 * $Date: 2010/09/01 19:29:58 $
 * $Revision: 1.5 $
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
#include "DQM/BeamMonitor/plugins/AlcaBeamMonitorClient.h"
#include "DQMServices/Core/interface/QReport.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <numeric>
//#include <iostream>

using namespace std;
using namespace edm;
using namespace reco;

//----------------------------------------------------------------------------------------------------------------------
AlcaBeamMonitorClient::AlcaBeamMonitorClient( const ParameterSet& ps ){

  parameters_         = ps;
  monitorName_        = parameters_.getUntrackedParameter<string>("MonitorName","YourSubsystemName");

  dbe_            = Service<DQMStore>().operator->();
  
//  LogInfo("AlcaBeamMonitorClient") 
//    << "Monitor name: " << monitorName_;

  
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  

  varNamesV_.push_back("x");
  varNamesV_.push_back("y");
  varNamesV_.push_back("z");
  varNamesV_.push_back("sigmaX");
  varNamesV_.push_back("sigmaY");
  varNamesV_.push_back("sigmaZ");

/*
  histoByCategoryNames_.insert( pair<string,string>("run",        "Coordinate"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-BeamFit"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex fit-Scalers"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-DataBase"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-BeamFit"));
  histoByCategoryNames_.insert( pair<string,string>("run",        "PrimaryVertex-Scalers"));
*/
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
      histosMap_[*itV][itM->first][itM->second] = 0;
      positionsMap_[*itV][itM->first][itM->second] = 3*numberOfValuesToSave_;//value, error, ok 
      ++numberOfValuesToSave_;
    }
  }
}


AlcaBeamMonitorClient::~AlcaBeamMonitorClient() {
}


//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::beginJob() {
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::beginRun(const edm::Run& r, const EventSetup& context) {
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::beginLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::analyze(const Event& iEvent, const EventSetup& iSetup ){
}


//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::endLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  ++numberOfLumis_;
  MonitorElement * tmp_ = 0;	
//  tmp_ = dbe_->get(monitorName_+"Service/hHistoLumiValues");
  tmp_ = dbe_->get("LumiFlag/AnotherFolder/HistoLumiFlag");
  if(!tmp_){
    return;
  }
  valuesMap_[iLumi.id().luminosityBlock()] = vector<double>();
  for(int i=0; i<3*numberOfValuesToSave_; i++){
//    cout << tmp_->getTProfile()->GetBinContent(i+1) << ":";
    valuesMap_[iLumi.id().luminosityBlock()].push_back(tmp_->getTProfile()->GetBinContent(i+1));	
  }
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::endRun(const Run& iRun, const EventSetup& context){

  // use this in case any LS is missing.
  if(valuesMap_.size() == 0){
    LogInfo("AlcaBeamMonitorClient") 
      << "The histogram " << monitorName_+"Service/hHistoLumiValues which contains all the values has not been found in any lumi!";
    return;
  }
  lastLumi_ = (--valuesMap_.end())->first;
  firstLumi_ = valuesMap_.begin()->first;

  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Validation");

  LogInfo("AlcaBeamMonitorClient") 
    << "End of run " << iRun.id().run() << "(" << firstLumi_ << "-" << lastLumi_ << ")";

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
	    itMMM->second = dbe_->book1D(name,title,lastLumi_-firstLumi_+1,firstLumi_-2,lastLumi_+2);
      	  }
          else if(itMM->first == "validation" && itMMM->first == "Lumibased Scalers-DataBase fit"){
            dbe_->setCurrentFolder(monitorName_+"Validation");
	    itMMM->second = dbe_->book1D(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5);
          }
	  else if(itMM->first == "validation" && itMMM->first != "Lumibased Scalers-DataBase fit" && (itM->first == "x" || itM->first == "y") ){
	    dbe_->setCurrentFolder(monitorName_+"Validation");
	    itMMM->second = dbe_->book1D(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5);
//LORE	    itMMM->second = dbe_->bookProfile(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5,-0.1,0.1,"");
	  }
	  else if(itMM->first == "validation" && itMMM->first != "Lumibased Scalers-DataBase fit" && itM->first == "z" ){
	    dbe_->setCurrentFolder(monitorName_+"Validation");
	    itMMM->second = dbe_->book1D(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5);
//LORE	    itMMM->second = dbe_->bookProfile(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5,-1,1,"");
	  }
//	  else if(itMM->first == "validation" && itMMM->first == "Lumibased Scalers-DataBase" && (itM->first == "sigmaX" || itM->first == "sigmaY") ){
//	    dbe_->setCurrentFolder(monitorName_+"Validation");
//	    itMMM->second = dbe_->bookProfile(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5,-0.1,0.1,"");
//	  }
//	  else if(itMM->first == "validation" && itMMM->first == "Lumibased Scalers-DataBase" && (itM->first == "sigmaZ") ){
//	    dbe_->setCurrentFolder(monitorName_+"Validation");
//	    itMMM->second = dbe_->bookProfile(name,title,lastLumi_-firstLumi_+1,firstLumi_-0.5,lastLumi_+0.5,-10,10,"");
//	  }
//	  else if(itMM->first == "validation" && itMMM->first != "Lumibased Scalers-DataBase" && (itM->first == "sigmaX" || itM->first == "sigmaY" || itM->first == "sigmaZ") ){
	  else if(itMM->first == "validation" && (itM->first == "sigmaX" || itM->first == "sigmaY" || itM->first == "sigmaZ") ){
	    itMMM->second = 0;
	  }
	  else{
 	    LogInfo("AlcaBeamMonitorClient") 
 	      << "Unrecognized category " << itMM->first;
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

/*  
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
            LogInfo("AlcaBeamMonitorClient")
	      << "The histosMap_ has been built with the name " << itH->first << " that I can't recognize!";
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
      	      LogInfo("AlcaBeamMonitorClient")										
		<< "The histosMap_ has been built with the name " << itH->first << " that I can't recognize!";
	      assert(0);												
	    }														
	  }
      	}
      }
    }
  }  

*/
  unsigned int bin=0;
  for(HistosContainer::iterator itH=histosMap_.begin(); itH!=histosMap_.end(); itH++){
    for(map<string, map<string,MonitorElement*> >::iterator itHH=itH->second.begin(); itHH!=itH->second.end(); itHH++){
      for(map<string,MonitorElement*>::iterator itHHH=itHH->second.begin(); itHHH!=itHH->second.end(); itHHH++){
    	for(map<LuminosityBlockNumber_t,vector<double> >::iterator itVal = valuesMap_.begin(); itVal != valuesMap_.end(); itVal++){
	  if(itHHH->second != 0){
//	    cout << positionsMap_[itH->first][itHH->first][itHHH->first] << endl;
	    if(itVal->second[positionsMap_[itH->first][itHH->first][itHHH->first]+2] == 1){
	      bin = itHHH->second->getTH1()->FindBin(itVal->first);
	      itHHH->second->setBinContent(bin,itVal->second[positionsMap_[itH->first][itHH->first][itHHH->first]]);
	      itHHH->second->setBinError(bin,itVal->second[positionsMap_[itH->first][itHH->first][itHHH->first]+1]);
	    }
	  }
	}
      }
    }
  }

/*  
  const double bigNumber = 1000000.;
  for(HistosContainer::iterator itH=histosMap_.begin(); itH!=histosMap_.end(); itH++){
    for(map<string, map<string,MonitorElement*> >::iterator itHH=itH->second.begin(); itHH!=itH->second.end(); itHH++){
      double min = bigNumber;
      double max = -bigNumber;
      double minDelta = bigNumber;
      double maxDelta = -bigNumber;
//      double minDeltaProf = bigNumber;
//      double maxDeltaProf = -bigNumber;
      if(itHH->first != "run"){
      	for(map<string,MonitorElement*>::iterator itHHH=itHH->second.begin(); itHHH!=itHH->second.end(); itHHH++){
	  if(itHHH->second != 0){
	    for(int bin=1; bin<=itHHH->second->getTH1()->GetNbinsX(); bin++){
	      if(itHHH->second->getTH1()->GetBinError(bin) != 0 || itHHH->second->getTH1()->GetBinContent(bin) != 0){
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
//	    	|| itHHH->first == "Lumibased Scalers-DataBase"
		){
	    	  if(minDelta > itHHH->second->getTProfile()->GetBinContent(bin)){
	    	    minDelta = itHHH->second->getTProfile()->GetBinContent(bin);
	    	  }
	    	  if(maxDelta < itHHH->second->getTProfile()->GetBinContent(bin)){
	    	    maxDelta = itHHH->second->getTProfile()->GetBinContent(bin);
	    	  }
	    	}
      	    	else{
	    	  LogInfo("AlcaBeamMonitorClient")
		    << "The histosMap_ have a histogram named " << itHHH->first << " that I can't recognize in this loop!";
	    	  assert(0);

	    	}
	      }
	    }
	  }
      	}
      	for(map<string,MonitorElement*>::iterator itHHH=itHH->second.begin(); itHHH!=itHH->second.end(); itHHH++){
//	  LogInfo("AlcaBeamMonitorClient")
//	    << itH->first << itHHH->first << " max-min=" << max-min << " delta=" << maxDelta-minDelta;
	  if(itHHH->second != 0){
	    if(itHHH->first == "Lumibased BeamSpotFit" 
	    || itHHH->first == "Lumibased PrimaryVertex" 
	    || itHHH->first == "Lumibased DataBase" 
	    || itHHH->first == "Lumibased Scalers"){
	      if((max == -bigNumber && min == bigNumber) || max-min == 0){
	        itHHH->second->getTH1()->SetMinimum(itHHH->second->getTH1()->GetMinimum()-0.01);
	        itHHH->second->getTH1()->SetMaximum(itHHH->second->getTH1()->GetMaximum()+0.01);
	      }
	      else{
	        itHHH->second->getTH1()->SetMinimum(min-0.1*(max-min));
	        itHHH->second->getTH1()->SetMaximum(max+0.1*(max-min));
	      }
	    }
	    else if(itHHH->first == "Lumibased PrimaryVertex-DataBase fit" 
	    || itHHH->first == "Lumibased PrimaryVertex-Scalers fit"
	    || itHHH->first == "Lumibased Scalers-DataBase fit"){
	      if((maxDelta == -bigNumber && minDelta == bigNumber) || maxDelta-minDelta == 0){
	        itHHH->second->getTH1()->SetMinimum(itHHH->second->getTH1()->GetMinimum()-0.01);
	        itHHH->second->getTH1()->SetMaximum(itHHH->second->getTH1()->GetMaximum()+0.01);
	      }
	      else{
	        itHHH->second->getTH1()->SetMinimum(minDelta-5*(maxDelta-minDelta));
	        itHHH->second->getTH1()->SetMaximum(maxDelta+5*(maxDelta-minDelta));
	      }
	    }
	    else if(itHHH->first == "Lumibased PrimaryVertex-DataBase" 
	    || itHHH->first == "Lumibased PrimaryVertex-Scalers"
//	    || itHHH->first == "Lumibased Scalers-DataBase"
	    ){
	      if((maxDelta == -bigNumber && minDelta == bigNumber) || maxDelta-minDelta == 0){
	        itHHH->second->getTProfile()->SetMinimum(itHHH->second->getTProfile()->GetMinimum()-0.01);
	        itHHH->second->getTProfile()->SetMaximum(itHHH->second->getTProfile()->GetMaximum()+0.01);
	      }
	      else{
	        itHHH->second->getTProfile()->SetMinimum(minDelta-5*(maxDelta-minDelta));
	        itHHH->second->getTProfile()->SetMaximum(maxDelta+5*(maxDelta-minDelta));
              }
	    }
      	    else{
	      LogInfo("AlcaBeamMonitorClient")
		<< "The histosMap_ have a histogram named " << itHHH->first << " that I can't recognize in this loop!";
	      assert(0);

	    }
	  }
      	}
      }
    }
  }
*/

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
void AlcaBeamMonitorClient::endJob(const LuminosityBlock& iLumi, const EventSetup& iSetup){
}


DEFINE_FWK_MODULE(AlcaBeamMonitorClient);
