/*
 * \file AlcaBeamMonitorClient.cc
 * \author Lorenzo Uplegger/FNAL
 * $Date: 2011/01/06 17:35:56 $
 * $Revision: 1.5 $
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQM/BeamMonitor/plugins/AlcaBeamMonitorClient.h"
#include "DQMServices/Core/interface/QReport.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <numeric>
//#include <iostream>

using namespace std;
using namespace edm;
//using namespace reco;

//----------------------------------------------------------------------------------------------------------------------
AlcaBeamMonitorClient::AlcaBeamMonitorClient( const ParameterSet& ps ) :
  parameters_         (ps),
  monitorName_        (parameters_.getUntrackedParameter<string>("MonitorName","YourSubsystemName")),
  numberOfValuesToSave_ (0)
{
  dbe_            = Service<DQMStore>().operator->();
  
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
  for(HistosContainer::iterator itM=histosMap_.begin(); itM!=histosMap_.end(); itM++){
    for(map<string,map<string,MonitorElement*> >::iterator itMM=itM->second.begin(); itMM!=itM->second.end(); itMM++){
      if(itMM->first != "run"){
      	for(map<string,MonitorElement*>::iterator itMMM=itMM->second.begin(); itMMM!=itMM->second.end(); itMMM++){
      	  if( itMMM->second != 0){
	    if(itMM->first == "lumi"){
      	      dbe_->removeElement(monitorName_+"Debug",itMMM->second->getName());
	    }
	    else if(itMM->first == "validation"){
      	      dbe_->removeElement(monitorName_+"Validation",itMMM->second->getName());
	    }
	    else{
 	      LogInfo("AlcaBeamMonitorClient") 
 	        << "Unrecognized category " << itMM->first;
	        //assert(0);	    
	    }
      	    itMMM->second = 0;
      	  }
      	}
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::beginLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::analyze(const Event& iEvent, const EventSetup& iSetup ){
}


//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::endLuminosityBlock(const LuminosityBlock& iLumi, const EventSetup& iSetup) {
  MonitorElement * tmp = 0;	
  tmp = dbe_->get(monitorName_+"Service/hHistoLumiValues");
  if(!tmp){
    return;
  }
  valuesMap_[iLumi.id().luminosityBlock()] = vector<double>();
  for(int i=0; i<3*numberOfValuesToSave_; i++){
    valuesMap_[iLumi.id().luminosityBlock()].push_back(tmp->getTProfile()->GetBinContent(i+1));	
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
  int lastLumi = (--valuesMap_.end())->first;
  int firstLumi = valuesMap_.begin()->first;

  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Validation");

  LogInfo("AlcaBeamMonitorClient") 
    << "End of run " << iRun.id().run() << "(" << firstLumi << "-" << lastLumi << ")";

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
	    itMMM->second = dbe_->book1D(name,title,lastLumi-firstLumi+1,firstLumi-0.5,lastLumi+0.5);
      	  }
          else if(itMM->first == "validation" && itMMM->first == "Lumibased Scalers-DataBase fit"){
            dbe_->setCurrentFolder(monitorName_+"Validation");
	    itMMM->second = dbe_->book1D(name,title,lastLumi-firstLumi+1,firstLumi-0.5,lastLumi+0.5);
          }
	  else if(itMM->first == "validation" && itMMM->first != "Lumibased Scalers-DataBase fit" && (itM->first == "x" || itM->first == "y" || itM->first == "z")){
//LORE	  else if(itMM->first == "validation" && itMMM->first != "Lumibased Scalers-DataBase fit" && (itM->first == "x" || itM->first == "y") ){
	    dbe_->setCurrentFolder(monitorName_+"Validation");
	    itMMM->second = dbe_->book1D(name,title,lastLumi-firstLumi+1,firstLumi-0.5,lastLumi+0.5);
//LORE	    itMMM->second = dbe_->bookProfile(name,title,lastLumi-firstLumi+1,firstLumi-0.5,lastLumi+0.5,-0.1,0.1,"");
	  }
//LORE	  else if(itMM->first == "validation" && itMMM->first != "Lumibased Scalers-DataBase fit" && itM->first == "z" ){
//LORE	    dbe_->setCurrentFolder(monitorName_+"Validation");
//LORE	    itMMM->second = dbe_->book1D(name,title,lastLumi-firstLumi+1,firstLumi-0.5,lastLumi+0.5);
//LORE	    itMMM->second = dbe_->bookProfile(name,title,lastLumi-firstLumi+1,firstLumi-0.5,lastLumi+0.5,-1,1,"");
//	  }
//	  else if(itMM->first == "validation" && itMMM->first == "Lumibased Scalers-DataBase" && (itM->first == "sigmaX" || itM->first == "sigmaY") ){
//	    dbe_->setCurrentFolder(monitorName_+"Validation");
//	    itMMM->second = dbe_->bookProfile(name,title,lastLumi-firstLumi+1,firstLumi-0.5,lastLumi+0.5,-0.1,0.1,"");
//	  }
//	  else if(itMM->first == "validation" && itMMM->first == "Lumibased Scalers-DataBase" && (itM->first == "sigmaZ") ){
//	    dbe_->setCurrentFolder(monitorName_+"Validation");
//	    itMMM->second = dbe_->bookProfile(name,title,lastLumi-firstLumi+1,firstLumi-0.5,lastLumi+0.5,-10,10,"");
//	  }
//	  else if(itMM->first == "validation" && itMMM->first != "Lumibased Scalers-DataBase" && (itM->first == "sigmaX" || itM->first == "sigmaY" || itM->first == "sigmaZ") ){
	  else if(itMM->first == "validation" && (itM->first == "sigmaX" || itM->first == "sigmaY" || itM->first == "sigmaZ") ){
            dbe_->setCurrentFolder(monitorName_+"Validation");
	    itMMM->second = 0;
	  }
	  else{
 	    LogInfo("AlcaBeamMonitorClient") 
 	      << "Unrecognized category " << itMM->first;
	      //assert(0);	    
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

/**/
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
		  if(min > itHHH->second->getTH1()->GetBinContent(bin)){					           
	    	    min = itHHH->second->getTH1()->GetBinContent(bin);
	    	  }												           
	    	  if(max < itHHH->second->getTH1()->GetBinContent(bin)){					           
	    	    max = itHHH->second->getTH1()->GetBinContent(bin);
	    	  }												           
	    	}
	    	else if(itHHH->first == "Lumibased PrimaryVertex-DataBase fit" 
	    	|| itHHH->first == "Lumibased PrimaryVertex-Scalers fit"
	    	|| itHHH->first == "Lumibased Scalers-DataBase fit"
		|| itHHH->first == "Lumibased PrimaryVertex-DataBase" 
		|| itHHH->first == "Lumibased PrimaryVertex-Scalers"
		){
	    	  if(minDelta > itHHH->second->getTH1()->GetBinContent(bin)){
	    	    minDelta = itHHH->second->getTH1()->GetBinContent(bin);
	    	  }
	    	  if(maxDelta < itHHH->second->getTH1()->GetBinContent(bin)){
	    	    maxDelta = itHHH->second->getTH1()->GetBinContent(bin);
	    	  }
	    	}
//	    	else if(itHHH->first == "Lumibased PrimaryVertex-DataBase" 
//	    	|| itHHH->first == "Lumibased PrimaryVertex-Scalers"
////	    	|| itHHH->first == "Lumibased Scalers-DataBase"
//		){
//	    	  if(minDelta > itHHH->second->getTProfile()->GetBinContent(bin)){
//	    	    minDelta = itHHH->second->getTProfile()->GetBinContent(bin);
//	    	  }
//	    	  if(maxDelta < itHHH->second->getTProfile()->GetBinContent(bin)){
//	    	    maxDelta = itHHH->second->getTProfile()->GetBinContent(bin);
//	    	  }
//	    	}
      	    	else{
	    	  LogInfo("AlcaBeamMonitorClient")
		    << "The histosMap_ have a histogram named " << itHHH->first << " that I can't recognize in this loop!";
	    	  //assert(0);

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
	    || itHHH->first == "Lumibased Scalers-DataBase fit"
	    || itHHH->first == "Lumibased PrimaryVertex-DataBase"
	    || itHHH->first == "Lumibased PrimaryVertex-Scalers"
	    ){
	      if((maxDelta == -bigNumber && minDelta == bigNumber) || maxDelta-minDelta == 0){
	        itHHH->second->getTH1()->SetMinimum(itHHH->second->getTH1()->GetMinimum()-0.01);
	        itHHH->second->getTH1()->SetMaximum(itHHH->second->getTH1()->GetMaximum()+0.01);
	      }
	      else{
	        itHHH->second->getTH1()->SetMinimum(minDelta-2*(maxDelta-minDelta));
	        itHHH->second->getTH1()->SetMaximum(maxDelta+2*(maxDelta-minDelta));
	      }
	    }
//	    else if(itHHH->first == "Lumibased PrimaryVertex-DataBase" 
//	    || itHHH->first == "Lumibased PrimaryVertex-Scalers"
////	    || itHHH->first == "Lumibased Scalers-DataBase"
//	    ){
//	      if((maxDelta == -bigNumber && minDelta == bigNumber) || maxDelta-minDelta == 0){
//	        itHHH->second->getTProfile()->SetMinimum(itHHH->second->getTProfile()->GetMinimum()-0.01);
//	        itHHH->second->getTProfile()->SetMaximum(itHHH->second->getTProfile()->GetMaximum()+0.01);
//	      }
//	      else{
//	        itHHH->second->getTProfile()->SetMinimum(minDelta-5*(maxDelta-minDelta));
//	        itHHH->second->getTProfile()->SetMaximum(maxDelta+5*(maxDelta-minDelta));
 //             }
//	    }
      	    else{
	      LogInfo("AlcaBeamMonitorClient")
		<< "The histosMap_ have a histogram named " << itHHH->first << " that I can't recognize in this loop!";
	      //assert(0);

	    }
	  }
      	}
      }
    }
  }
/**/

}

//----------------------------------------------------------------------------------------------------------------------
void AlcaBeamMonitorClient::endJob(const LuminosityBlock& iLumi, const EventSetup& iSetup){
}


DEFINE_FWK_MODULE(AlcaBeamMonitorClient);
