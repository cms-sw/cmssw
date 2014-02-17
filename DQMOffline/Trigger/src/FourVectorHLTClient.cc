/*

   FourVectorHLTClient.cc
   author:  Vladimir Rekovic, U Minn. 
   date of first version: Sept 2008

*/
//$Id: FourVectorHLTClient.cc,v 1.29 2011/09/27 16:29:40 bjk Exp $

#include "DQMOffline/Trigger/interface/FourVectorHLTClient.h"

#include "DQMServices/Core/interface/QTest.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TRandom.h"
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TGraph.h>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <memory>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include "TROOT.h"


using namespace edm;
using namespace std;

FourVectorHLTClient::FourVectorHLTClient(const edm::ParameterSet& ps)
{

  parameters_=ps;
  initialize();

}

FourVectorHLTClient::~FourVectorHLTClient(){

  LogDebug("FourVectorHLTClient")<< "FourVectorHLTClient: ending...." ;

}

//--------------------------------------------------------
void FourVectorHLTClient::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();

  processname_ = parameters_.getUntrackedParameter<std::string>("processname","HLT");
  

  // base folder for the contents of this job
  sourceDir_ = TString(parameters_.getUntrackedParameter<string>("hltSourceDir",""));



  // remove trainling "/" from dirname
  while(sourceDir_.Last('/') == sourceDir_.Length()-1) {
    sourceDir_.Remove(sourceDir_.Length()-1);
  }
  LogDebug("FourVectorHLTClient")<< "Source dir = " << sourceDir_ << endl;
    
  clientDir_ = TString(parameters_.getUntrackedParameter<string>("hltClientDir",""));

  // remove trainling "/" from dirname
  while(clientDir_.Last('/') == clientDir_.Length()-1) {
    clientDir_.Remove(clientDir_.Length()-1);
  }
  LogDebug("FourVectorHLTClient")<< "Client dir = " << clientDir_ << endl;
    
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  LogDebug("FourVectorHLTClient")<< "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  LogDebug("FourVectorHLTClient")<< "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
  
  customEffDir_ = TString(parameters_.getUntrackedParameter<string>("customEffDir","custom-efficiencies"));
  LogDebug("FourVectorHLTClient")<< "Custom Efficiencies dir = " << customEffDir_ << endl;
    
  std::vector<edm::ParameterSet> effpaths = parameters_.getParameter<std::vector<edm::ParameterSet> >("effpaths");

  std::pair<std::string, std::string> custompathnamepair;
  for(std::vector<edm::ParameterSet>::iterator pathconf = effpaths.begin() ; pathconf != effpaths.end(); 
      pathconf++) {
       custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
       custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
       custompathnamepairs_.push_back(custompathnamepair);
       //    customdenompathnames_.push_back(pathconf->getParameter<std::string>("denompathname"));  
       // custompathnames_.push_back(pathconf->getParameter<std::string>("pathname"));  
    }

  pathsSummaryFolder_ = parameters_.getUntrackedParameter ("pathsSummaryFolder",std::string("HLT/FourVector/PathsSummary/"));
  pathsSummaryHLTCorrelationsFolder_ = parameters_.getUntrackedParameter ("hltCorrelationsFolder",std::string("HLT/FourVector/PathsSummary/HLT Correlations/"));
  pathsSummaryFilterCountsFolder_ = parameters_.getUntrackedParameter ("filterCountsFolder",std::string("HLT/FourVector/PathsSummary/Filters Counts/"));
  pathsSummaryFilterEfficiencyFolder_ = parameters_.getUntrackedParameter ("filterEfficiencyFolder",std::string("HLT/FourVector/PathsSummary/Filters Efficiencies/"));

}

//--------------------------------------------------------
void FourVectorHLTClient::beginJob(){

  LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient]: beginJob" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

}

//--------------------------------------------------------
void FourVectorHLTClient::beginRun(const Run& r, const EventSetup& context) {

  LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient]: beginRun" << endl;
  // HLT config does not change within runs!
  bool changed=false;
 
  if (!hltConfig_.init(r, context, processname_, changed)) {

    processname_ = "FU";

    if (!hltConfig_.init(r, context, processname_, changed)){

      LogDebug("FourVectorHLTOffline") << "HLTConfigProvider failed to initialize.";

    }

    // check if trigger name in (new) config
    //  cout << "Available TriggerNames are: " << endl;
    //  hltConfig_.dump("Triggers");
  }

}

//--------------------------------------------------------
void FourVectorHLTClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void FourVectorHLTClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){   
} 



//--------------------------------------------------------
void FourVectorHLTClient::analyze(const Event& e, const EventSetup& context){
   
  counterEvt_++;
  if (prescaleEvt_<1) return;
  if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;
  
  LogDebug("FourVectorHLTClient")<<"analyze..." << endl;

}

//--------------------------------------------------------
void FourVectorHLTClient::endRun(const Run& r, const EventSetup& context){

  LogDebug("FourVectorHLTClient")<<"FourVectorHLTClient:: endLuminosityBlock "  << endl;


  // Work with PathsSummary ME's
  //////////////////////////////
  dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());
 
  std::vector<MonitorElement*> summaryME;
  summaryME = dbe_->getContents(pathsSummaryFilterCountsFolder_.c_str());

  vector<float> hltPassCount;

  for(unsigned int i=0; i< summaryME.size(); i++) {

    TString nameME = (summaryME[i])->getName();
    LogDebug("FourVectorHLTClient") << "ME = " << nameME << endl;

    TString fullPathToME    = pathsSummaryFilterCountsFolder_ + nameME;
    LogDebug("FourVectorHLTClient") << "fullPathME = " << fullPathToME << endl;

    if(nameME.Contains("Filters_")) {


      TH1F* MEHist = summaryME[i]->getTH1F() ;
      LogDebug("FourVectorHLTClient") << "this is TH1 first bin label = " << MEHist->GetXaxis()->GetBinLabel(1) << endl;
      LogDebug("FourVectorHLTClient") << "is this is TH1 = " << fullPathToME << endl;

      TString nameMEeff           = nameME+"_Eff";
      TString titleMEeff           = nameME+" Efficiency wrt input to L1";

      TH1F* effHist = (TH1F*) MEHist->Clone(nameMEeff.Data());
      effHist->SetTitle(titleMEeff.Data());

      float firstBinContent = MEHist->GetBinContent(1);

      if(firstBinContent !=0 ) {

        effHist->Scale(1./firstBinContent);

      }
      else {

        unsigned int nBins = effHist->GetNbinsX();
        for(unsigned int bin = 0; bin < nBins+1; bin++)  effHist->SetBinContent(bin, 0);

      }

      dbe_->setCurrentFolder(pathsSummaryFilterEfficiencyFolder_.c_str());

      dbe_->book1D(nameMEeff.Data(), effHist);


     } // end if "Filters_" ME

  } // end for MEs

  // Normalize PassPass and PassFail Matrices
  //////////////////////////////////////////////
  vector<string> name;
  name.push_back("All");
  name.push_back("Muon");
  name.push_back("Egamma");
  name.push_back("Tau");
  name.push_back("JetMET");
  name.push_back("Rest");
  name.push_back("Special");

  /// add dataset name and thier triggers to the list 
  vector<string> datasetNames =  hltConfig_.datasetNames() ;

  for (unsigned int i=0;i<datasetNames.size();i++) {

    name.push_back(datasetNames[i]);

  }

  string fullPathToME; 

  for (unsigned int i=0;i<name.size();i++) {

    fullPathToME = pathsSummaryFolder_ +"HLT_"+name[i]+"_PassPass";
    v_ME_HLTPassPass_.push_back( dbe_->get(fullPathToME));

    fullPathToME = pathsSummaryHLTCorrelationsFolder_+"HLT_"+name[i]+"_PassPass_Normalized";
    v_ME_HLTPassPass_Normalized_.push_back( dbe_->get(fullPathToME));

    fullPathToME = pathsSummaryHLTCorrelationsFolder_+"HLT_"+name[i]+"_Pass_Normalized_Any";
    v_ME_HLTPass_Normalized_Any_.push_back( dbe_->get(fullPathToME));

  }
  normalizeHLTMatrix();
  


  // HLT paths
  TObjArray *hltPathNameColl = new TObjArray(100);
  hltPathNameColl->SetOwner();
  std::vector<std::string> fullPathHLTFolders;
 
  // get subdir of the sourceDir_ to get HLT path names
  /////////////////////////////////////////////////////
  dbe_->setCurrentFolder(sourceDir_.Data());
 
  fullPathHLTFolders = dbe_->getSubdirs();
 

  LogDebug("FourVectorHLTClient")<<"endRun: sourceDir_(" << sourceDir_.Data() << ") has " << fullPathHLTFolders.size() << " folders. " << endl;
 
 
  for(unsigned int i=0;i<fullPathHLTFolders.size();i++) {
 
    LogDebug("FourVectorHLTClient")<<"endRun: sourceDir_(" << sourceDir_.Data() << ") folder["<< i << "] = " << fullPathHLTFolders[i] << endl;
    TString hltPath = fullPathHLTFolders[i].substr(fullPathHLTFolders[i].rfind('/')+1, fullPathHLTFolders[i].size());
 
    //*****
    hltPath = removeVersions(hltPath);
    //*****

    ///////////////////////////
    // Efficiencies
    ///////////////////////////

    TString currEffFolder = clientDir_ + "/" + hltPath + "/" + customEffDir_ + "/";

    //*****
    currEffFolder = removeVersions(currEffFolder);
    //*****


    LogDebug("FourVectorHLTClient")<< "Custom Efficiencies dir path = " << currEffFolder << endl;
  
    dbe_->setCurrentFolder(currEffFolder.Data());
 

    hltMEs = dbe_->getContents(fullPathHLTFolders[i]);
    LogDebug("FourVectorHLTClient")<< "Number of MEs for this HLT path = " << hltMEs.size() << endl;
  
    // custom efficiencies
  
    for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair)
    {
  
      //TString numPathName=TString(custompathnamepair->first);
      //if(!hltPath.Contains(numPathName,TString::kExact)) continue;
      //TString numPathName=hltPath;
      //if(!hltPath.Contains(numPathName,TString::kExact)) continue;
      //LogDebug("FourVectorHLTClient")<< "hltPath = " << hltPath <<  "   matchString = " << custompathnamepair->first << endl;
      //if(!hltPath.Contains(TString(custompathnamepair->first),TString::kExact)) continue;
      if(!hltPath.Contains(TString(custompathnamepair->first))) continue;
  
      TString numPathName=hltPath;
      TString denPathName=TString(custompathnamepair->second);

      numPathName = removeVersions(numPathName);



      vector<TString> vStage;
      vStage.push_back(TString("L1"));
      vStage.push_back(TString("On"));
      vStage.push_back(TString("Off"));
      vStage.push_back(TString("Mc"));
      vStage.push_back(TString("L1UM"));
      vStage.push_back(TString("OnUM"));
      vStage.push_back(TString("OffUM"));
      vStage.push_back(TString("McUM"));
  
      vector<TString> vObj;
      vObj.push_back(TString("l1Et"));
      vObj.push_back(TString("onEt"));
      vObj.push_back(TString("offEt"));
      vObj.push_back(TString("mcEt"));
  
  
      // TH1Fs
      for (unsigned int k=0; k<vObj.size();k++) {
       for (unsigned int l=0; l<vStage.size();l++) {
        for (unsigned int m=0; m<vStage.size();m++) {
  
          TString oldHistPathNum;
          TString oldHistPathNumBck;  
          TString oldHistPathDen;
   
          // not the differeence b/w  Num and NumBck, as in OnOff vs OffOn
          oldHistPathNum    = sourceDir_+"/"+hltPath+"/"+numPathName+"_wrt_"+denPathName+"_"+vObj[k]+vStage[l]+vStage[m]; 
          oldHistPathNumBck = sourceDir_+"/"+hltPath+"/"+numPathName+"_wrt_"+denPathName+"_"+vObj[k]+vStage[m]+vStage[l]; 
  
          // In the deominator hist name, we don't have any "UM" substrings, so remove them
          TString tempDenString = vStage[l];
          tempDenString.ReplaceAll("UM","") ;
          oldHistPathDen    = sourceDir_+"/"+hltPath+"/"+numPathName+"_wrt_"+denPathName+"_"+vObj[k]+tempDenString;
     
          MonitorElement *numME    = dbe_->get(oldHistPathNum.Data());
          MonitorElement *numMEBck = dbe_->get(oldHistPathNumBck.Data());
          MonitorElement *denME    = dbe_->get(oldHistPathDen.Data());
   
          LogDebug("FourVectorHLTClient")<< " oldHistPathNum    = " << oldHistPathNum    << endl;
          LogDebug("FourVectorHLTClient")<< " oldHistPathNumBck = " << oldHistPathNumBck << endl;
          LogDebug("FourVectorHLTClient")<< " oldHistPathDen    = " << oldHistPathDen    << endl;
     
          //check if HLTOffline histogram exist
          if ( numME &&  denME ) { 
   
            LogDebug("FourVectorHLTClient")<< "DID find NUM and DEN histograms to derive eff " << vStage[m]<<"To"<<vStage[l]<<" using:" <<endl 
            << " NUM = " << oldHistPathNum  << endl
            << " DEN = " << oldHistPathDen  << endl;
   
          }
          else { 
   
           LogDebug("FourVectorHLTClient")<< "Cannot find NUM and DEN histograms to derive eff " << vStage[m]<<"To"<<vStage[l]<<" using:" <<endl 
           << " NUM = " << oldHistPathNum  << endl
           << " DEN = " << oldHistPathDen  << endl;
   
           if ( numMEBck &&  denME) { 
   
             LogDebug("FourVectorHLTClient")<< "DID find NUM and DEN histograms to derive eff " << vStage[m]<<"To"<<vStage[l]<<" using:" <<endl 
             << " NUM = " << oldHistPathNumBck  << endl
             << " DEN = " << oldHistPathDen  << endl;
   
             numME = numMEBck;
   
           }
           else {
   
             LogDebug("FourVectorHLTClient")<< "Cannot find NUM and DEN histograms to derive eff " << vStage[m]<<"To"<<vStage[l]<<" using:" <<endl 
             << " NUM = " << oldHistPathNumBck  << endl
             << " DEN = " << oldHistPathDen  << endl;
   
             continue;
           }
   
          }
   
          // EtaPhi histos are TH2s, the rest are TH1s
          TH1F * numHist = numME->getTH1F();
          TH1F * denHist = denME->getTH1F();
  
          // build names and title for efficiency histogram
          TString newHistName   = hltPath +"_wrt_" + denPathName +"_"+vObj[k]+"_Eff_"+vStage[m]+"To"+vStage[l];

          // if there is "UM", remove it first,then append it to the end of the name
          if(newHistName.Contains("UM")) {
  
            newHistName.ReplaceAll("UM","");
            newHistName.Append("_UM");
  
          }
  
          TString newHistTitle  = numPathName+" given " + denPathName +"  "+vObj[k]+" Eff  "+vStage[m]+"To"+vStage[l];

          // if there is "UM", remove it first,then append it to the end of the name
          if(newHistTitle.Contains("UM")) {
  
            newHistTitle.ReplaceAll("UM","");
            newHistTitle.Append("_UM");
  
          }
  
          TString newHistPath   = currEffFolder+newHistName;
          
          LogDebug("FourVectorHLTClient")<< "Will make efficiency histogram " << newHistPath << endl;
     
          TH1F* effHist = (TH1F*) numHist->Clone(newHistName.Data());
  
          effHist->SetTitle(newHistTitle.Data());
  
          //effHist->Sumw2(); 
          //denHist->Sumw2();
          //effHist->Divide(numHist,denHist);
  
          //effHist->Divide(numHist,denHist,1.,1.,"B");
  
          calculateRatio(effHist, denHist);
  
          /*
          effHist->Divide(numHist,denHist,1.,1.);
  
          for (int i=0;i<=effHist->GetNbinsX();i++) {
  
            float err = 0;
            if(numHist->GetBinContent(i)>0) err = 1./sqrt(numHist->GetBinContent(i));   // 1/sqrt(number of total)
            effHist->SetBinError(i,err);
  
          }
          */
         
          LogDebug("FourVectorHLTClient")<< "Numerator   hist " << numHist->GetName() << endl;
          LogDebug("FourVectorHLTClient")<< "Denominator hist " << denHist->GetName() << endl;
         
          LogDebug("FourVectorHLTClient")<< "Booking efficiency histogram path " << newHistPath << endl;
          LogDebug("FourVectorHLTClient")<< "Booking efficiency histogram name " << newHistName << endl;
  
          // book eff histos
          dbe_->book1D(newHistName.Data(), effHist);
        
  
        } // end for Stage m
       } // end for Stage l
      } //end for obj k
  
      vObj.clear();
      vObj.push_back(TString("l1Etal1Phi"));
      vObj.push_back(TString("onEtaonPhi"));
      vObj.push_back(TString("offEtaoffPhi"));
      vObj.push_back(TString("mcEtamcPhi"));
 
      for (unsigned int k=0; k<vObj.size();k++) {
       for (unsigned int l=0; l<vStage.size();l++) {
        for (unsigned int m=0; m<vStage.size();m++) {
  
  
         TString oldHistPathNum;
         TString oldHistPathNumBck;  
         TString oldHistPathDen;
  
         // not the differeence b/w  Num and NumBck, as in OnOff vs OffOn
         oldHistPathNum    = sourceDir_+"/"+hltPath+"/"+numPathName+"_wrt_"+denPathName+"_"+vObj[k]+vStage[l]+vStage[m]; 
         oldHistPathNumBck = sourceDir_+"/"+hltPath+"/"+numPathName+"_wrt_"+denPathName+"_"+vObj[k]+vStage[m]+vStage[l]; 

         // In the deominator hist name, we don't have any "UM" substrings, so remove them
         TString tempDenString = vStage[l];
         tempDenString.ReplaceAll("UM","") ;
         oldHistPathDen    = sourceDir_+"/"+hltPath+"/"+numPathName+"_wrt_"+denPathName+"_"+vObj[k]+tempDenString;
    
         MonitorElement *numME    = dbe_->get(oldHistPathNum.Data());
         MonitorElement *numMEBck = dbe_->get(oldHistPathNumBck.Data());
         MonitorElement *denME    = dbe_->get(oldHistPathDen.Data());
  
         LogDebug("FourVectorHLTClient")<< " oldHistPathNum    = " << oldHistPathNum    << endl;
         LogDebug("FourVectorHLTClient")<< " oldHistPathNumBck = " << oldHistPathNumBck << endl;
         LogDebug("FourVectorHLTClient")<< " oldHistPathDen    = " << oldHistPathDen    << endl;
    
         //check if HLTOffline histogram exist
         if ( numME &&  denME ) { 
  
           LogDebug("FourVectorHLTClient")<< "DID find NUM and DEN histograms to derive eff " << vStage[m]<<"To"<<vStage[l]<<" using:" <<endl 
           << " NUM = " << oldHistPathNum  << endl
           << " DEN = " << oldHistPathDen  << endl;
  
         }
         else { 
  
           LogDebug("FourVectorHLTClient")<< "Cannot find NUM and DEN histograms to derive eff " << vStage[m]<<"To"<<vStage[l]<<" using:" <<endl 
           << " NUM = " << oldHistPathNum  << endl
           << " DEN = " << oldHistPathDen  << endl;
  
           if ( numMEBck &&  denME) { 
  
             LogDebug("FourVectorHLTClient")<< "DID find NUM and DEN histograms to derive eff " << vStage[m]<<"To"<<vStage[l]<<" using:" <<endl 
             << " NUM = " << oldHistPathNumBck  << endl
             << " DEN = " << oldHistPathDen  << endl;
             numME = numMEBck;
  
           }
           else {
  
             LogDebug("FourVectorHLTClient")<< "Cannot find NUM and DEN histograms to derive eff " << vStage[m]<<"To"<<vStage[l]<<" using:" <<endl 
             << " NUM = " << oldHistPathNumBck  << endl
             << " DEN = " << oldHistPathDen  << endl;
  
             continue;
 
           }
  
         }
  
         TH2F* numHist = numME->getTH2F();
         TH2F* denHist = denME->getTH2F();
 
         // build names and title for efficiency histogram
 
         TString newHistName   = hltPath +"_wrt_" + denPathName +"_"+vObj[k]+"_Eff_"+vStage[m]+"To"+vStage[l];
 
         // if there is "UM", remove it first,then append it to the end of the name
         if(newHistName.Contains("UM")) {
 
           newHistName.ReplaceAll("UM","");
           newHistName.Append("_UM");
 
         }
 
         TString newHistTitle  = numPathName+" given " + denPathName +"  "+vObj[k]+" Eff  "+vStage[m]+"To"+vStage[l];
         // if there is "UM", remove it first,then append it to the end of the name
         if(newHistTitle.Contains("UM")) {
           
           newHistTitle.ReplaceAll("UM","");
           newHistTitle.Append("_UM");
 
         }
 
         TString newHistPath   = currEffFolder+newHistName;
	 newHistPath = removeVersions(newHistPath);

         LogDebug("FourVectorHLTClient")<< "Will make efficiency histogram " << newHistPath << endl;
    
         TH2F* effHist = (TH2F*) numHist->Clone(newHistName.Data());
 
         effHist->SetTitle(newHistTitle.Data());
         effHist->Sumw2(); 
         //denHist->Sumw2();
         //effHist->Divide(numHist,denHist,1.,1.,"B");
         effHist->Divide(numHist,denHist,1.,1.);
        
         //reportSummaryMap_ = dbe_->book2D(numHist->Divide(denHist));
        
         LogDebug("FourVectorHLTClient")<< "Booking efficiency histogram path " << newHistPath << endl;
         LogDebug("FourVectorHLTClient")<< "Booking efficiency histogram name " << newHistName << endl;
 
         // book eff histos
         dbe_->book2D(newHistName.Data(), effHist);
        
        } // end for Stage m
       } // end for Stage l
      } //end for obj k
 
    } // end for custompathpair
 
  } // end loop over folders for i
 
 
  hltPathNameColl->Delete();

}

//--------------------------------------------------------
void FourVectorHLTClient::endJob(){
}


TH1F * FourVectorHLTClient::get1DHisto(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogDebug("FourVectorHLTClient")<< "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH1F();

}


TH2F * FourVectorHLTClient::get2DHisto(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogDebug("FourVectorHLTClient")<< "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH2F();
}


TProfile2D *  FourVectorHLTClient::get2DProfile(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogDebug("FourVectorHLTClient")<< "ME NOT FOUND." << endl;
   return NULL;
  }

  return me_->getTProfile2D();
}


TProfile *  FourVectorHLTClient::get1DProfile(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogDebug("FourVectorHLTClient")<< "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTProfile();
}

TString FourVectorHLTClient::removeVersions(TString histVersion) {
  for (int ii = 100; ii >= 0; ii--) {
    string ver = "_v";
    string version ="";
    stringstream ss;
    ss << ver << ii;
    ss >> version;
    
    if (histVersion.Contains(version)){
      histVersion.ReplaceAll(version,"");
    }
  }
  return histVersion;
}

void FourVectorHLTClient::calculateRatio(TH1F* effHist, TH1F* denHist) {

  //cout << " NUMERATOR histogram name = " << effHist->GetName() << endl;
  int nBins= effHist->GetNbinsX();
  for (int i=0;i<=nBins;i++) {

    float k = effHist->GetBinContent(i);   // number of pass
    float N = denHist->GetBinContent(i);   // number of total

    float ratio;
    float err;

    //ratio = N ? k / N : 0; 
    if(N>0) ratio = k / N;
    else ratio = 0;

    if(N > 0) {

      if(ratio <= 1) {
        err = sqrt(ratio*(1-ratio)/N);
      }
      //else if(ratio == 1) {
        //err = 1./sqrt(2*N);
      //}
      else {
        err = 0;
      }

      effHist->SetBinContent(i,ratio);
      effHist->SetBinError(i,err);

    }

  }

}


void FourVectorHLTClient::normalizeHLTMatrix() {

    
  for (unsigned int i =0;i<v_ME_HLTPassPass_.size();i++) {

    MonitorElement* ME_HLTPassPass_ = v_ME_HLTPassPass_[i]; 
    MonitorElement* ME_HLTPassPass_Normalized_ = v_ME_HLTPassPass_Normalized_[i]; 
    MonitorElement* ME_HLTPass_Normalized_Any_ = v_ME_HLTPass_Normalized_Any_[i]; 

    if(!ME_HLTPassPass_ || !ME_HLTPassPass_Normalized_ || !ME_HLTPass_Normalized_Any_) return;

    float passCount = 0;
    unsigned int nBinsX = ME_HLTPassPass_->getTH2F()->GetNbinsX();
    unsigned int nBinsY = ME_HLTPassPass_->getTH2F()->GetNbinsY();

    for(unsigned int binX = 0; binX < nBinsX+1; binX++) {
       
      passCount = ME_HLTPassPass_->getTH2F()->GetBinContent(binX,binX);


      for(unsigned int binY = 0; binY < nBinsY+1; binY++) {

        if(passCount != 0) {

          // normalize each bin to number of passCount
          float normalizedBinContentPassPass = (ME_HLTPassPass_->getTH2F()->GetBinContent(binX,binY))/passCount;
          //float normalizedBinContentPassFail = (ME_HLTPassFail_->getTH2F()->GetBinContent(binX,binY))/passCount;

          ME_HLTPassPass_Normalized_->getTH2F()->SetBinContent(binX,binY,normalizedBinContentPassPass);
          //ME_HLTPassFail_Normalized_->getTH2F()->SetBinContent(binX,binY,normalizedBinContentPassFail);

          if(binX == nBinsX) {

            ME_HLTPass_Normalized_Any_->getTH1F()->SetBinContent(binY,normalizedBinContentPassPass);

          }

        }
        else {

          ME_HLTPassPass_Normalized_->getTH2F()->SetBinContent(binX,binY,0);
          //ME_HLTPassFail_Normalized_->getTH2F()->SetBinContent(binX,binY,0);

        } // end if else
     
      } // end for binY

    } // end for binX
  
  } // end for i

}
