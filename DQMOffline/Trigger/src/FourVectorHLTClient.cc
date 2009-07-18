/*

   FourVectorHLTClient.cc
   author:  Vladimir Rekovic, U Minn. 
   date of first version: Sept 2008

*/
//$Id: FourVectorHLTClient.cc,v 1.13 2009/06/11 20:22:33 rekovic Exp $

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


      
}

//--------------------------------------------------------
void FourVectorHLTClient::beginJob(const EventSetup& context){


  LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient]: beginJob" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();



  

}

//--------------------------------------------------------
void FourVectorHLTClient::beginRun(const Run& r, const EventSetup& context) {

  LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient]: beginRun" << endl;
	/*
  TString summaryFolder = clientDir_ + TString("/reportSummaryContents/");
  TString summaryPath = summaryFolder + TString("reportSummary");

  dbe_->setCurrentFolder(summaryFolder.Data());
  

  reportSummary_ = dbe_->get(summaryPath.Data());
  if ( reportSummary_ ) {
      dbe_->removeElement(reportSummary_->getName()); 
  }

  reportSummary_ = dbe_->bookFloat("reportSummary");

  //initialize reportSummary to 1
  if (reportSummary_) reportSummary_->Fill(-999);
	*/

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

  


  //reportSummary = average of report summaries of each system
  
 
}

//--------------------------------------------------------
void FourVectorHLTClient::endRun(const Run& r, const EventSetup& context){
 LogDebug("FourVectorHLTClient")<<"FourVectorHLTClient:: endLuminosityBlock "  << endl;

 // QTests
 ////////////////////////////
 Comp2RefKolmogorov * kl_test_ = new Comp2RefKolmogorov("my_kolm");

 // HLT paths
 TObjArray *hltPathNameColl = new TObjArray(100);
 hltPathNameColl->SetOwner();
 std::vector<std::string> fullPathHLTFolders;

 // get subdir of the sourceDir_ to get HLT path names
 //////////////////////////////////////////////////
 dbe_->setCurrentFolder(sourceDir_.Data());

 fullPathHLTFolders = dbe_->getSubdirs();

 LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient] endRun: sourceDir_(" << sourceDir_.Data() << ") has " << fullPathHLTFolders.size() << " folders. " << endl;

 //if(! dbe_->containsAnyMonitorable(sourceDir_.Data())) {

   //LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient] endLuminosityBlock: sourceDir_(" << sourceDir_.Data() << ") has no MEs " << endl;
    //return;
 //}

 //LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient] endLuminosityBlock: Interesting dir has MEs " << endl;
 for(unsigned int i=0;i<fullPathHLTFolders.size();i++) {

   LogDebug("FourVectorHLTClient")<<"[FourVectorHLTClient] endRun: sourceDir_(" << sourceDir_.Data() << ") folder["<< i << "] = " << fullPathHLTFolders[i] << endl;
	 TString hltPath = fullPathHLTFolders[i].substr(fullPathHLTFolders[i].rfind('/')+1, fullPathHLTFolders[i].size());

   ///////////////////////////
   // Efficiencies
   ///////////////////////////
   TString currEffFolder = clientDir_ + "/" + hltPath + "/" + customEffDir_ + "/";
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
 
 
       //TString oldHistPathNum = sourceDir_+TString("/")+curHltPath->String()+" "+curHltPath->String()+TString("_")+vObj[k]+TString("L1On"); 
       //TString oldHistPathDen = sourceDir_+TString("/")+curHltPath->String()+" "+curHltPath->String()+TString("_")+vObj[k]+TString("L1"); 
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
   
			 // ****************************************************************
			 //
			 //  V. Rekovic:  WARNING:  This needs attention.  When running, the code complains to be removing non-existant MEs
			 //
			 // ****************************************************************
       //check if booked HLTClient histogram exist
       //if ( dbe_->get(newHistPath.Data()) ) {
         //LogDebug("FourVectorHLTClient")<< "Will remove ME " << newHistPath.Data() << endl;
         //dbe_->removeElement(newHistPath.Data());
       //}
       
       TH1F* effHist = (TH1F*) numHist->Clone(newHistName.Data());

       effHist->SetTitle(newHistTitle.Data());
       //effHist->Sumw2(); 
			 //denHist->Sumw2();
       //effHist->Divide(numHist,denHist);
       effHist->Divide(numHist,denHist,1.,1.);
	     for (int i=0;i<=effHist->GetNbinsX();i++) {
			  float err = 0;
				if(numHist->GetBinContent(i)>0) err = 1./sqrt(numHist->GetBinContent(i));   // 1/sqrt(number of total)
		    effHist->SetBinError(i,err);
			 }
			 /*
			 */
       //effHist->Divide(numHist,denHist,1.,1.,"B");
			 //calculateRatio(effHist, denHist);

			 //TGraph effTGraph;
			 
       LogDebug("FourVectorHLTClient")<< "Numerator   hist " << numHist->GetName() << endl;
       LogDebug("FourVectorHLTClient")<< "Denominator hist " << denHist->GetName() << endl;
			 //TGraphAsymmErrors effTGraph(numHist, denHist, "");
       
   
       //reportSummaryMap_ = dbe_->book1D(numHist->Divide(denHist));
       
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
 
 
       //TString oldHistPathNum = sourceDir_+TString("/")+curHltPath->String()+" "+curHltPath->String()+TString("_")+vObj[k]+TString("L1On"); 
       //TString oldHistPathDen = sourceDir_+TString("/")+curHltPath->String()+" "+curHltPath->String()+TString("_")+vObj[k]+TString("L1"); 
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
 			 
        LogDebug("FourVectorHLTClient")<< "Will make efficiency histogram " << newHistPath << endl;
   
			 // ****************************************************************
			 //
			 //  V. Rekovic:  WARNING:  This needs attention.  When running, the code complains to be removing non-existant MEs
			 //
			 // ****************************************************************
       //check if booked HLTClient histogram exist
       //if ( dbe_->get(newHistPath.Data()) ) {
         //LogDebug("FourVectorHLTClient")<< "Will remove ME " << newHistPath.Data() << endl;
         //dbe_->removeElement(newHistPath.Data());
       //}
       
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
  delete kl_test_;


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

void FourVectorHLTClient::calculateRatio(TH1F* effHist, TH1F* denHist) {

	//cout << " NUMERATOR histogram name = " << effHist->GetName() << endl;
	int nBins= effHist->GetNbinsX();
	for (int i=0;i<=nBins;i++) {
    
		float k = effHist->GetBinContent(i);   // number of pass
		float N = denHist->GetBinContent(i);   // number of total

		float ratio;
		float err;;

		if(k > N) {

		  ratio = k/N;
		  err = 1.;

		}
		else if(N == 0) {      

		  ratio = 0;;
		  err = 0;

		}
		else {

		  ratio = k/N;
		  //err = (1.0/N)*sqrt(k*(1-k/N));
		  err = (1.0/sqrt(N));
		
		}

		//cout << "bin " << i << " num = " << k << "  den = " << N << "  ratio = " << ratio << "  err = " << err << endl;
		effHist->SetBinContent(i,ratio);
		effHist->SetBinError(i,err);
	
	}


}
