
/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */


#include <DQMOffline/Muon/interface/MuonTrackResidualsTest.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h" 
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Run.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <math.h>
#include "TF1.h"

using namespace edm;
using namespace std;


MuonTrackResidualsTest::MuonTrackResidualsTest(const edm::ParameterSet& ps){
  parameters = ps;

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

  GaussianCriterionName = parameters.getUntrackedParameter<string>("resDistributionTestName","ResidualsDistributionGaussianTest");
  SigmaCriterionName = parameters.getUntrackedParameter<string>("sigmaTestName","ResidualsSigmaInRange"); 
  MeanCriterionName = parameters.getUntrackedParameter<string>("meanTestName","ResidualsMeanInRange"); 
}
void MuonTrackResidualsTest::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter & igetter){

  ////////////////////////////////////////////////////
  ////   BOOK NEW HISTOGRAMS             
  ////////////////////////////////////////////////////
  ibooker.setCurrentFolder("Muons/Tests/trackResidualsTest");
  
  string histName, MeanHistoName, SigmaHistoName,  MeanHistoTitle, SigmaHistoTitle;
  vector<string> type;
  type.push_back("eta");
  type.push_back("theta");
  type.push_back("phi");


  for(unsigned int c=0; c<type.size(); c++){

    MeanHistoName =  "MeanTest_" + type[c]; 
    SigmaHistoName =  "SigmaTest_" + type[c];

    MeanHistoTitle =  "Mean of the #" + type[c] + " residuals distribution"; 
    SigmaHistoTitle =  "Sigma of the #" + type[c] + " residuals distribution"; 
 
    histName = "Res_GlbSta_"+type[c];
    histoNames[type[c]].push_back(histName);
    histName = "Res_TkGlb_"+type[c];
    histoNames[type[c]].push_back(histName);
    histName = "Res_TkSta_"+type[c];
    histoNames[type[c]].push_back(histName);

    
    MeanHistos[type[c]] = ibooker.book1D(MeanHistoName.c_str(),MeanHistoTitle.c_str(),3,0.5,3.5);
    (MeanHistos[type[c]])->setBinLabel(1,"Res_StaGlb",1);
    (MeanHistos[type[c]])->setBinLabel(2,"Res_TkGlb",1);
    (MeanHistos[type[c]])->setBinLabel(3,"Res_TkSta",1);
    
    
    SigmaHistos[type[c]] = ibooker.book1D(SigmaHistoName.c_str(),SigmaHistoTitle.c_str(),3,0.5,3.5);
    (SigmaHistos[type[c]])->setBinLabel(1,"Res_StaGlb",1);  
    (SigmaHistos[type[c]])->setBinLabel(2,"Res_TkGlb",1);
    (SigmaHistos[type[c]])->setBinLabel(3,"Res_TkSta",1);
    
  }
  
  ////////////////////////////////////////////////////
  ////   OPERATIONS WITH OTHER HISTOGRAMS
  ////////////////////////////////////////////////////
  for(map<string, vector<string> > ::const_iterator histo = histoNames.begin();
      histo != histoNames.end();
      histo++) {
   
    for (unsigned int type=0; type< (*histo).second.size(); type++){

      string path = "Muons/MuonRecoAnalyzer/" + (*histo).second[type];
      MonitorElement * res_histo = igetter.get(path);
      if (res_histo) {
 
	// gaussian test
	//	const QReport *GaussianReport = res_histo->getQReport(GaussianCriterionName);
	int BinNumber = type+1;
	float mean = (*res_histo).getMean(1);
	float sigma = (*res_histo).getRMS(1);
	MeanHistos.find((*histo).first)->second->setBinContent(BinNumber, mean);	
	SigmaHistos.find((*histo).first)->second->setBinContent(BinNumber, sigma);
      }
    }
  }


  // Mean test 
  for(map<string, MonitorElement*>::const_iterator hMean = MeanHistos.begin();
      hMean != MeanHistos.end();
      hMean++) {
    const QReport * theMeanQReport = (*hMean).second->getQReport(MeanCriterionName);
    if(theMeanQReport) {
      vector<dqm::me_util::Channel> badChannels = theMeanQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
      }
    }
  }
  
  // Sigma test
  for(map<string, MonitorElement*>::const_iterator hSigma = SigmaHistos.begin();
      hSigma != SigmaHistos.end();
      hSigma++) {
    const QReport * theSigmaQReport = (*hSigma).second->getQReport(SigmaCriterionName);
    if(theSigmaQReport) {
      vector<dqm::me_util::Channel> badChannels = theSigmaQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
      }
    }
  }
}



