#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiStripPerformanceSummaryRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricPlot.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <sys/time.h>
using namespace cms;

//--------------------------------------------------------------------------------------
SiStripHistoricPlot::SiStripHistoricPlot( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)),presentRunNr_(0){}
SiStripHistoricPlot::~SiStripHistoricPlot(){}

//--------------------------------------------------------------------------------------
void SiStripHistoricPlot::analyze( const edm::Event& e, const edm::EventSetup& iSetup){
/*
  edm::LogInfo("SiStripHistoricPlot") << "[SiStripHistoricPlot::analyze] Start Reading SiStripPerformanceSummary" << std::endl;
  edm::ESHandle<SiStripPerformanceSummary> SiStripPerformanceSummary_;
  iSetup.get<SiStripPerformanceSummaryRcd>().get(SiStripPerformanceSummary_);
  if(presentRunNr_ != SiStripPerformanceSummary_->getRunNr()){ // only analyze the SiStripPerformanceSummary objects once
    presentRunNr_ = SiStripPerformanceSummary_->getRunNr();
    SiStripPerformanceSummary_->print();
  }
*/
//  std::vector<uint32_t> all_detids;
//  all_detids.clear();
//  SiStripPerformanceSummary_->getDetIds(all_detids);
//  if( all_detids.size()>0 ) SiStripPerformanceSummary_->print(all_detids[0]);
//  SiStripPerformanceSummary_->printall(); // print all summaries
}

//--------------------------------------------------------------------------------------
void SiStripHistoricPlot::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){
  //
  edm::ESHandle<SiStripPerformanceSummary> SiStripPerformanceSummary_;
  iSetup.get<SiStripPerformanceSummaryRcd>().get(SiStripPerformanceSummary_);
//  edm::LogInfo("SiStripHistoricPlot")<<"SiStripPerformanceSummary for run="<<run.id()<<" SiStripPerformanceSummary_->getRunNr()="<<SiStripPerformanceSummary_->getRunNr()<< std::endl;
  if(presentRunNr_ != SiStripPerformanceSummary_->getRunNr()){ // only analyze the SiStripPerformanceSummary objects once
    // book histograms
    std::ostringstream oshistoid; TString ohistoid;
    oshistoid.str(""); oshistoid <<"ARun"<<run.id()<< "_ClusterSizesAllDets"; ohistoid=oshistoid.str();
    ClusterSizesAllDets = new TH1F(ohistoid,ohistoid,1,0,1); ClusterSizesAllDets->SetBit(TH1::kCanRebin);
    oshistoid.str(""); oshistoid <<"ARun"<<run.id()<< "_ClusterChargeAllDets"; ohistoid=oshistoid.str();
    ClusterChargeAllDets = new TH1F(ohistoid,ohistoid,1,0,1); ClusterChargeAllDets->SetBit(TH1::kCanRebin);
    oshistoid.str(""); oshistoid <<"ARun"<<run.id()<< "_OccupancyAllDets"; ohistoid=oshistoid.str();
    OccupancyAllDets = new TH1F(ohistoid,ohistoid,1,0,1); OccupancyAllDets->SetBit(TH1::kCanRebin);
    oshistoid.str(""); oshistoid <<"ARun"<<run.id()<< "_PercentNoisyStripsAllDets"; ohistoid=oshistoid.str();
    PercentNoisyStripsAllDets = new TH1F(ohistoid,ohistoid,1,0,1); PercentNoisyStripsAllDets->SetBit(TH1::kCanRebin);
    //
    presentRunNr_ = SiStripPerformanceSummary_->getRunNr();
    SiStripPerformanceSummary_->print();
    fillHistograms(SiStripPerformanceSummary_);
  }
}

void SiStripHistoricPlot::endRun(const edm::Run& run, const edm::EventSetup& iSetup){
}

void SiStripHistoricPlot::fillHistograms(edm::ESHandle<SiStripPerformanceSummary> pS){
  std::vector<uint32_t> vdetids; vdetids.clear();
  pS->getDetIds(vdetids); //  vdetids = activeDets;
  for(std::vector<uint32_t>::const_iterator idet = vdetids.begin(); idet != vdetids.end(); ++idet){
    std::vector<float> allValues; allValues.clear();
    pS->getSummary(*idet, allValues);
    if(allValues.size()==7){
      std::ostringstream osdetid; osdetid<<*idet; TString sdetid = osdetid.str();
      ClusterSizesAllDets->Fill(sdetid,allValues[0]);
      ClusterChargeAllDets->Fill(sdetid,allValues[2]);
      OccupancyAllDets->Fill(sdetid,allValues[4]);
      PercentNoisyStripsAllDets->Fill(sdetid,allValues[6]);
      // 
      std::vector<uint32_t>::const_iterator aDet = std::find(activeDets.begin(),activeDets.end(), *idet);
      if( aDet!=activeDets.end() ){
        std::ostringstream osrunid; osrunid<<presentRunNr_; TString srunid = osrunid.str();
        int ibin; TString ohistoid;
        ohistoid=detHistoTitle(*idet,"ClusterSizesAllRuns");
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->Fill(srunid,allValues[0]);
        ibin = ((TH1F*) AllDetHistograms->FindObject(ohistoid))->GetXaxis()->FindBin(srunid);
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->SetBinError(ibin,allValues[1]);
        ohistoid=detHistoTitle(*idet,"ClusterChargeAllRuns");
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->Fill(srunid,allValues[2]);
        ibin = ((TH1F*) AllDetHistograms->FindObject(ohistoid))->GetXaxis()->FindBin(srunid);
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->SetBinError(ibin,allValues[3]);
        ohistoid=detHistoTitle(*idet,"OccupancyAllRuns");
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->Fill(srunid,allValues[4]);
        ibin = ((TH1F*) AllDetHistograms->FindObject(ohistoid))->GetXaxis()->FindBin(srunid);
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->SetBinError(ibin,allValues[5]);
        ohistoid=detHistoTitle(*idet,"PercentNoisyStripsAllRuns");
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->Fill(srunid,allValues[6]);
      }
    }else{
      edm::LogError("WrongSize")<<" performance summary has wrong size allValues.size()="<<allValues.size();
    }
  }

  //
  ClusterSizesAllDets->LabelsDeflate("X");
  ClusterChargeAllDets->LabelsDeflate("X");
  OccupancyAllDets->LabelsDeflate("X");
  PercentNoisyStripsAllDets->LabelsDeflate("X");
}


void SiStripHistoricPlot::beginJob(const edm::EventSetup& iSetup){
  AllDetHistograms = new TObjArray();
  outputfile = new TFile("historic_plot.root","RECREATE");
  activeDets.clear();
/*
  activeDets.push_back(369345800);
  activeDets.push_back(369345804);
  activeDets.push_back(369346052);
  activeDets.push_back(369346056);
  activeDets.push_back(369346060);
  activeDets.push_back(369346308);
  activeDets.push_back(369346312);
  activeDets.push_back(369346316);
  activeDets.push_back(369346564);
  activeDets.push_back(369346568);
  activeDets.push_back(436309262);
  activeDets.push_back(436309265);
  activeDets.push_back(436309266);
*/
  // take from eventSetup the SiStripDetCabling object
  edm::ESHandle<SiStripDetCabling> tkmechstruct;
  iSetup.get<SiStripDetCablingRcd>().get(tkmechstruct);
  // get list of active detectors from SiStripDetCabling - this will change and be taken from a SiStripDetControl object
  tkmechstruct->addActiveDetectorsRawIds(activeDets);
  //
  for(std::vector<uint32_t>::const_iterator idet = activeDets.begin(); idet != activeDets.end(); ++idet){
     TString ohistoid;
     ohistoid = detHistoTitle(*idet,"ClusterSizesAllRuns");
     AllDetHistograms->Add( new TH1F(ohistoid,ohistoid,1,0,1));
     ((TH1F*) AllDetHistograms->FindObject(ohistoid))->SetBit(TH1::kCanRebin);
     ohistoid = detHistoTitle(*idet,"ClusterChargeAllRuns");
     AllDetHistograms->Add( new TH1F(ohistoid,ohistoid,1,0,1));
     ((TH1F*) AllDetHistograms->FindObject(ohistoid))->SetBit(TH1::kCanRebin);
     ohistoid = detHistoTitle(*idet,"OccupancyAllRuns");
     AllDetHistograms->Add( new TH1F(ohistoid,ohistoid,1,0,1));
     ((TH1F*) AllDetHistograms->FindObject(ohistoid))->SetBit(TH1::kCanRebin);
     ohistoid = detHistoTitle(*idet,"PercentNoisyStripsAllRuns");
     AllDetHistograms->Add( new TH1F(ohistoid,ohistoid,1,0,1));
     ((TH1F*) AllDetHistograms->FindObject(ohistoid))->SetBit(TH1::kCanRebin);
  }

// contiguous maps
  std::map<uint32_t, unsigned int> allContiguous; tkmechstruct->getAllDetectorsContiguousIds(allContiguous);
  std::map<uint32_t, unsigned int> activeContiguous; tkmechstruct->getActiveDetectorsContiguousIds(activeContiguous);

  std::cout<<"Contiguous"<<"allContiguous.size()="<<allContiguous.size()<<std::endl;
  for(std::map<uint32_t, unsigned int>::const_iterator deco=allContiguous.begin(); deco!=allContiguous.end(); ++deco){
    std::cout<<"Contiguous"<<"allContiguous "<<deco->first<<" -  "<<deco->second<<std::endl;
  }

  std::cout<<"Contiguous"<<"activeContiguous.size()="<<activeContiguous.size()<<std::endl;
  for(std::map<uint32_t, unsigned int>::const_iterator deco=activeContiguous.begin(); deco!=activeContiguous.end(); ++deco){
    std::cout<<"Contiguous"<<"activeContiguous "<<deco->first<<" -  "<<deco->second<<std::endl;
  }
}


void SiStripHistoricPlot::endJob(){
  for(std::vector<uint32_t>::const_iterator idet = activeDets.begin(); idet != activeDets.end(); ++idet){
    ((TH1F*) AllDetHistograms->FindObject(detHistoTitle(*idet,"ClusterSizesAllRuns")))->LabelsDeflate("X");
    ((TH1F*) AllDetHistograms->FindObject(detHistoTitle(*idet,"ClusterChargeAllRuns")))->LabelsDeflate("X");
    ((TH1F*) AllDetHistograms->FindObject(detHistoTitle(*idet,"OccupancyAllRuns")))->LabelsDeflate("X");
    ((TH1F*) AllDetHistograms->FindObject(detHistoTitle(*idet,"PercentNoisyStripsAllRuns")))->LabelsDeflate("X");
  }
  outputfile->Write();
  outputfile->Close();
}


TString SiStripHistoricPlot::detHistoTitle(uint32_t detid, std::string description){
  std::ostringstream oshistoid; TString ohistoid;
  oshistoid.str(""); oshistoid <<"DetId"<<detid<<"_"<<description; ohistoid=oshistoid.str();
  return ohistoid;
}

