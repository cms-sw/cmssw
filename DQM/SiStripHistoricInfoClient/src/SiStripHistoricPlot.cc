#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "CondFormats/DataRecord/interface/SiStripSummaryRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricPlot.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <sys/time.h>
using namespace cms;


/*

THIS IS A DRAFT !!!! 
UNDER DEVELOPMENT !!!!

*/


//--------------------------------------------------------------------------------------
SiStripHistoricPlot::SiStripHistoricPlot( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)),presentRunNr_(0){}
SiStripHistoricPlot::~SiStripHistoricPlot(){}

//--------------------------------------------------------------------------------------
void SiStripHistoricPlot::analyze( const edm::Event& e, const edm::EventSetup& iSetup){}

//--------------------------------------------------------------------------------------
void SiStripHistoricPlot::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){
  //
  edm::ESHandle<SiStripSummary> SiStripSummary_;
  iSetup.get<SiStripSummaryRcd>().get(SiStripSummary_);
//  edm::LogInfo("SiStripHistoricPlot")<<"SiStripSummary for run="<<run.id()<<" SiStripSummary_->getRunNr()="<<SiStripSummary_->getRunNr()<< std::endl;
  if(presentRunNr_ != SiStripSummary_->getRunNr()){ // only analyze the SiStripSummary objects once
    // book histograms
    std::ostringstream oshistoid; TString ohistoid;
    oshistoid.str(""); oshistoid <<"ARun"<<run.id()<< "_ClusterSizesAllDets"; ohistoid=oshistoid.str();
    ClusterSizesAllDets = new TH1F(ohistoid,ohistoid,1,0,1); ClusterSizesAllDets->SetBit(TH1::kCanRebin);
    //
    presentRunNr_ = SiStripSummary_->getRunNr();
    //SiStripSummary_->print();
    fillHistograms(SiStripSummary_);
  }
}

void SiStripHistoricPlot::endRun(const edm::Run& run, const edm::EventSetup& iSetup){
}

void SiStripHistoricPlot::fillHistograms(edm::ESHandle<SiStripSummary> pS){
  std::vector<unsigned int> vdetids; vdetids.clear();
  vdetids = pS->getDetIds(); //  vdetids = activeDets;
  for(unsigned int i =0; i < vdetids.size(); i++){
 
      std::string mean = "ClusterWidth@mean";
      std::string rms = "ClusterWidth@rms";
      std::vector<std::string> items;
        
      std::ostringstream osdetid; osdetid<<vdetids.at(i); TString sdetid = osdetid.str();
      int jbin;
      items.clear();
      items.push_back(mean);
      ClusterSizesAllDets->Fill(sdetid,pS->getSummaryObj(vdetids.at(i),items).back());
      jbin = ClusterSizesAllDets->GetXaxis()->FindBin(sdetid);
      items.clear();
      items.push_back(rms);
      ClusterSizesAllDets->SetBinError(jbin,pS->getSummaryObj(vdetids.at(i),items).back());
     
      // 
      std::vector<uint32_t>::const_iterator aDet = std::find(activeDets.begin(),activeDets.end(), vdetids.at(i));
      if( aDet!=activeDets.end() ){
        std::ostringstream osrunid; osrunid<<presentRunNr_; TString srunid = osrunid.str();
        int ibin; TString ohistoid;
        ohistoid=detHistoTitle(vdetids.at(i),"ClusterSizesAllRuns");
	items.clear();
	items.push_back(mean);
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->Fill(srunid,pS->getSummaryObj(vdetids.at(i), items).back());
        ibin = ((TH1F*) AllDetHistograms->FindObject(ohistoid))->GetXaxis()->FindBin(srunid);
	items.clear();
	items.push_back(rms);
        ((TH1F*) AllDetHistograms->FindObject(ohistoid))->SetBinError(ibin,pS->getSummaryObj(vdetids.at(i), items).back());
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
  }
  outputfile->Write();
  outputfile->Close();
}


TString SiStripHistoricPlot::detHistoTitle(uint32_t detid, std::string description){
  std::ostringstream oshistoid; TString ohistoid;
  oshistoid.str(""); oshistoid <<"DetId"<<detid<<"_"<<description; ohistoid=oshistoid.str();
  return ohistoid;
}

