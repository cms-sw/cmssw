
// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      BuildTrackerMapPlugin
// 
/**\class BuildTrackerMapPlugin BuildTrackerMap.cc DQM/SiStripMonitorHardware/plugins/BuildTrackerMap.cc

 Description: DQM source application to monitor common mode for SiStrip data
*/
//
//         Created:  2009/07/22
// $Id: BuildTrackerMap.cc,v 1.2 2012/11/27 15:55:53 speer Exp $
//

#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>

#include "TCanvas.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TPaveStats.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include "CommonTools/TrackerMap/interface/TmApvPair.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h" 

#include "DQMServices/Core/interface/DQMStore.h"

//
// Class declaration
//

class BuildTrackerMapPlugin : public edm::EDAnalyzer
{
 public:

  explicit BuildTrackerMapPlugin(const edm::ParameterSet&);
  ~BuildTrackerMapPlugin();
 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  void read(bool aMechView,
	    std::string aFile,
	    std::vector<TkHistoMap*> & aTkMapVec,
	    std::vector<bool> & aValidVec);
  void subtractMap(TkHistoMap*& aResult,
		   TkHistoMap*& aSubtr);



  //input file names
  std::string fileName_;
  //do mechanical view or not
  bool mechanicalView_;
  //folder name for histograms in DQMStore
  std::string folderName_;
  //print debug messages when problems are found: 1=error debug, 2=light debug, 3=full debug
  unsigned int printDebug_;

  bool doDiff_;
  std::string fileNameDiff_;

  std::vector<TkHistoMap*> tkHistoMapVec_;
  std::vector<TkHistoMap*> tkHistoMapVecDiff_;
  
  //name of the tkHistoMap to extract
  std::vector<std::string> tkHistoMapNameVec_;
  std::vector<double> minVal_;
  std::vector<double> maxVal_;

  std::vector<bool> isValidMap_;
  std::vector<bool> isValidMapDiff_;

  edm::ParameterSet pset_;
  std::vector<TrackerMap*> tkmap_;

};


//
// Constructors and destructor
//

BuildTrackerMapPlugin::BuildTrackerMapPlugin(const edm::ParameterSet& iConfig)
  : fileName_(iConfig.getUntrackedParameter<std::string>("InputFileName","DQMStore.root")),
    mechanicalView_(iConfig.getUntrackedParameter<bool>("MechanicalView",true)),
    folderName_(iConfig.getUntrackedParameter<std::string>("HistogramFolderName","DQMData/")),
    printDebug_(iConfig.getUntrackedParameter<unsigned int>("PrintDebugMessages",1)),
    doDiff_(iConfig.getUntrackedParameter<bool>("DoDifference",false)),
    fileNameDiff_(iConfig.getUntrackedParameter<std::string>("InputFileNameForDiff","DQMStore.root")),
    tkHistoMapNameVec_(iConfig.getUntrackedParameter<std::vector<std::string> >("TkHistoMapNameVec")),
    minVal_(iConfig.getUntrackedParameter<std::vector<double> >("MinValueVec")),
    maxVal_(iConfig.getUntrackedParameter<std::vector<double> >("MaxValueVec")),
    pset_(iConfig.getParameter<edm::ParameterSet>("TkmapParameters"))
{

  read(mechanicalView_,fileName_,tkHistoMapVec_,isValidMap_);
  if (doDiff_) read(mechanicalView_,fileNameDiff_,tkHistoMapVecDiff_,isValidMapDiff_);


//   for (unsigned int i(0); i<34; i++){
//     if (i<4) histName_[i] << "TIB/layer_" << i+1 << "/" << tkDetMapName_ << "_TIB_L" << i+1;
//     else if (i<7)  histName_[i] << "TID/side_1/wheel_" << i-3 << "/" << tkDetMapName_ << "_TIDM_D" << i-3;
//     else if (i<10)  histName_[i] << "TID/side_2/wheel_" << i-6 << "/" << tkDetMapName_ << "_TIDP_D" << i-6;
//     else if (i<16) histName_[i] << "TOB/layer_" << i-9 << "/" << tkDetMapName_ << "_TOB_L" << i-9;
//     else if (i<25) histName_[i] << "TEC/side_1/wheel_" << i-15 << "/" << tkDetMapName_ << "_TECM_W" << i-15;
//     else if (i<34) histName_[i] << "TEC/side_2/wheel_" << i-24 << "/" << tkDetMapName_ << "_TECP_W" << i-24;

//     std::cout << "histName[" << i << "] =" << histName_[i] << std::endl;

//   } 
  
}

BuildTrackerMapPlugin::~BuildTrackerMapPlugin()
{

  tkHistoMapVec_.clear();
  if (doDiff_) tkHistoMapVecDiff_.clear();
}


//
// Member functions
//

/*Check that is possible to load in tkhistomaps histograms already stored in a DQM root file (if the folder and name are known)*/
void BuildTrackerMapPlugin::read(bool aMechView,
				 std::string aFile,
				 std::vector<TkHistoMap*> & aTkMapVec,
				 std::vector<bool> & aValidVec)
{
  
  DQMStore * lDqmStore = edm::Service<DQMStore>().operator->();
  lDqmStore->open(aFile);  
  std::vector<TkHistoMap *> tkHistoMap;

  unsigned int nHists = tkHistoMapNameVec_.size();
  tkHistoMap.resize(nHists,0);
  aValidVec.resize(nHists,true);

  std::string dirName = folderName_;
  if (dirName == "") {
    dirName += "Run ";
    dirName += aFile.substr(aFile.find_last_of("_")+5,6);
    dirName += "/SiStrip/Run summary";
    std::cout << " -- DirName = " << dirName << std::endl;
   }

  //lDqmStore->setCurrentFolder(dirName);
  //lDqmStore->showDirStructure();
  
  unsigned int nFailTot=0;
  unsigned int nTotTot = 0;
  for (unsigned int i(0); i<nHists; i++){

    tkHistoMap[i] = new TkHistoMap();

    tkHistoMap[i]->loadTkHistoMap(dirName,tkHistoMapNameVec_.at(i),aMechView);
    
    std::vector<MonitorElement*>& lMaps = tkHistoMap[i]->getAllMaps();

    std::cout << " -- map " << i << ", nHistos = " << lMaps.size() << std::endl;
    unsigned int nFail=0;
    unsigned int nTot=0;
 
    for (unsigned int im(0); im<lMaps.size(); im++){
      if (!lMaps[im]) {
	std::cout << " -- Failed to get element " << im << " for map " << i << std::endl;
	nFail++;
	nFailTot++;
      }
      nTot++;
      nTotTot++;
    }

    if (nFail == nTot) aValidVec[i] = false;
    aTkMapVec.push_back(tkHistoMap[i]);
  }

  if (nFailTot < nTotTot) std::cout << " - " << nTotTot-nFailTot << "/" << nTotTot 
			      << " histomaps read with success for file ." << aFile << std::endl;
  else {
    std::cout << " - Failed to read any map for file " << aFile << ". Exiting line ... " << __LINE__ << std::endl;
     exit(1);
  }

//   //get list of detid for which |deltaRMS(APV0-APV1)|>1
//   unsigned int lHistoNumber = 35;
//   TkDetMap lTkDetMap;
//   std::ofstream list,listRms0,listRms1;
//   list.open("./cmBadModuleList.dat",std::ios::out);
//   listRms0.open("./cmBadModuleList_rms0.dat",std::ios::out);
//   listRms1.open("./cmBadModuleList_rms1.dat",std::ios::out);
//   if (!list || !listRms0 || !listRms1) {
//     std::cout << "Warning, can't open output file to write bad module list !" << std::endl;
//     exit(1);
//   }

//   TCanvas *lCan = new TCanvas("lCan","",1);
//   TH1F *p_deltaMean = new TH1F("p_deltaMean",";CM_{mean}(APV0)-CM_{mean}(APV1)",500,-2,2);
//   TH1F *p_deltaRMS = new TH1F("p_deltaRMS",";CM_{RMS}(APV0)-CM_{RMS}(APV1)",500,0,3);
//   TH1F *p_MeanAPV0 = new TH1F("p_MeanAPV0",";CM_{mean}(APV0)",500,100,140);
//   //TH1F *p_MeanAPV1 = new TH1F("p_MeanAPV1",";CM_{mean}(APV1)",500,100,140);
//   TH1F *p_RMSAPV0 = new TH1F("p_RMSAPV0",";CM_{RMS}(APV0)",500,0,10);
//   //TH1F *p_RMSAPV1 = new TH1F("p_RMSAPV1",";CM_{RMS}(APV1)",500,0,10);



//   gStyle->SetOptStat(1111111);

//   for(unsigned int layer=1;layer<lHistoNumber;++layer){
//     std::vector<uint32_t> dets;
//     lTkDetMap.getDetsForLayer(layer,dets);
//     for(size_t i=0;i<dets.size();++i){
//       if(dets[i]>0){
// 	//if(tkHistoMap[5]->getEntries(dets[i])>0 && tkHistoMap[5]->getValue(dets[i])) {
// 	if(nHists > 3){
// 	  if (tkHistoMap[3]->getValue(dets[i]) > 1) {
// 	    list << dets[i] << " " << tkHistoMap[3]->getValue(dets[i]) << std::endl;
// 	  }
// 	p_deltaRMS->Fill(tkHistoMap[3]->getValue(dets[i]));
// 	}
// 	p_MeanAPV0->Fill(tkHistoMap[0]->getValue(dets[i]));
// 	//p_MeanAPV1->Fill(tkHistoMap[1]->getValue(dets[i]));
// 	p_RMSAPV0->Fill(tkHistoMap[1]->getValue(dets[i]));
// 	if (tkHistoMap[1]->getValue(dets[i]) > 2)
// 	  listRms0 << dets[i] << " " << tkHistoMap[1]->getValue(dets[i]) << std::endl;
// 	//p_RMSAPV1->Fill(tkHistoMap[3]->getValue(dets[i]));
// 	//if (tkHistoMap[3]->getValue(dets[i]) > 2)
// 	//listRms1 << dets[i] << " " << tkHistoMap[3]->getValue(dets[i]) << std::endl;

// 	if(nHists > 2) p_deltaMean->Fill(tkHistoMap[2]->getValue(dets[i]));
//       }
//     }
//   }
//   list.close();
//   listRms0.close();
//   listRms1.close();

//   lCan->cd();
//   p_deltaRMS->Draw();
//   //lCan->Print("./deltaRMStotal.png");
//   lCan->Print("./deltaRMStotal.C");

//   p_deltaMean->Draw();
//   lCan->Update();
//   lCan->Print("./deltaMeantotal.C");

//   TPaveStats *statBox[2] = {0,0};  
//   statBox[0] = (TPaveStats*)p_MeanAPV0->FindObject("stats");
//   //statBox[1] = (TPaveStats*)p_MeanAPV1->FindObject("stats");

//   p_MeanAPV0->Draw();
//   //p_MeanAPV1->SetLineColor(2);
//   //p_MeanAPV1->Draw("same");
//   if (statBox[0]) statBox[0]->Draw("same");
//   if (statBox[1]) { 
//     statBox[1]->SetLineColor(2);
//     statBox[1]->SetTextColor(2);
//     statBox[1]->Draw("same");
//   }
//   lCan->Update();
//   lCan->Print("./meanAPVstotal.C");

//   statBox[0] = (TPaveStats*)p_RMSAPV0->FindObject("stats");
//   //statBox[1] = (TPaveStats*)p_RMSAPV1->FindObject("stats");

//   p_RMSAPV0->Draw();
//   //p_RMSAPV1->SetLineColor(2);
//   //p_RMSAPV1->Draw("same");
//   if (statBox[0]) statBox[0]->Draw("same");
//   if (statBox[1]) { 
//     statBox[1]->SetLineColor(2);
//     statBox[1]->SetTextColor(2);
//     statBox[1]->Draw("same");
//   }
//   lCan->Update();
//   lCan->Print("./rmsAPVstotal.C");


}


// ------------ method called to for each event  ------------
void
BuildTrackerMapPlugin::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{

  static bool firstEvent = true;

  edm::ESHandle<SiStripFedCabling> fedcabling;
  iSetup.get<SiStripFedCablingRcd>().get(fedcabling );
  
  if (firstEvent) {
    for (unsigned int i(0); i<tkHistoMapNameVec_.size(); i++){
      tkmap_.push_back(new TrackerMap(pset_,&(*fedcabling)));
    }

  }

  firstEvent = false;

  std::cout << "End of analyze method: tkmap_ size = " << tkmap_.size() << std::endl;

}//analyze method

// ------------ method called once each job just before starting event loop  ------------
void 
BuildTrackerMapPlugin::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
BuildTrackerMapPlugin::endJob()
{
  //edm::ESHandle<SiStripFedCabling> pDD1;
  //iSetup.get<SiStripFedCablingRcd>().get(pDD1);
  std::cout << "Processing endjob with " << tkHistoMapNameVec_.size()<< " elements." << std::endl;

  assert (minVal_.size() == tkHistoMapNameVec_.size());
  assert (maxVal_.size() == tkHistoMapNameVec_.size());

  for (unsigned int i(0); i<tkHistoMapNameVec_.size(); i++){

    std::cout << "Processing element " << i << ": " << tkHistoMapNameVec_.at(i) << std::endl;
    std::cout << "Min, max = " << minVal_.at(i) << " " << maxVal_.at(i) << std::endl;

    TrackerMap* lTkMap = tkmap_.at(i);

    if (!lTkMap) {
      std::cout << "tkmap_ is NULL for element " << i << "... continuing ..." << std::endl;
      continue;
    }

    subtractMap(tkHistoMapVec_.at(i),tkHistoMapVecDiff_.at(i));


    //(pset_,pDD1); 
    lTkMap->setPalette(1);
    lTkMap->showPalette(1);
    if (!tkHistoMapVec_.at(i) || !isValidMap_.at(i)) {
      std::cout << "Warning, tkHistoMap is invalid for element " << i << "... continuing ..." << std::endl;
      continue;
    }
    tkHistoMapVec_.at(i)->dumpInTkMap(lTkMap);

    //to print all figures to create fancy view
    //lTkMap->printall(true,0,255,tkHistoMapNameVec_.at(i));
    lTkMap->save(true,
		 minVal_.at(i),
		 maxVal_.at(i),
		 tkHistoMapNameVec_.at(i)+std::string(".png"));
    lTkMap->save_as_fedtrackermap(true,
				  minVal_.at(i),
				  maxVal_.at(i),
				  tkHistoMapNameVec_.at(i)+std::string("_FED.png"));

  }

}


void BuildTrackerMapPlugin::subtractMap(TkHistoMap *& aResult,
					TkHistoMap *& aSubtr)
{
  
  std::vector<MonitorElement*>& lMaps = aResult->getAllMaps();
  std::vector<MonitorElement*>& lMapsDiff = aSubtr->getAllMaps();

  assert(lMaps.size() == lMapsDiff.size());  

    for (unsigned int im(0); im<lMaps.size(); im++){
      if (!lMaps[im] || !lMapsDiff[im]) {
	std::cout << " -- Failed to get element " << im << " for maps." << std::endl;
      }
      else {
	(lMaps[im]->getTProfile2D())->Add(lMapsDiff[im]->getTProfile2D(),-1);
      }
    }

}



// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BuildTrackerMapPlugin);
