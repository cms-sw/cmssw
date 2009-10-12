
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
// $Id: BuildTrackerMap.cc,v 1.2 2009/08/14 09:22:06 amagnan Exp $
//

#include <sstream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>

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
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  void read(bool aMechView);



  //input file name
  std::string fileName_;
  //name of the tkHistoMap to extract
  std::vector<std::string> tkHistoMapNameVec_;
  //do mechanical view or not
  bool mechanicalView_;
  //folder name for histograms in DQMStore
  std::string folderName_;
  //print debug messages when problems are found: 1=error debug, 2=light debug, 3=full debug
  unsigned int printDebug_;

  std::vector<TkHistoMap*> tkHistoMapVec_;

  edm::ParameterSet pset_;
  TrackerMap * tkmap_;

};


//
// Constructors and destructor
//

BuildTrackerMapPlugin::BuildTrackerMapPlugin(const edm::ParameterSet& iConfig)
  : fileName_(iConfig.getUntrackedParameter<std::string>("InputFileName","DQMStore.root")),
    tkHistoMapNameVec_(iConfig.getUntrackedParameter<std::vector<std::string> >("TkHistoMapNameVec")),
    mechanicalView_(iConfig.getUntrackedParameter<bool>("MechanicalView",true)),
    folderName_(iConfig.getUntrackedParameter<std::string>("HistogramFolderName","DQMData/")),
    printDebug_(iConfig.getUntrackedParameter<unsigned int>("PrintDebugMessages",1)),
    pset_(iConfig.getParameter<edm::ParameterSet>("TkmapParameters"))
{

  read(mechanicalView_);

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
}


//
// Member functions
//

/*Check that is possible to load in tkhistomaps histograms already stored in a DQM root file (if the folder and name are known)*/
void BuildTrackerMapPlugin::read(bool aMechView){

  edm::Service<DQMStore>().operator->()->open(fileName_);  

  for (unsigned int i(0); i<tkHistoMapNameVec_.size(); i++){

    TkHistoMap *tkHistoMap = new TkHistoMap();
    tkHistoMap->loadTkHistoMap(folderName_,tkHistoMapNameVec_.at(i),aMechView);
    
    tkHistoMapVec_.push_back(tkHistoMap);
  }

  std::cout << "Maps read with success." << std::endl;
 	    
}


// ------------ method called to for each event  ------------
void
BuildTrackerMapPlugin::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{

  static bool firstEvent = true;

  edm::ESHandle<SiStripFedCabling> fedcabling;
  iSetup.get<SiStripFedCablingRcd>().get(fedcabling );
  
  if (firstEvent) tkmap_ = new TrackerMap(pset_,fedcabling);

  firstEvent = false;

}//analyze method

// ------------ method called once each job just before starting event loop  ------------
void 
BuildTrackerMapPlugin::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BuildTrackerMapPlugin::endJob()
{
  //edm::ESHandle<SiStripFedCabling> pDD1;
  //iSetup.get<SiStripFedCablingRcd>().get(pDD1);
  std::cout << "Processing endjob with " << tkHistoMapNameVec_.size()<< " elements." << std::endl;

  for (unsigned int i(0); i<tkHistoMapNameVec_.size(); i++){

    std::cout << "Processing element " << i << ": " << tkHistoMapNameVec_.at(i) << std::endl;
    
    //(pset_,pDD1); 
    tkmap_->setPalette(1);
    tkmap_->showPalette(1);
    tkHistoMapVec_.at(i)->dumpInTkMap(tkmap_);
    tkmap_->printall(true,0,255,"tmap");
    tkmap_->save(true,0,0,tkHistoMapNameVec_.at(i)+std::string(".png"));
    tkmap_->save_as_fedtrackermap(true,0,0,tkHistoMapNameVec_.at(i)+std::string("_FED.png"));

  }

}


// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BuildTrackerMapPlugin);
