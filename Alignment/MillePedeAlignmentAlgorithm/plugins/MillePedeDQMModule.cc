/**
 * @package   Alignment/MillePedeAlignmentAlgorithm
 * @file      MillePedeDQMModule.cc
 *
 * @author    Max Stark (max.stark@cern.ch)
 * @date      Feb 19, 2016
 */


/*** header-file ***/
#include "Alignment/MillePedeAlignmentAlgorithm/plugins/MillePedeDQMModule.h"

/*** ROOT objects ***/
#include "TH1F.h"

/*** Core framework functionality ***/
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*** Geometry ***/
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

/*** Alignment ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerPluginFactory.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

/*** Thresholds from DB ***/
#include "CondFormats/DataRecord/interface/AlignPCLThresholdsRcd.h"
 

MillePedeDQMModule
::MillePedeDQMModule(const edm::ParameterSet& config) :
  mpReaderConfig_(
    config.getParameter<edm::ParameterSet>("MillePedeFileReader")
  )
{
}

MillePedeDQMModule
::~MillePedeDQMModule()
{
  delete theThresholds;
}

//=============================================================================
//===   INTERFACE IMPLEMENTATION                                            ===
//=============================================================================

void MillePedeDQMModule
::bookHistograms(DQMStore::IBooker& booker)
{
  edm::LogInfo("MillePedeDQMModule") << "Booking histograms";

  booker.cd();
  booker.setCurrentFolder("AlCaReco/SiPixelAli/");

  h_xPos = booker.book1D("Xpos",   "#Delta X;;#mu m", 30, 0., 30.);
  h_xRot = booker.book1D("Xrot",   "#Delta #theta_{X};;#mu rad", 30, 0., 30.);
  h_yPos = booker.book1D("Ypos",   "#Delta Y;;#mu m", 30, 0., 30.);
  h_yRot = booker.book1D("Yrot",   "#Delta #theta_{Y};;#mu rad", 30, 0., 30.);
  h_zPos = booker.book1D("Zpos",   "#Delta Z;;#mu m", 30, 0., 30.);
  h_zRot = booker.book1D("Zrot",   "#Delta #theta_{Z};;#mu rad", 30, 0., 30.);

  booker.cd();
}


void MillePedeDQMModule
::dqmEndJob(DQMStore::IBooker & booker, DQMStore::IGetter &)
{
  bookHistograms(booker);
  if (mpReader_) {
    mpReader_->read();
  } else {
    throw cms::Exception("LogicError")
      << "@SUB=MillePedeDQMModule::dqmEndJob\n"
      << "Try to read MillePede results before initializing MillePedeFileReader";
  }
  fillExpertHistos();
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

void MillePedeDQMModule
::beginRun(const edm::Run&, const edm::EventSetup& setup) {

  if (!setupChanged(setup)) return;

  edm::ESHandle<TrackerTopology> tTopo;
  setup.get<TrackerTopologyRcd>().get(tTopo);
  edm::ESHandle<GeometricDet> geometricDet;
  setup.get<IdealGeometryRecord>().get(geometricDet);
  edm::ESHandle<PTrackerParameters> ptp;
  setup.get<PTrackerParametersRcd>().get(ptp);

  // take the thresholds from DB
  edm::ESHandle<AlignPCLThresholds> thresholdHandle;
  setup.get<AlignPCLThresholdsRcd>().get(thresholdHandle);
  theThresholds = thresholdHandle.product();

  auto myThresholds = new AlignPCLThresholds();
  myThresholds->setAlignPCLThresholds(theThresholds->getNrecords(),theThresholds->getThreshold_Map());

  TrackerGeomBuilderFromGeometricDet builder;

  const auto trackerGeometry = builder.build(&(*geometricDet), *ptp, &(*tTopo));
  tracker_ = std::make_unique<AlignableTracker>(trackerGeometry, &(*tTopo));

  const std::string labelerPlugin{"PedeLabeler"};
  edm::ParameterSet labelerConfig{};
  labelerConfig.addUntrackedParameter("plugin", labelerPlugin);
  labelerConfig.addUntrackedParameter("RunRangeSelection", edm::VParameterSet{});

  std::shared_ptr<PedeLabelerBase> pedeLabeler{
    PedeLabelerPluginFactory::get()
      ->create(labelerPlugin,
              PedeLabelerBase::TopLevelAlignables(tracker_.get(), nullptr, nullptr),
              labelerConfig)
  };


  mpReader_ = std::make_unique<MillePedeFileReader>(mpReaderConfig_, pedeLabeler, std::shared_ptr<const AlignPCLThresholds>(myThresholds)); 

}


void MillePedeDQMModule
::fillExpertHistos()
{
 
  std::array<double, 6> Xcut_,  sigXcut_,  maxMoveXcut_,  maxErrorXcut_; 
  std::array<double, 6> tXcut_, sigtXcut_, maxMovetXcut_, maxErrortXcut_;
                                                    
  std::array<double, 6> Ycut_,  sigYcut_,  maxMoveYcut_,  maxErrorYcut_; 
  std::array<double, 6> tYcut_, sigtYcut_, maxMovetYcut_, maxErrortYcut_;
                                                    
  std::array<double, 6> Zcut_,  sigZcut_,  maxMoveZcut_,  maxErrorZcut_; 
  std::array<double, 6> tZcut_, sigtZcut_, maxMovetZcut_, maxErrortZcut_;

  auto myMap = theThresholds->getThreshold_Map(); ;
  theThresholds->printAll();

  std::vector<std::string> alignablesList;
  for(auto it = myMap.begin(); it != myMap.end() ; ++it){
    alignablesList.push_back(it->first);
  }

  for (auto &alignable : alignablesList){

    int detIndex = getIndexFromString(alignable);

    Xcut_[detIndex]          = myMap[alignable].getXcut() ;
    sigXcut_[detIndex]       = myMap[alignable].getSigXcut() ;
    maxMoveXcut_[detIndex]   = myMap[alignable].getMaxMoveXcut() ;
    maxErrorXcut_[detIndex]  = myMap[alignable].getErrorXcut() ;

    Ycut_[detIndex]          = myMap[alignable].getYcut() ;
    sigYcut_[detIndex]       = myMap[alignable].getSigYcut() ;
    maxMoveYcut_[detIndex]   = myMap[alignable].getMaxMoveYcut() ; 
    maxErrorYcut_[detIndex]  = myMap[alignable].getErrorYcut() ;
 
    Zcut_[detIndex]          = myMap[alignable].getZcut() ;
    sigZcut_[detIndex]       = myMap[alignable].getSigZcut() ;
    maxMoveZcut_[detIndex]   = myMap[alignable].getMaxMoveZcut() ;
    maxErrorZcut_[detIndex]  = myMap[alignable].getErrorZcut() ;

    tXcut_[detIndex]         = myMap[alignable].getThetaXcut() ;
    sigtXcut_[detIndex]      = myMap[alignable].getSigThetaXcut() ; 
    maxMovetXcut_[detIndex]  = myMap[alignable].getMaxMoveThetaXcut() ;
    maxErrortXcut_[detIndex] = myMap[alignable].getErrorThetaXcut() ;

    tYcut_[detIndex]         = myMap[alignable].getThetaYcut() ;
    sigtYcut_[detIndex]      = myMap[alignable].getSigThetaYcut() ;
    maxMovetYcut_[detIndex]  = myMap[alignable].getMaxMoveThetaYcut() ;
    maxErrortYcut_[detIndex] = myMap[alignable].getErrorThetaYcut() ;
 
    tZcut_[detIndex]         = myMap[alignable].getThetaZcut() ;
    sigtZcut_[detIndex]      = myMap[alignable].getSigThetaYcut() ;
    maxMovetZcut_[detIndex]  = myMap[alignable].getMaxMoveThetaYcut() ;
    maxErrortZcut_[detIndex] = myMap[alignable].getErrorThetaYcut() ;

  }

  fillExpertHisto(h_xPos,  Xcut_, sigXcut_,  maxMoveXcut_,  maxErrorXcut_,  mpReader_->getXobs(),  mpReader_->getXobsErr());
  fillExpertHisto(h_xRot, tXcut_, sigtXcut_, maxMovetXcut_, maxErrortXcut_, mpReader_->getTXobs(), mpReader_->getTXobsErr());

  fillExpertHisto(h_yPos,  Ycut_, sigYcut_,  maxMoveYcut_,  maxErrorYcut_,  mpReader_->getYobs(),  mpReader_->getYobsErr());
  fillExpertHisto(h_yRot, tYcut_, sigtYcut_, maxMovetYcut_, maxErrortYcut_, mpReader_->getTYobs(), mpReader_->getTYobsErr());

  fillExpertHisto(h_zPos,  Zcut_, sigZcut_,  maxMoveZcut_,  maxErrorZcut_,  mpReader_->getZobs(),  mpReader_->getZobsErr());
  fillExpertHisto(h_zRot, tZcut_, sigtZcut_, maxMovetZcut_, maxErrortZcut_, mpReader_->getTZobs(), mpReader_->getTZobsErr());

}

void MillePedeDQMModule
::fillExpertHisto(MonitorElement* histo, 
		  std::array<double, 6> cut, std::array<double, 6> sigCut, 
		  std::array<double, 6> maxMoveCut, std::array<double, 6> maxErrorCut,
                  std::array<double, 6> obs, std::array<double, 6> obsErr)
{
  TH1F* histo_0 = histo->getTH1F();

  double max_ = *std::max_element(maxMoveCut.begin(),maxMoveCut.end());

  histo_0->SetMinimum(-(max_));
  histo_0->SetMaximum(  max_);

  // first 6 bins for movements

  for (size_t i = 0; i < obs.size(); ++i) {
    histo_0->SetBinContent(i+1, obs[i]);
    histo_0->SetBinError(i+1, obsErr[i]); 
  }

  // next 6 bins for cutoffs

  for (size_t i = obs.size(); i < obs.size()*2; ++i) {
    histo_0->SetBinContent(i+1, cut[i-obs.size()]); 	
  }

  // next 6 bins for significances

  for (size_t i = obs.size()*2; i < obs.size()*3; ++i) {
    histo_0->SetBinContent(i+1, sigCut[i-obs.size()*2]);
  }

  // next 6 bins for maximum movements

  for (size_t i = obs.size()*3; i < obs.size()*4; ++i) {
    histo_0->SetBinContent(i+1, maxMoveCut[i-obs.size()*3]);	
  }

  // final 6 bins for maximum errors

  for (size_t i = obs.size()*4; i < obs.size()*5; ++i) {
    histo_0->SetBinContent(i+1, maxErrorCut[i-obs.size()*4]);
  }

}

bool MillePedeDQMModule
::setupChanged(const edm::EventSetup& setup)
{
  bool changed{false};

  if (watchIdealGeometryRcd_.check(setup)) changed = true;
  if (watchTrackerTopologyRcd_.check(setup)) changed = true;
  if (watchPTrackerParametersRcd_.check(setup)) changed = true;

  return changed;
}


int MillePedeDQMModule
::getIndexFromString(const std::string& alignableId)
{
  
  if(alignableId == "TPBHalfBarrelXminus"){
    return 3;
  } else if(alignableId == "TPBHalfBarrelXplus"){
    return 2;
  } else if(alignableId == "TPEHalfCylinderXminusZminus") {
    return 1;
  } else if(alignableId == "TPEHalfCylinderXplusZminus") {
    return 0;
  } else if(alignableId == "TPEHalfCylinderXminusZplus") {
    return 5;
  } else if(alignableId == "TPEHalfCylinderXplusZplus") {
    return 4;
  } else{
    throw cms::Exception("LogicError") 
      << "@SUB=MillePedeDQMModule::getIndexFromString\n"
      << "Retrieving conversion for not supported Alignable partition" 
      << alignableId;
  }
}
