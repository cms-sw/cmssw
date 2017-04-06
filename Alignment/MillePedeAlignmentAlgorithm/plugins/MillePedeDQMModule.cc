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



MillePedeDQMModule
::MillePedeDQMModule(const edm::ParameterSet& config) :
  mpReaderConfig_(
    config.getParameter<edm::ParameterSet>("MillePedeFileReader")
  ),

  sigCut_     (mpReaderConfig_.getParameter<double>("sigCut")),
  Xcut_       (mpReaderConfig_.getParameter<double>("Xcut")),
  tXcut_      (mpReaderConfig_.getParameter<double>("tXcut")),
  Ycut_       (mpReaderConfig_.getParameter<double>("Ycut")),
  tYcut_      (mpReaderConfig_.getParameter<double>("tYcut")),
  Zcut_       (mpReaderConfig_.getParameter<double>("Zcut")),
  tZcut_      (mpReaderConfig_.getParameter<double>("tZcut")),
  maxMoveCut_ (mpReaderConfig_.getParameter<double>("maxMoveCut")),
  maxErrorCut_ (mpReaderConfig_.getParameter<double>("maxErrorCut"))
{
}

MillePedeDQMModule
::~MillePedeDQMModule()
{
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

  h_xPos = booker.book1D("Xpos",   "#Delta X;;#mu m", 10, 0, 10.);
  h_xRot = booker.book1D("Xrot",   "#Delta #theta_{X};;#mu rad", 10, 0, 10.);
  h_yPos = booker.book1D("Ypos",   "#Delta Y;;#mu m", 10, 0., 10.);
  h_yRot = booker.book1D("Yrot",   "#Delta #theta_{Y};;#mu rad", 10, 0, 10.);
  h_zPos = booker.book1D("Zpos",   "#Delta Z;;#mu m", 10, 0., 10.);
  h_zRot = booker.book1D("Zrot",   "#Delta #theta_{Z};;#mu rad", 10, 0, 10.);

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

  mpReader_ = std::make_unique<MillePedeFileReader>(mpReaderConfig_, pedeLabeler);
}


void MillePedeDQMModule
::fillExpertHistos()
{

  fillExpertHisto(h_xPos,  Xcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader_->getXobs(),  mpReader_->getXobsErr());
  fillExpertHisto(h_xRot, tXcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader_->getTXobs(), mpReader_->getTXobsErr());

  fillExpertHisto(h_yPos,  Ycut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader_->getYobs(),  mpReader_->getYobsErr());
  fillExpertHisto(h_yRot, tYcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader_->getTYobs(), mpReader_->getTYobsErr());

  fillExpertHisto(h_zPos,  Zcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader_->getZobs(),  mpReader_->getZobsErr());
  fillExpertHisto(h_zRot, tZcut_, sigCut_, maxMoveCut_, maxErrorCut_, mpReader_->getTZobs(), mpReader_->getTZobsErr());

}

void MillePedeDQMModule
::fillExpertHisto(MonitorElement* histo, const double cut, const double sigCut, const double maxMoveCut, const double maxErrorCut,
                  std::array<double, 6> obs, std::array<double, 6> obsErr)
{
  TH1F* histo_0 = histo->getTH1F();

  histo_0->SetMinimum(-(maxMoveCut_));
  histo_0->SetMaximum(  maxMoveCut_);

  for (size_t i = 0; i < obs.size(); ++i) {
    histo_0->SetBinContent(i+1, obs[i]);
    histo_0->SetBinError(i+1, obsErr[i]);
  }
  histo_0->SetBinContent(8,cut);
  histo_0->SetBinContent(9,sigCut);
  histo_0->SetBinContent(10,maxMoveCut);
  histo_0->SetBinContent(11,maxErrorCut);

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
