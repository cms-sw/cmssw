/****************************************************************************
 *
 *  CondFormats/PPSObjects/plugins/PPSPixelTopologyESSource.cc
 *
 *  Description :  - Loads PPSPixelTopology from the PPSPixelTopologyESSource_cfi.py
 *                   config file.
 *
 *
 * Author: F.Ferro ferro@ge.infn.it
 * 
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"
#include "CondFormats/DataRecord/interface/PPSPixelTopologyRcd.h"

#include <memory>

/**
 * \brief Loads PPSPixelTopology from a config file.
 **/

class PPSPixelTopologyESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  PPSPixelTopologyESSource(const edm::ParameterSet&);
  ~PPSPixelTopologyESSource() override = default;

  std::unique_ptr<PPSPixelTopology> produce(const PPSPixelTopologyRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  /// Set PPS Topology parameters to their values from config
  void setPPSPixelTopology(const edm::ParameterSet&);
  ///  Fill PPSPixelTopology object
  std::unique_ptr<PPSPixelTopology> fillPPSPixelTopology();

  // Topology parameters
  std::string runType_;
  double pitch_simY_;
  double pitch_simX_;
  double thickness_;
  unsigned short no_of_pixels_simX_;
  unsigned short no_of_pixels_simY_;
  unsigned short no_of_pixels_;
  double simX_width_;
  double simY_width_;
  double dead_edge_width_;
  double active_edge_sigma_;
  double phys_active_edge_dist_;

protected:
  /// sets infinite validity of this data
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;
};

//----------------------------------------------------------------------------------------------------

PPSPixelTopologyESSource::PPSPixelTopologyESSource(const edm::ParameterSet& iConfig)
    : runType_(""),
      pitch_simY_(0.),
      pitch_simX_(0.),
      thickness_(0.),
      no_of_pixels_simX_(0.),
      no_of_pixels_simY_(0.),
      no_of_pixels_(0.),
      simX_width_(0.),
      simY_width_(0.),
      dead_edge_width_(0.),
      active_edge_sigma_(0.),
      phys_active_edge_dist_(0.) {
  setPPSPixelTopology(iConfig);
  setWhatProduced(this);
  findingRecord<PPSPixelTopologyRcd>();
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<PPSPixelTopology> PPSPixelTopologyESSource::produce(const PPSPixelTopologyRcd&) {
  auto topo = fillPPSPixelTopology();

  edm::LogInfo("PPS") << "PixelTopologyESSource::produce \n" << *topo;

  return topo;
}

//----------------------------------------------------------------------------------------------------

void PPSPixelTopologyESSource::setPPSPixelTopology(const edm::ParameterSet& iConfig) {
  runType_ = iConfig.getParameter<std::string>("RunType");
  pitch_simY_ = iConfig.getParameter<double>("PitchSimY");
  pitch_simX_ = iConfig.getParameter<double>("PitchSimX");
  thickness_ = iConfig.getParameter<double>("thickness");
  no_of_pixels_simX_ = iConfig.getParameter<int>("noOfPixelSimX");
  no_of_pixels_simY_ = iConfig.getParameter<int>("noOfPixelSimY");
  no_of_pixels_ = iConfig.getParameter<int>("noOfPixels");
  simX_width_ = iConfig.getParameter<double>("simXWidth");
  simY_width_ = iConfig.getParameter<double>("simYWidth");
  dead_edge_width_ = iConfig.getParameter<double>("deadEdgeWidth");
  active_edge_sigma_ = iConfig.getParameter<double>("activeEdgeSigma");
  phys_active_edge_dist_ = iConfig.getParameter<double>("physActiveEdgeDist");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<PPSPixelTopology> PPSPixelTopologyESSource::fillPPSPixelTopology() {
  auto p = std::make_unique<PPSPixelTopology>();

  p->setRunType(runType_);
  p->setPitchSimY(pitch_simY_);
  p->setPitchSimX(pitch_simX_);
  p->setThickness(thickness_);
  p->setNoPixelsSimX(no_of_pixels_simX_);
  p->setNoPixelsSimY(no_of_pixels_simY_);
  p->setNoPixels(no_of_pixels_);
  p->setSimXWidth(simX_width_);
  p->setSimYWidth(simY_width_);
  p->setDeadEdgeWidth(dead_edge_width_);
  p->setActiveEdgeSigma(active_edge_sigma_);
  p->setPhysActiveEdgeDist(phys_active_edge_dist_);
  p->setActiveEdgeX(simX_width_ / 2. - phys_active_edge_dist_);
  p->setActiveEdgeY(simY_width_ / 2. - phys_active_edge_dist_);

  return p;
}

//----------------------------------------------------------------------------------------------------

void PPSPixelTopologyESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& key,
                                              const edm::IOVSyncValue& iosv,
                                              edm::ValidityInterval& oValidity) {
  edm::LogInfo("PPS") << ">> PPSPixelTopologyESSource::setIntervalFor(" << key.name() << ")\n"
                      << "    run=" << iosv.eventID().run() << ", event=" << iosv.eventID().event();

  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

//----------------------------------------------------------------------------------------------------

void PPSPixelTopologyESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("RunType", "Run3");
  desc.add<double>("PitchSimY", 150e-3);
  desc.add<double>("PitchSimX", 100e-3);
  desc.add<double>("thickness", 0.23);
  desc.add<int>("noOfPixelSimX", 160);
  desc.add<int>("noOfPixelSimY", 104);
  desc.add<int>("noOfPixels", 160 * 104);
  desc.add<double>("simXWidth", 16.6);
  desc.add<double>("simYWidth", 16.2);
  desc.add<double>("deadEdgeWidth", 200e-3);
  desc.add<double>("activeEdgeSigma", 0.02);
  desc.add<double>("physActiveEdgeDist", 0.150);

  descriptions.add("ppsPixelTopologyESSource", desc);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(PPSPixelTopologyESSource);
