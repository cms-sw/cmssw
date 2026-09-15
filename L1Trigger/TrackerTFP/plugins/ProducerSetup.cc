#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackerTFP/interface/Setup.h"

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerSetup
   *  \brief  Class to produce setup of Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class ProducerSetup : public edm::ESProducer {
  public:
    ProducerSetup(const edm::ParameterSet& iConfig);
    ~ProducerSetup() override = default;
    std::unique_ptr<Setup> produce(const trackerDTC::SetupRcd& setupRcd);

  private:
    Setup::Config config_;
    edm::ESGetToken<trackerDTC::Setup, trackerDTC::SetupRcd> esGetToken_;
  };

  ProducerSetup::ProducerSetup(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
    const edm::ParameterSet& pSetTFP = iConfig.getParameter<edm::ParameterSet>("TFP");
    config_.tfpWidthPhi0 = pSetTFP.getParameter<int>("WidthPhi0");
    config_.tfpWidthInvR = pSetTFP.getParameter<int>("WidthInvR");
    config_.tfpWidthCot = pSetTFP.getParameter<int>("WidthCot");
    config_.tfpWidthZ0 = pSetTFP.getParameter<int>("WidthZ0");
    config_.tfpNumChannel = pSetTFP.getParameter<int>("NumChannel");
    const edm::ParameterSet& pSetGP = iConfig.getParameter<edm::ParameterSet>("GeometricProcessor");
    config_.gpNumBinsPhiT = pSetGP.getParameter<int>("NumBinsPhiT");
    config_.gpNumBinsZT = pSetGP.getParameter<int>("NumBinsZT");
    config_.gpDepthMemory = pSetGP.getParameter<int>("DepthMemory");
    const edm::ParameterSet& pSetHT = iConfig.getParameter<edm::ParameterSet>("HoughTransform");
    config_.htNumBinsInv2R = pSetHT.getParameter<int>("NumBinsInv2R");
    config_.htNumBinsPhiT = pSetHT.getParameter<int>("NumBinsPhiT");
    config_.htMinLayers = pSetHT.getParameter<int>("MinLayers");
    config_.htDepthMemory = pSetHT.getParameter<int>("DepthMemory");
    const edm::ParameterSet& pSetCTB = iConfig.getParameter<edm::ParameterSet>("CleanTrackBuilder");
    config_.ctbNumBinsInv2R = pSetCTB.getParameter<int>("NumBinsInv2R");
    config_.ctbNumBinsPhiT = pSetCTB.getParameter<int>("NumBinsPhiT");
    config_.ctbNumBinsCot = pSetCTB.getParameter<int>("NumBinsCot");
    config_.ctbNumBinsZT = pSetCTB.getParameter<int>("NumBinsZT");
    config_.ctbMinLayers = pSetCTB.getParameter<int>("MinLayers");
    config_.ctbMaxTracks = pSetCTB.getParameter<int>("MaxTracks");
    config_.ctbMaxStubs = pSetCTB.getParameter<int>("MaxStubs");
    config_.ctbDepthMemory = pSetCTB.getParameter<int>("DepthMemory");
    const edm::ParameterSet& pSetKF = iConfig.getParameter<edm::ParameterSet>("KalmanFilter");
    config_.kfNumWorker = pSetKF.getParameter<int>("NumWorker");
    config_.kfMaxTracks = pSetKF.getParameter<int>("MaxTracks");
    config_.kfMinLayers = pSetKF.getParameter<int>("MinLayers");
    config_.kfMaxLayers = pSetKF.getParameter<int>("MaxLayers");
    config_.kfMaxGaps = pSetKF.getParameter<int>("MaxGaps");
    config_.kfMaxSeedingLayer = pSetKF.getParameter<int>("MaxSeedingLayer");
    config_.kfNumSeedStubs = pSetKF.getParameter<int>("NumSeedStubs");
    config_.kfShiftChi20 = pSetKF.getParameter<int>("ShiftChi20");
    config_.kfShiftChi21 = pSetKF.getParameter<int>("ShiftChi21");
    config_.kfCutChi2 = pSetKF.getParameter<double>("CutChi2");
    config_.kfWidthChi2 = pSetKF.getParameter<int>("WidthChi2");
    const edm::ParameterSet& pSetDR = iConfig.getParameter<edm::ParameterSet>("DuplicateRemoval");
    config_.drDepthMemory = pSetDR.getParameter<int>("DepthMemory");
    const edm::ParameterSet& pSetTQ = iConfig.getParameter<edm::ParameterSet>("TrackQuality");
    config_.tqNumChannel = pSetTQ.getParameter<int>("NumChannel");
    config_.tqWidthChi21 = pSetTQ.getParameter<int>("WidthChi21");
    config_.tqWidthChi20 = pSetTQ.getParameter<int>("WidthChi20");
    config_.tqBaseShiftChi21 = pSetTQ.getParameter<int>("BaseShiftChi21");
    config_.tqBaseShiftChi20 = pSetTQ.getParameter<int>("BaseShiftChi20");
    config_.tqWidthInvV0 = pSetTQ.getParameter<int>("WidthInvV0");
    config_.tqWidthInvV1 = pSetTQ.getParameter<int>("WidthInvV1");
    config_.tqWidthMVA = pSetTQ.getParameter<int>("WidthMVA");
    config_.tqBinEdges = pSetTQ.getParameter<std::vector<int>>("BinEdges");
  }

  std::unique_ptr<Setup> ProducerSetup::produce(const trackerDTC::SetupRcd& setupRcd) {
    const trackerDTC::Setup* setup = &setupRcd.get(esGetToken_);
    return std::make_unique<Setup>(config_, setup);
  }
}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerSetup);
