#ifndef AnalysisAlgos_SiStripClusterInfoProducer_SiStripProcessedRawDigiProducer_H
#define AnalysisAlgos_SiStripClusterInfoProducer_SiStripProcessedRawDigiProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include <memory>
#include <string>

class SiStripProcessedRawDigi;

class SiStripProcessedRawDigiProducer : public edm::EDProducer {
public:
  explicit SiStripProcessedRawDigiProducer(edm::ParameterSet const&);

private:
  void produce(edm::Event& e, const edm::EventSetup& es) override;
  template <class T>
  std::string findInput(edm::Handle<T>& handle, const std::vector<edm::EDGetTokenT<T> >& tokens, const edm::Event& e);

  void vr_process(const edm::DetSetVector<SiStripRawDigi>&,
                  edm::DetSetVector<SiStripProcessedRawDigi>&,
                  const SiStripGain&);
  void pr_process(const edm::DetSetVector<SiStripRawDigi>&,
                  edm::DetSetVector<SiStripProcessedRawDigi>&,
                  const SiStripGain&);
  void zs_process(const edm::DetSetVector<SiStripDigi>&,
                  edm::DetSetVector<SiStripProcessedRawDigi>&,
                  const SiStripGain&);
  void common_process(const uint32_t,
                      std::vector<float>&,
                      edm::DetSetVector<SiStripProcessedRawDigi>&,
                      const SiStripGain&);

  std::vector<edm::InputTag> inputTags_;
  std::vector<edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > > inputTokensDigi_;
  std::vector<edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > > inputTokensRawDigi_;
  edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken_;

  std::unique_ptr<SiStripPedestalsSubtractor> subtractorPed_;
  std::unique_ptr<SiStripCommonModeNoiseSubtractor> subtractorCMN_;
};
#endif
