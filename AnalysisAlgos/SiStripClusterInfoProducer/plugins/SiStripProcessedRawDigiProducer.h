#ifndef AnalysisAlgos_SiStripClusterInfoProducer_SiStripProcessedRawDigiProducer_H
#define AnalysisAlgos_SiStripClusterInfoProducer_SiStripProcessedRawDigiProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include <memory>
#include <string>

class SiStripGain;
class SiStripProcessedRawDigi;

class SiStripProcessedRawDigiProducer : public edm::EDProducer {

 public:

  explicit SiStripProcessedRawDigiProducer(edm::ParameterSet const&);

 private:

  void produce(edm::Event& e, const edm::EventSetup& es);
  template<class T> std::string findInput(edm::Handle<T>& handle, const std::vector<edm::EDGetTokenT<T> >& tokens, const edm::Event& e);

  void vr_process(const edm::DetSetVector<SiStripRawDigi>&, edm::DetSetVector<SiStripProcessedRawDigi>&);
  void pr_process(const edm::DetSetVector<SiStripRawDigi>&, edm::DetSetVector<SiStripProcessedRawDigi>&);
  void zs_process(const edm::DetSetVector<SiStripDigi>&,    edm::DetSetVector<SiStripProcessedRawDigi>&);
  void common_process( const uint32_t, std::vector<float>&, edm::DetSetVector<SiStripProcessedRawDigi>&);


  std::vector<edm::InputTag> inputTags;
  std::vector<edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > > inputTokensDigi;
  std::vector<edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > > inputTokensRawDigi;
  edm::ESHandle<SiStripGain> gainHandle;

  std::auto_ptr<SiStripPedestalsSubtractor>       subtractorPed;
  std::auto_ptr<SiStripCommonModeNoiseSubtractor> subtractorCMN;

};
#endif

