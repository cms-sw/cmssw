#ifndef SiStripZeroSuppression_h
#define SiStripZeroSuppression_h
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"

class SiStripDigi;
class SiStripRawDigi;

class SiStripZeroSuppression : public edm::stream::EDProducer<>
{
 public:

  explicit SiStripZeroSuppression(const edm::ParameterSet&);
  void produce(edm::Event& , const edm::EventSetup& ) override;

 private:

  enum class RawType { Unknown, VirginRaw, ProcessedRaw, ScopeMode };

  void clearOutputs();
  void putOutputs(edm::Event& evt, const std::string& tagName);

  void processRaw(const edm::DetSetVector<SiStripRawDigi>& input, RawType inType);
  void processHybrid(const edm::DetSetVector<SiStripDigi>& input);
  void storeExtraOutput(uint32_t, int16_t);
  edm::DetSet<SiStripRawDigi> formatRawDigis(const edm::DetSet<SiStripRawDigi>& rawDigis);
  edm::DetSet<SiStripRawDigi> formatRawDigis(uint32_t detId, const std::vector<int16_t>& rawDigis);
  using medians_t = std::vector<std::pair<short,float>>;
  void storeCMN(uint32_t, const medians_t&);
  void storeBaseline(uint32_t, const medians_t&);
  void storeBaselinePoints(uint32_t);

  std::unique_ptr<SiStripRawProcessingAlgorithms> algorithms;

  bool produceRawDigis;
  bool storeCM;
  bool fixCM;
  bool produceCalculatedBaseline;
  bool produceBaselinePoints;
  bool storeInZScollBadAPV;
  bool produceHybridFormat;

  using rawtoken_t = edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi>>;
  using zstoken_t = edm::EDGetTokenT<edm::DetSetVector<SiStripDigi>>;
  std::vector<std::tuple<std::string,RawType,rawtoken_t>> rawInputs;
  std::vector<std::tuple<std::string,zstoken_t>> hybridInputs;

  std::vector<edm::DetSet<SiStripDigi> > output_base;
  std::vector<edm::DetSet<SiStripRawDigi> > output_base_raw;
  std::vector< edm::DetSet<SiStripProcessedRawDigi> > output_apvcm;
  std::vector< edm::DetSet<SiStripProcessedRawDigi> > output_baseline;
  std::vector< edm::DetSet<SiStripDigi> > output_baseline_points;
};
#endif
