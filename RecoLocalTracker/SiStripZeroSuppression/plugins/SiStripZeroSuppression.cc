#include "RecoLocalTracker/SiStripZeroSuppression/plugins/SiStripZeroSuppression.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include "FWCore/Utilities/interface/transform.h"
#include <memory>

SiStripZeroSuppression::SiStripZeroSuppression(edm::ParameterSet const& conf)
  : algorithms(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
    produceRawDigis(conf.getParameter<bool>("produceRawDigis")),
    storeCM(conf.getParameter<bool>("storeCM")),
    fixCM(conf.getParameter<bool>("fixCM")),
    produceCalculatedBaseline(conf.getParameter<bool>("produceCalculatedBaseline")),
    produceBaselinePoints(conf.getParameter<bool>("produceBaselinePoints")),
    storeInZScollBadAPV(conf.getParameter<bool>("storeInZScollBadAPV")),
    produceHybridFormat(conf.getParameter<bool>("produceHybridFormat"))
{
  for ( const auto& inputTag : conf.getParameter<std::vector<edm::InputTag> >("RawDigiProducersList") )
  {
    const auto& tagName = inputTag.instance();
    produces<edm::DetSetVector<SiStripDigi>>(tagName);
    if (produceRawDigis)
      produces<edm::DetSetVector<SiStripRawDigi>>(tagName);
    if (storeCM)
      produces<edm::DetSetVector<SiStripProcessedRawDigi>>("APVCM"+tagName);
    if (produceCalculatedBaseline)
      produces<edm::DetSetVector<SiStripProcessedRawDigi>>("BADAPVBASELINE"+tagName);
    if (produceBaselinePoints)
      produces<edm::DetSetVector<SiStripDigi>>("BADAPVBASELINEPOINTS"+tagName);

    RawType inputType = RawType::Unknown;
    if        ( tagName == "ProcessedRaw" ) {
      inputType = RawType::ProcessedRaw;
      if (produceHybridFormat) throw cms::Exception("Processed Raw Cannot be converted in hybrid Format");
    } else if ( tagName == "VirginRaw" ) {
      inputType = RawType::VirginRaw;
    } else if ( tagName == "ScopeMode" ) {
      inputType = RawType::ScopeMode;
      if (produceHybridFormat) throw cms::Exception("Scope Mode cannot be converted in hybrid Format");
    }
    if ( RawType::Unknown != inputType ) {
      rawInputs.emplace_back(tagName, inputType, consumes<edm::DetSetVector<SiStripRawDigi>>(inputTag));
    } else if ( tagName == "ZeroSuppressed" ) {
      hybridInputs.emplace_back(tagName, consumes<edm::DetSetVector<SiStripDigi>>(inputTag));
    } else {
      throw cms::Exception("Unknown input type") << tagName << " unknown.  "
        << "SiStripZeroZuppression can only process types \"VirginRaw\", \"ProcessedRaw\" and \"ZeroSuppressed\"";
    }
  }

  if ( produceHybridFormat && ( "HybridEmulation" != conf.getParameter<edm::ParameterSet>("Algorithms").getParameter<std::string>("APVInspectMode") ) )
    throw cms::Exception("Invalid option") << "When producing data in the hybrid format, the APV restorer must be configured with APVInspectMode='HybridEmulation'";

  if ( ! ( rawInputs.empty() && hybridInputs.empty() ) ) {
    output_base.reserve(16000);
    if (produceRawDigis) output_base_raw.reserve(16000);
    if (storeCM) output_apvcm.reserve(16000);
    if (produceCalculatedBaseline) output_baseline.reserve(16000);
    if (produceBaselinePoints) output_baseline_points.reserve(16000);
  }
}

void SiStripZeroSuppression::produce(edm::Event& e, const edm::EventSetup& es)
{
  algorithms->initialize(es, e);

  for ( const auto& input : rawInputs ) {
    clearOutputs();
    edm::Handle<edm::DetSetVector<SiStripRawDigi>> inDigis;
    e.getByToken(std::get<rawtoken_t>(input), inDigis);
    if ( ! inDigis->empty() )
      processRaw(*inDigis, std::get<RawType>(input));
    putOutputs(e, std::get<std::string>(input));
  }
  for ( const auto& input : hybridInputs ) {
    clearOutputs();
    edm::Handle<edm::DetSetVector<SiStripDigi>> inDigis;
    e.getByToken(std::get<zstoken_t>(input), inDigis);
    if ( ! inDigis->empty() ) {
      processHybrid(*inDigis);
    }
    putOutputs(e, std::get<std::string>(input));
  }
}

inline void SiStripZeroSuppression::clearOutputs()
{
  output_base.clear();
  output_base_raw.clear();
  output_baseline.clear();
  output_baseline_points.clear();
  output_apvcm.clear();
}
inline void SiStripZeroSuppression::putOutputs(edm::Event& evt, const std::string& tagName)
{
  evt.put(std::make_unique<edm::DetSetVector<SiStripDigi>>(output_base), tagName);
  if (produceRawDigis)
    evt.put(std::make_unique<edm::DetSetVector<SiStripRawDigi>>(output_base_raw), tagName);
  if (produceCalculatedBaseline)
    evt.put(std::make_unique<edm::DetSetVector<SiStripProcessedRawDigi>>(output_baseline), "BADAPVBASELINE"+tagName);
  if (produceBaselinePoints)
    evt.put(std::make_unique<edm::DetSetVector<SiStripDigi>>(output_baseline_points), "BADAPVBASELINEPOINTS"+tagName);
  if (storeCM)
    evt.put(std::make_unique<edm::DetSetVector<SiStripProcessedRawDigi>>(output_apvcm), "APVCM"+tagName);
}

inline void SiStripZeroSuppression::processRaw(const edm::DetSetVector<SiStripRawDigi>& input, RawType inType)
{
  for ( const auto& rawDigis : input ) {
    edm::DetSet<SiStripDigi> suppressedDigis(rawDigis.id);

    int16_t nAPVflagged = 0;
    if ( RawType::ProcessedRaw == inType ) {
      nAPVflagged = algorithms->suppressProcessedRawData(rawDigis, suppressedDigis);
    } else if ( RawType::ScopeMode == inType) {
      nAPVflagged = algorithms->suppressVirginRawData(rawDigis, suppressedDigis);
    } else if ( RawType::VirginRaw == inType ) {
      if ( produceHybridFormat ) {
        nAPVflagged = algorithms->convertVirginRawToHybrid(rawDigis, suppressedDigis);
      } else {
        nAPVflagged = algorithms->suppressVirginRawData(rawDigis, suppressedDigis);
      }
    }

    storeExtraOutput(rawDigis.id, nAPVflagged);
    if (!suppressedDigis.empty() && (storeInZScollBadAPV || nAPVflagged ==0))
      output_base.push_back(std::move(suppressedDigis));

    if (produceRawDigis && nAPVflagged > 0) {
      output_base_raw.push_back(formatRawDigis(rawDigis));
    }
  }
}

inline void SiStripZeroSuppression::processHybrid(const edm::DetSetVector<SiStripDigi>& input)
{
  for ( const auto& inDigis : input ) {
    edm::DetSet<SiStripDigi> suppressedDigis(inDigis.id);

    std::vector<int16_t> rawDigis;
    const auto nAPVflagged = algorithms->suppressHybridData(inDigis, suppressedDigis, rawDigis);

    storeExtraOutput(inDigis.id, nAPVflagged);
    if (!suppressedDigis.empty() && (storeInZScollBadAPV || nAPVflagged ==0))
      output_base.push_back(std::move(suppressedDigis));

    if (produceRawDigis && nAPVflagged > 0) {
      output_base_raw.push_back(formatRawDigis(inDigis.id, rawDigis));
    }
  }
}

inline edm::DetSet<SiStripRawDigi> SiStripZeroSuppression::formatRawDigis(const edm::DetSet<SiStripRawDigi>& rawDigis)
{
  edm::DetSet<SiStripRawDigi> outRawDigis(rawDigis.id);
  outRawDigis.reserve(rawDigis.size());
  const std::vector<bool>& apvf = algorithms->getAPVFlags();
  uint32_t strip=0;
  for ( const auto rawDigi : rawDigis ) {
    int16_t apvN = strip/128;
    if (apvf[apvN]) outRawDigis.push_back(rawDigi);
    else outRawDigis.push_back(SiStripRawDigi(0));
    ++strip;
  }
  return outRawDigis;
}

inline edm::DetSet<SiStripRawDigi> SiStripZeroSuppression::formatRawDigis(uint32_t detId, const std::vector<int16_t>& rawDigis)
{
  edm::DetSet<SiStripRawDigi> outRawDigis(detId);
  outRawDigis.reserve(rawDigis.size());
  const std::vector<bool>& apvf = algorithms->getAPVFlags();
  uint32_t strip=0;
  for ( const auto rawDigi : rawDigis ) {
    int16_t apvN = strip/128;
    if (apvf[apvN]) outRawDigis.push_back(SiStripRawDigi(rawDigi));
    else outRawDigis.push_back(SiStripRawDigi(0));
    ++strip;
  }
  return outRawDigis;
}

inline void SiStripZeroSuppression::storeExtraOutput(uint32_t id, int16_t nAPVflagged)
{
  const auto& vmedians = algorithms->getAPVsCM();
  if (storeCM) storeCMN(id, vmedians);
  if (nAPVflagged > 0){
    if (produceCalculatedBaseline) storeBaseline(id, vmedians);
    if (produceBaselinePoints) storeBaselinePoints(id);
  }
}

inline void SiStripZeroSuppression::storeBaseline(uint32_t id, const medians_t& vmedians)
{
  const auto& baselinemap = algorithms->getBaselineMap();

  edm::DetSet<SiStripProcessedRawDigi> baselineDetSet(id);
  baselineDetSet.reserve(vmedians.size()*128);
  for ( const auto& vmed : vmedians ) {
    const uint16_t apvN = vmed.first;
    const float median = vmed.second;
    auto itBaselineMap = baselinemap.find(apvN);
    if ( baselinemap.end() == itBaselineMap ) {
      for (size_t strip=0; strip < 128; ++strip)
        baselineDetSet.push_back(SiStripProcessedRawDigi(median));
    } else {
      for (size_t strip=0; strip < 128; ++strip)
        baselineDetSet.push_back(SiStripProcessedRawDigi((itBaselineMap->second)[strip]));
    }
  }

  if ( ! baselineDetSet.empty() )
    output_baseline.push_back(baselineDetSet);
}

inline void SiStripZeroSuppression::storeBaselinePoints(uint32_t id)
{
  edm::DetSet<SiStripDigi> baspointDetSet(id);
  for ( const auto& itBaselinePointVect : algorithms->getSmoothedPoints() ) {
    const uint16_t apvN = itBaselinePointVect.first;
    for ( const auto& itBaselinePointMap : itBaselinePointVect.second ) {
      const uint16_t bpstrip = itBaselinePointMap.first + apvN*128;
      const int16_t  bp = itBaselinePointMap.second;
      baspointDetSet.push_back(SiStripDigi(bpstrip, bp+128));
    }
  }

  if ( ! baspointDetSet.empty() )
    output_baseline_points.push_back(std::move(baspointDetSet));
}

inline void SiStripZeroSuppression::storeCMN(uint32_t id, const medians_t& vmedians)
{
  std::vector<bool> apvf(6, false);
  if (fixCM) {
    const auto& apvFlagged = algorithms->getAPVFlags();
    std::copy(std::begin(apvFlagged), std::end(apvFlagged), std::begin(apvf));
  }

  edm::DetSet<SiStripProcessedRawDigi> apvDetSet(id);
  short apvNb=0;
  for ( const auto& vmed : vmedians ) {
    if ( vmed.first > apvNb ) {
      for ( int i{0}; i < vmed.first-apvNb; ++i ) {
	apvDetSet.push_back(SiStripProcessedRawDigi(-999.));
	apvNb++;
      }
    }

    if ( fixCM && apvf[vmed.first] ) {
      apvDetSet.push_back(SiStripProcessedRawDigi(-999.));
    } else {
      apvDetSet.push_back(SiStripProcessedRawDigi(vmed.second));
    }
    apvNb++;
  }

  if(!apvDetSet.empty())
    output_apvcm.push_back(std::move(apvDetSet));
}
