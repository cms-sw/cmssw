#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

class SiStripMergeZeroSuppression : public edm::stream::EDProducer<>
{
public:
  explicit SiStripMergeZeroSuppression(const edm::ParameterSet& conf);
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;
private:
  std::unique_ptr<SiStripRawProcessingAlgorithms> m_algorithms;

  using rawtoken_t = edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi>>;
  using zstoken_t = edm::EDGetTokenT<edm::DetSetVector<SiStripDigi>>;
  rawtoken_t m_rawDigisToMerge;
  zstoken_t m_zsDigisToMerge;
};


SiStripMergeZeroSuppression::SiStripMergeZeroSuppression(const edm::ParameterSet& conf)
  : m_algorithms(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms")))
{
  m_rawDigisToMerge = consumes<edm::DetSetVector<SiStripRawDigi>>(conf.getParameter<edm::InputTag>("DigisToMergeVR"));
  m_zsDigisToMerge = consumes<edm::DetSetVector<SiStripDigi>>(conf.getParameter<edm::InputTag>("DigisToMergeZS"));
  produces<edm::DetSetVector<SiStripDigi>>("ZeroSuppressed");
}

void SiStripMergeZeroSuppression::produce(edm::Event& event, const edm::EventSetup& /*eventSetup*/)
{
  LogTrace("SiStripMergeZeroSuppression::produce") << "Starting merging " << "\n";
  edm::Handle<edm::DetSetVector<SiStripDigi>> inputdigi;
  edm::Handle<edm::DetSetVector<SiStripRawDigi>> inputraw;
  event.getByToken(m_rawDigisToMerge, inputdigi);
  event.getByToken(m_zsDigisToMerge, inputraw);

  LogTrace("SiStripMergeZeroSuppression::produce") << inputdigi->size() << " " << inputraw->size() << "\n";
  if ( ! inputraw->empty() ) {
    std::vector<edm::DetSet<SiStripDigi>> outputdigi(inputdigi->begin(), inputdigi->end());

    LogTrace("SiStripMergeZeroSuppression::produce") << "Looping over the raw data collection " << "\n";
    for ( const auto& rawDigis : *inputraw ) {
      edm::DetSet<SiStripRawDigi>::const_iterator itRawDigis = rawDigis.begin();
      uint16_t nAPV = rawDigis.size()/128;
      uint32_t rawDetId = rawDigis.id;

      std::vector<bool> restoredAPV(std::size_t(nAPV), false);

      bool isModuleRestored = false;
      for ( uint16_t strip = 0; strip < rawDigis.size(); ++strip ) {
        if(itRawDigis[strip].adc()!=0){
          restoredAPV[strip/128] = true;
          isModuleRestored = true;
        }
      }

      if ( isModuleRestored ) {
        LogTrace("SiStripMergeZeroSuppression::produce") << "Apply the ZS to the raw data collection " << "\n";
        edm::DetSet<SiStripDigi> suppressedDigis(rawDetId);
        m_algorithms->suppressVirginRawData(rawDigis, suppressedDigis);

        if ( ! suppressedDigis.empty() ) {
          LogTrace("SiStripMergeZeroSuppression::produce") << "Looking for the detId with the new ZS in the collection of the zero suppressed data" << "\n"; 
          auto zsModule = std::lower_bound(std::begin(outputdigi), std::end(outputdigi), rawDetId,
              [] ( const edm::DetSet<SiStripDigi>& elem, uint32_t testDetId ) { return elem.id < testDetId; });
          const bool isModuleInZscollection = ( ( std::end(outputdigi) != zsModule ) && ( zsModule->id == rawDetId ) );
          LogTrace("SiStripMergeZeroSuppression::produce") << "After the look " << rawDetId << " ==== " <<  ( isModuleInZscollection ? zsModule->id : 0 ) << "\n";
          LogTrace("SiStripMergeZeroSuppression::produce") << "Exiting looking for the detId with the new ZS in the collection of the zero suppressed data" << "\n";

          //creating the map containing the digis (in rawdigi format) merged
          std::vector<uint16_t> mergedRawDigis(size_t(nAPV*128),0);

          uint32_t count=0; // to be removed...
          edm::DetSet<SiStripDigi> newDigiToInsert(rawDetId);
          if ( ! isModuleInZscollection ) {
            LogTrace("SiStripMergeZeroSuppression::produce") << "WE HAVE A PROBLEM, THE MODULE IS NTOT FOUND" << "\n";
            zsModule = outputdigi.insert(zsModule, newDigiToInsert);
            LogTrace("SiStripMergeZeroSuppression::produce") << "new module id -1 " << (zsModule-1)->id << "\n";
            LogTrace("SiStripMergeZeroSuppression::produce") << "new module id " << zsModule->id << "\n"; 
            LogTrace("SiStripMergeZeroSuppression::produce") << "new module id +1 " << (zsModule+1)->id << "\n";
          } else {
            LogTrace("SiStripMergeZeroSuppression::produce") << "inserting only the digis for not restored APVs" << "\n"; 
            LogTrace("SiStripMergeZeroSuppression::produce") << "size : " << zsModule->size() << "\n";
            for ( const auto itZsMod : *zsModule ) {
              const uint16_t adc = itZsMod.adc();
              const uint16_t strip = itZsMod.strip();
              if ( ! restoredAPV[strip/128] ) {
                mergedRawDigis[strip] = adc;
                ++count;
                LogTrace("SiStripMergeZeroSuppression::produce") << "original count: "<< count << " strip: " << strip << " adc: " << adc << "\n";
              }
            }
          }

          LogTrace("SiStripMergeZeroSuppression::produce") << "size of digis to keep: " << count << "\n";
          LogTrace("SiStripMergeZeroSuppression::produce") << "inserting only the digis for the restored APVs" << "\n";
          LogTrace("SiStripMergeZeroSuppression::produce") << "size : " << suppressedDigis.size() << "\n"; 
          for ( const auto itSuppDigi : suppressedDigis ) {
            const uint16_t adc = itSuppDigi.adc();
            const uint16_t strip = itSuppDigi.strip();
            if ( restoredAPV[strip/128] ) {
              mergedRawDigis[strip] = adc;
              LogTrace("SiStripMergeZeroSuppression::produce") << "new suppressed strip: " << strip << " adc: " << adc << "\n";
            }
          }

          LogTrace("SiStripMergeZeroSuppression::produce") << "suppressing the raw digis" << "\n";
          zsModule->clear();
          for ( uint16_t strip=0; strip < mergedRawDigis.size(); ++strip ) {
            uint16_t adc = mergedRawDigis[strip];
            if (adc) zsModule->push_back(SiStripDigi(strip, adc));
          }
          LogTrace("SiStripMergeZeroSuppression::produce") << "size zsModule after the merging: " << zsModule->size() << "\n"; 
          if ( (count + suppressedDigis.size()) != zsModule->size() )
            LogTrace("SiStripMergeZeroSuppression::produce") << "WE HAVE A PROBLEM!!!! THE NUMBER OF DIGIS IS NOT RIGHT==============" << "\n";
          LogTrace("SiStripMergeZeroSuppression::produce") << "exiting suppressing the raw digis" << "\n";
        }//if new ZS digis size
      } //if module restored
    }//loop over raw data collection

    uint32_t oldid =0;
    for ( const auto& dg : outputdigi ) {
      uint32_t iddg = dg.id;
      if ( iddg < oldid ) {
        LogTrace("SiStripMergeZeroSuppression::produce") << "NOT IN THE RIGHT ORGER" << "\n";
        LogTrace("SiStripMergeZeroSuppression::produce") << "=======================" << "\n";  
      }
      oldid = iddg;
    }

    LogTrace("SiStripMergeZeroSuppression::produce") << "write the output vector" << "\n";  
    event.put(std::make_unique<edm::DetSetVector<SiStripDigi>>(outputdigi), "ZeroSuppressed" );
  }
}
