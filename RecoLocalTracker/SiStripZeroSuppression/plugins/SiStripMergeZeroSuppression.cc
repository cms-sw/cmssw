#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
  std::cout<< "starting Merging" << std::endl;
  edm::Handle<edm::DetSetVector<SiStripDigi>> inputdigi;
  edm::Handle<edm::DetSetVector<SiStripRawDigi>> inputraw;
  event.getByToken(m_rawDigisToMerge, inputdigi);
  event.getByToken(m_zsDigisToMerge, inputraw);

  std::cout << inputdigi->size() << " " << inputraw->size() << std::endl;
  if ( ! inputraw->empty() ) {
    std::vector<edm::DetSet<SiStripDigi>> outputdigi(inputdigi->begin(), inputdigi->end());

    std::cout << "looping over the raw data collection" << std::endl;
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
        std::cout << "apply the ZS to the raw data collection" << std::endl;
        edm::DetSet<SiStripDigi> suppressedDigis(rawDetId);
        m_algorithms->SuppressVirginRawData(rawDigis, suppressedDigis);

        if ( ! suppressedDigis.empty() ) {
          std::cout << "looking for the detId with the new ZS in the collection of the zero suppressed data" << std::endl;
          bool isModuleInZscollection = false;
          uint32_t zsDetId{0};
          auto zsModule = std::lower_bound(std::begin(outputdigi), std::end(outputdigi), rawDetId,
              [] ( const edm::DetSet<SiStripDigi>& elem, uint32_t testDetId ) { return elem.id < testDetId; });
          if ( ( std::end(outputdigi) != zsModule ) && ( zsModule->id == rawDetId ) ) {
            isModuleInZscollection = true;
            zsDetId = zsModule->id;
          }
          std::cout << "after the look " << rawDetId << " ==== " <<  zsDetId << std::endl;
          std::cout << "exiting looking for the detId with the new ZS in the collection of the zero suppressed data" << std::endl;

          //creating the map containing the digis (in rawdigi format) merged
          std::vector<uint16_t> MergedRawDigis(size_t(nAPV*128),0);

          uint32_t count=0; // to be removed...
          edm::DetSet<SiStripDigi> newDigiToIndert(rawDetId);
          if ( ! isModuleInZscollection ) {
            std::cout << "WE HAVE A PROBLEM, THE MODULE IS NTOT FOUND" << std::endl;
            zsModule = outputdigi.insert(zsModule, newDigiToIndert);
            std::cout << "new module id -1 " << (zsModule-1)->id << std::endl;
            std::cout << "new module id " << zsModule->id << std::endl;
            std::cout << "new module id +1 " << (zsModule+1)->id << std::endl;
          } else {
            std::cout << "inserting only the digis for not restored APVs" << std::endl;
            std::cout << "size : " << zsModule->size() << std::endl;
            for ( const auto itZsMod : *zsModule ) {
              const uint16_t adc = itZsMod.adc();
              const uint16_t strip = itZsMod.strip();
              if ( ! restoredAPV[strip/128] ) {
                MergedRawDigis[strip] = adc;
                ++count;
                std::cout << "original count: "<< count << " strip: " << strip << " adc: " << adc << std::endl;
              }
            }
          }

          std::cout << "size of digis to keep: " << count << std::endl;
          std::cout << "inserting only the digis for the restored APVs" << std::endl;
          std::cout << "size : " << suppressedDigis.size() << std::endl;
          for ( const auto itSuppDigi : suppressedDigis ) {
            const uint16_t adc = itSuppDigi.adc();
            const uint16_t strip = itSuppDigi.strip();
            if ( restoredAPV[strip/128] ) {
              MergedRawDigis[strip] = adc;
              std::cout << "new suppressed strip: " << strip << " adc: " << adc << std::endl;
            }
          }

          std::cout << "suppressing the raw digis" << std::endl;
          zsModule->clear();
          for ( uint16_t strip=0; strip < MergedRawDigis.size(); ++strip ) {
            uint16_t adc = MergedRawDigis[strip];
            if (adc) zsModule->push_back(SiStripDigi(strip, adc));
          }
          std::cout << "size zsModule after the merging: " << zsModule->size() << std::endl;
          if ( (count + suppressedDigis.size()) != zsModule->size() )
            std::cout << "WE HAVE A PROBLEM!!!! THE NUMBER OF DIGIS IS NOT RIGHT==============" << std::endl;
          std::cout << "exiting suppressing the raw digis" << std::endl;
        }//if new ZS digis size
      } //if module restored
    }//loop over raw data collection

    uint32_t oldid =0;
    for ( const auto& dg : outputdigi ) {
      uint32_t iddg = dg.id;
      if ( iddg < oldid ) {
        std::cout << "NOT IN THE RIGHT ORGER"  << std:: endl;
        std::cout << "=======================" << std:: endl;
      }
      oldid = iddg;
    }

    std::cout << "write the output vector" << std::endl;
    event.put(std::make_unique<edm::DetSetVector<SiStripDigi>>(outputdigi), "ZeroSuppressed" );
  }
}
