#ifndef L1TStage2CaloLayer1_H
#define L1TStage2CaloLayer1_H

// system include files
#include <memory>
#include <string>
#include <array>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

namespace ComparisonHelper {
  // Designed for edm::SortedCollection<T>
  // Iterators should have a value type that derives from DetID, or
  // otherwise implements value_type::id().
  // Compare should be strict weak ordering (e.g. edm::SortedCollection<T>::key_compare)
  // Mostly http://www.cplusplus.com/reference/algorithm/set_union/
  // But with both collections saved in std::pair
  // Is this necessary? Absolutely not... but it was fun to make!
  template <class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
  OutputIterator zip(InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator2 last2,
                     OutputIterator result,
                     Compare comp) {
    typedef typename InputIterator1::value_type dummy1;
    typedef typename InputIterator2::value_type dummy2;
    while (first1 != last1 || first2 != last2) {
      if (first1 == last1) {
        while (first2 != last2) {
          *result = std::make_pair(dummy1{first2->id()}, *first2);
          ++first2;
          ++result;
        }
        return result;
      }
      if (first2 == last2) {
        while (first1 != last1) {
          *result = std::make_pair(*first1, dummy2{first1->id()});
          ++first1;
          ++result;
        }
        return result;
      }
      if (comp(*first1, *first2)) {
        *result = std::make_pair(*first1, dummy2{first1->id()});
        ++first1;
      } else if (comp(*first2, *first1)) {
        *result = std::make_pair(dummy1{first2->id()}, *first2);
        ++first2;
      } else {
        *result = std::make_pair(*first1, *first2);
        ++first1;
        ++first2;
      }
      ++result;
    }
    return result;
  }
}  // namespace ComparisonHelper

class L1TStage2CaloLayer1 : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  L1TStage2CaloLayer1(const edm::ParameterSet &ps);
  ~L1TStage2CaloLayer1() override;

protected:
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &, const edm::EventSetup &) override;
  void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override;
  void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override;

private:
  void updateMismatch(const edm::Event &e, int mismatchType);
  // Input and config info
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPSourceRecd_;
  std::string ecalTPSourceRecdLabel_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPSourceRecd_;
  std::string hcalTPSourceRecdLabel_;
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPSourceSent_;
  std::string ecalTPSourceSentLabel_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPSourceSent_;
  std::string hcalTPSourceSentLabel_;
  edm::EDGetTokenT<FEDRawDataCollection> fedRawData_;
  std::string histFolder_;
  int tpFillThreshold_;
  bool ignoreHFfbs_;

  MonitorElement *ecalDiscrepancy_;
  MonitorElement *ecalLinkError_;
  MonitorElement *ecalOccupancy_;
  MonitorElement *hcalDiscrepancy_;
  MonitorElement *hcalLinkError_;
  MonitorElement *hcalOccupancy_;

  MonitorElement *ecalOccEtDiscrepancy_;
  MonitorElement *ecalOccFgDiscrepancy_;
  MonitorElement *ecalOccLinkMasked_;
  MonitorElement *ecalOccRecdEtWgt_;
  MonitorElement *ecalOccRecdFgVB_;
  MonitorElement *ecalOccSentAndRecd_;
  MonitorElement *ecalOccSentFgVB_;
  MonitorElement *ecalOccSent_;
  MonitorElement *ecalOccTowerMasked_;
  MonitorElement *ecalTPRawEtCorrelation_;
  MonitorElement *ecalTPRawEtDiffNoMatch_;
  MonitorElement *ecalTPRawEtRecd_;
  MonitorElement *ecalTPRawEtSentAndRecd_;
  MonitorElement *ecalTPRawEtSent_;

  MonitorElement *ecalOccSentNotRecd_;
  MonitorElement *ecalOccRecdNotSent_;
  MonitorElement *ecalOccNoMatch_;

  MonitorElement *hcalOccEtDiscrepancy_;
  MonitorElement *hcalOccFbDiscrepancy_;
  MonitorElement *hcalOccFb2Discrepancy_;
  MonitorElement *hcalOccLinkMasked_;
  MonitorElement *hcalOccRecdEtWgt_;
  MonitorElement *hcalOccRecdFb_;
  MonitorElement *hcalOccRecdFb2_;
  MonitorElement *hcalOccSentAndRecd_;
  MonitorElement *hcalOccSentFb_;
  MonitorElement *hcalOccSentFb2_;
  MonitorElement *hcalOccSent_;
  MonitorElement *hcalOccTowerMasked_;
  MonitorElement *hcalTPRawEtCorrelationHBHE_;
  MonitorElement *hcalTPRawEtCorrelationHF_;
  MonitorElement *hcalTPRawEtDiffNoMatch_;
  MonitorElement *hcalTPRawEtRecd_;
  MonitorElement *hcalTPRawEtSentAndRecd_;
  MonitorElement *hcalTPRawEtSent_;

  MonitorElement *hcalOccSentNotRecd_;
  MonitorElement *hcalOccRecdNotSent_;
  MonitorElement *hcalOccNoMatch_;

  MonitorElement *last20Mismatches_;
  std::array<std::pair<std::string, int>, 20> last20MismatchArray_;
  size_t lastMismatchIndex_{0};

  MonitorElement *ecalLinkErrorByLumi_;
  MonitorElement *ecalMismatchByLumi_;
  MonitorElement *hcalLinkErrorByLumi_;
  MonitorElement *hcalMismatchByLumi_;

  MonitorElement *maxEvtLinkErrorsByLumiECAL_;
  MonitorElement *maxEvtLinkErrorsByLumiHCAL_;
  MonitorElement *maxEvtLinkErrorsByLumi_;
  int maxEvtLinkErrorsECALCurrentLumi_{0};
  int maxEvtLinkErrorsHCALCurrentLumi_{0};

  MonitorElement *maxEvtMismatchByLumiECAL_;
  MonitorElement *maxEvtMismatchByLumiHCAL_;
  MonitorElement *maxEvtMismatchByLumi_;
  int maxEvtMismatchECALCurrentLumi_{0};
  int maxEvtMismatchHCALCurrentLumi_{0};

  MonitorElement *ECALmismatchesPerBx_;
  MonitorElement *HBHEmismatchesPerBx_;
  MonitorElement *HFmismatchesPerBx_;

  MonitorElement *bxidErrors_;
  MonitorElement *l1idErrors_;
  MonitorElement *orbitErrors_;

  // Prevent reallocation per event
  std::vector<std::pair<EcalTriggerPrimitiveDigi, EcalTriggerPrimitiveDigi> > ecalTPSentRecd_;
  std::vector<std::pair<HcalTriggerPrimitiveDigi, HcalTriggerPrimitiveDigi> > hcalTPSentRecd_;
};

#endif
