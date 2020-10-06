#ifndef L1TStage2CaloLayer1Offline_H
#define L1TStage2CaloLayer1Offline_H

//change from "One" to "Global"
//#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

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

//This now stores our monitor elements,
//and is made accessible as a part of the class template and arguments of many class functions
namespace CaloL1Information {
  struct data {
    dqm::reco::MonitorElement *ecalDiscrepancy_;
    dqm::reco::MonitorElement *ecalLinkError_;
    dqm::reco::MonitorElement *ecalOccupancy_;
    dqm::reco::MonitorElement *hcalDiscrepancy_;
    dqm::reco::MonitorElement *hcalLinkError_;
    dqm::reco::MonitorElement *hcalOccupancy_;

    dqm::reco::MonitorElement *ecalOccEtDiscrepancy_;
    dqm::reco::MonitorElement *ecalOccFgDiscrepancy_;
    dqm::reco::MonitorElement *ecalOccLinkMasked_;
    dqm::reco::MonitorElement *ecalOccRecdEtWgt_;
    dqm::reco::MonitorElement *ecalOccRecdFgVB_;
    dqm::reco::MonitorElement *ecalOccSentAndRecd_;
    dqm::reco::MonitorElement *ecalOccSentFgVB_;
    dqm::reco::MonitorElement *ecalOccSent_;
    dqm::reco::MonitorElement *ecalOccTowerMasked_;
    dqm::reco::MonitorElement *ecalTPRawEtCorrelation_;
    dqm::reco::MonitorElement *ecalTPRawEtDiffNoMatch_;
    dqm::reco::MonitorElement *ecalTPRawEtRecd_;
    dqm::reco::MonitorElement *ecalTPRawEtSentAndRecd_;
    dqm::reco::MonitorElement *ecalTPRawEtSent_;

    dqm::reco::MonitorElement *ecalOccSentNotRecd_;
    dqm::reco::MonitorElement *ecalOccRecdNotSent_;
    dqm::reco::MonitorElement *ecalOccNoMatch_;

    dqm::reco::MonitorElement *hcalOccEtDiscrepancy_;
    dqm::reco::MonitorElement *hcalOccFbDiscrepancy_;
    dqm::reco::MonitorElement *hcalOccFb2Discrepancy_;
    dqm::reco::MonitorElement *hcalOccLinkMasked_;
    dqm::reco::MonitorElement *hcalOccRecdEtWgt_;
    dqm::reco::MonitorElement *hcalOccRecdFb_;
    dqm::reco::MonitorElement *hcalOccRecdFb2_;
    dqm::reco::MonitorElement *hcalOccSentAndRecd_;
    dqm::reco::MonitorElement *hcalOccSentFb_;
    dqm::reco::MonitorElement *hcalOccSentFb2_;
    dqm::reco::MonitorElement *hcalOccSent_;
    dqm::reco::MonitorElement *hcalOccTowerMasked_;
    dqm::reco::MonitorElement *hcalTPRawEtCorrelationHBHE_;
    dqm::reco::MonitorElement *hcalTPRawEtCorrelationHF_;
    dqm::reco::MonitorElement *hcalTPRawEtDiffNoMatch_;
    dqm::reco::MonitorElement *hcalTPRawEtRecd_;
    dqm::reco::MonitorElement *hcalTPRawEtSentAndRecd_;
    dqm::reco::MonitorElement *hcalTPRawEtSent_;

    dqm::reco::MonitorElement *hcalOccSentNotRecd_;
    dqm::reco::MonitorElement *hcalOccRecdNotSent_;
    dqm::reco::MonitorElement *hcalOccNoMatch_;

    dqm::reco::MonitorElement *ECALmismatchesPerBx_;
    dqm::reco::MonitorElement *HBHEmismatchesPerBx_;
    dqm::reco::MonitorElement *HFmismatchesPerBx_;

    dqm::reco::MonitorElement *bxidErrors_;
    dqm::reco::MonitorElement *l1idErrors_;
    dqm::reco::MonitorElement *orbitErrors_;

    dqm::reco::MonitorElement *ecalLinkErrorByLumi_;
    dqm::reco::MonitorElement *ecalMismatchByLumi_;
    dqm::reco::MonitorElement *hcalLinkErrorByLumi_;
    dqm::reco::MonitorElement *hcalMismatchByLumi_;
  };
}  // namespace CaloL1Information

class L1TStage2CaloLayer1Offline : public DQMGlobalEDAnalyzer<CaloL1Information::data> {
public:
  L1TStage2CaloLayer1Offline(const edm::ParameterSet &ps);
  ~L1TStage2CaloLayer1Offline() override;

protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &, CaloL1Information::data &) const override{};
  void bookHistograms(DQMStore::IBooker &ibooker,
                      const edm::Run &,
                      const edm::EventSetup &,
                      CaloL1Information::data &data) const override;
  void dqmAnalyze(edm::Event const &, edm::EventSetup const &, CaloL1Information::data const &) const override;
  void dqmEndRun(edm::Run const &, edm::EventSetup const &, CaloL1Information::data const &) const override{};

private:
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

  //Elements removed from online version of module.
  //They were processed at luminosity block transitions incompatible with DQMGlobalEDAnalyzer
  /* MonitorElement *last20Mismatches_; */
  /* std::array<std::pair<std::string, int>, 20> last20MismatchArray_; */
  /* size_t lastMismatchIndex_{0}; */

  /* MonitorElement *maxEvtLinkErrorsByLumiECAL_; */
  /* MonitorElement *maxEvtLinkErrorsByLumiHCAL_; */
  /* MonitorElement *maxEvtLinkErrorsByLumi_; */
  /* int maxEvtLinkErrorsECALCurrentLumi_{0}; */
  /* int maxEvtLinkErrorsHCALCurrentLumi_{0}; */

  /* MonitorElement *maxEvtMismatchByLumiECAL_; */
  /* MonitorElement *maxEvtMismatchByLumiHCAL_; */
  /* MonitorElement *maxEvtMismatchByLumi_; */
  /* int maxEvtMismatchECALCurrentLumi_{0}; */
  /* int maxEvtMismatchHCALCurrentLumi_{0}; */
};
#endif
