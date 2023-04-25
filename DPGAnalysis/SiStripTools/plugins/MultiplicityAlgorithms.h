#ifndef DPGAnalysis_SiStripTools_MultiplicityAlgorithms_H
#define DPGAnalysis_SiStripTools_MultiplicityAlgorithms_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "DPGAnalysis/SiStripTools/interface/Multiplicities.h"

namespace edm {
  class EventSetup;
};

#include <string>

namespace sistriptools::algorithm {
  class ClusterSummarySingleMultiplicity {
  public:
    using value_t = values::Multiplicity;
    ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
    ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);

    value_t getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) const;

  private:
    ClusterSummary::CMSTracker m_subdetenum;
    ClusterSummary::VariablePlacement m_varenum;
    edm::EDGetTokenT<ClusterSummary> m_collection;
  };

  template <class T>
  class SingleMultiplicity {
  public:
    using value_t = values::Multiplicity;
    SingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
    SingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);

    value_t getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) const;

  private:
    int m_modthr;
    bool m_useQuality;
    edm::ESGetToken<SiStripQuality, SiStripQualityRcd> m_qualityToken;
    edm::EDGetTokenT<T> m_collection;
  };

  template <class T>
  SingleMultiplicity<T>::SingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
      : m_modthr(iConfig.getUntrackedParameter<int>("moduleThreshold")),
        m_useQuality(iConfig.getUntrackedParameter<bool>("useQuality", false)),
        m_qualityToken(m_useQuality
                           ? decltype(m_qualityToken){iC.esConsumes<SiStripQuality, SiStripQualityRcd>(
                                 edm::ESInputTag{"", iConfig.getUntrackedParameter<std::string>("qualityLabel", "")})}
                           : decltype(m_qualityToken){}),
        m_collection(iC.consumes<T>(iConfig.getParameter<edm::InputTag>("collectionName"))) {}
  template <class T>
  SingleMultiplicity<T>::SingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC)
      : m_modthr(iConfig.getUntrackedParameter<int>("moduleThreshold")),
        m_useQuality(iConfig.getUntrackedParameter<bool>("useQuality", false)),
        m_qualityToken(m_useQuality
                           ? decltype(m_qualityToken){iC.esConsumes<SiStripQuality, SiStripQualityRcd>(
                                 edm::ESInputTag(iConfig.getUntrackedParameter<std::string>("qualityLabel", "")))}
                           : decltype(m_qualityToken){}),
        m_collection(iC.consumes<T>(iConfig.getParameter<edm::InputTag>("collectionName"))) {}

  template <class T>
  typename SingleMultiplicity<T>::value_t SingleMultiplicity<T>::getEvent(const edm::Event& iEvent,
                                                                          const edm::EventSetup& iSetup) const {
    int mult = 0;

    const SiStripQuality* quality = nullptr;
    if (m_useQuality) {
      quality = &iSetup.getData(m_qualityToken);
    }

    edm::Handle<T> digis;
    iEvent.getByToken(m_collection, digis);

    for (typename T::const_iterator it = digis->begin(); it != digis->end(); it++) {
      if (!m_useQuality || !quality->IsModuleBad(it->detId())) {
        if (m_modthr < 0 || int(it->size()) < m_modthr) {
          mult += it->size();
          //      mult += it->size();
        }
      }
    }
    return value_t(mult);
  }

  template <class T1, class T2>
  class MultiplicityPair {
  public:
    using value_t = values::MultiplicityPair<typename T1::value_t, typename T2::value_t>;
    MultiplicityPair();
    MultiplicityPair(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
    MultiplicityPair(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);

    value_t getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) const;

  private:
    T1 m_multiplicity1;
    T2 m_multiplicity2;
  };

  template <class T1, class T2>
  MultiplicityPair<T1, T2>::MultiplicityPair() : m_multiplicity1(), m_multiplicity2() {}

  template <class T1, class T2>
  MultiplicityPair<T1, T2>::MultiplicityPair(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
      : m_multiplicity1(iConfig.getParameter<edm::ParameterSet>("firstMultiplicityConfig"), iC),
        m_multiplicity2(iConfig.getParameter<edm::ParameterSet>("secondMultiplicityConfig"), iC) {}
  template <class T1, class T2>
  MultiplicityPair<T1, T2>::MultiplicityPair(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC)
      : m_multiplicity1(iConfig.getParameter<edm::ParameterSet>("firstMultiplicityConfig"), iC),
        m_multiplicity2(iConfig.getParameter<edm::ParameterSet>("secondMultiplicityConfig"), iC) {}

  template <class T1, class T2>
  typename MultiplicityPair<T1, T2>::value_t MultiplicityPair<T1, T2>::getEvent(const edm::Event& iEvent,
                                                                                const edm::EventSetup& iSetup) const {
    auto m1 = m_multiplicity1.getEvent(iEvent, iSetup);
    auto m2 = m_multiplicity2.getEvent(iEvent, iSetup);
    return value_t(m1, m2);
  }

  typedef SingleMultiplicity<edm::DetSetVector<SiStripDigi> > SingleSiStripDigiMultiplicity;
  typedef SingleMultiplicity<edmNew::DetSetVector<SiStripCluster> > SingleSiStripClusterMultiplicity;
  typedef SingleMultiplicity<edmNew::DetSetVector<SiPixelCluster> > SingleSiPixelClusterMultiplicity;
  typedef MultiplicityPair<SingleMultiplicity<edmNew::DetSetVector<SiPixelCluster> >,
                           SingleMultiplicity<edmNew::DetSetVector<SiStripCluster> > >
      SiPixelClusterSiStripClusterMultiplicityPair;
  typedef MultiplicityPair<ClusterSummarySingleMultiplicity, ClusterSummarySingleMultiplicity>
      ClusterSummaryMultiplicityPair;
}  // namespace sistriptools::algorithm

#endif  // DPGAnalysis_SiStripTools_Multiplicities_H
