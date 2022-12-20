#ifndef DPGAnalysis_SiStripTools_Multiplicities_H
#define DPGAnalysis_SiStripTools_Multiplicities_H

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

namespace edm {
  class EventSetup;
};

#include <string>

class ClusterSummarySingleMultiplicity {
public:
  ClusterSummarySingleMultiplicity();
  ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
  ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);

  void getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  int mult() const;

private:
  ClusterSummary::CMSTracker m_subdetenum;
  ClusterSummary::VariablePlacement m_varenum;
  int m_mult;
  edm::EDGetTokenT<ClusterSummary> m_collection;
};

template <class T>
class SingleMultiplicity {
public:
  SingleMultiplicity();
  SingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
  SingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);

  void getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  int mult() const;
  //  int mult;

private:
  int m_modthr;
  bool m_useQuality;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> m_qualityToken;
  int m_mult;
  edm::EDGetTokenT<T> m_collection;
};

template <class T>
SingleMultiplicity<T>::SingleMultiplicity() : m_modthr(-1), m_useQuality(false), m_mult(0), m_collection() {}

template <class T>
SingleMultiplicity<T>::SingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
    : m_modthr(iConfig.getUntrackedParameter<int>("moduleThreshold")),
      m_useQuality(iConfig.getUntrackedParameter<bool>("useQuality", false)),
      m_qualityToken(m_useQuality
                         ? decltype(m_qualityToken){iC.esConsumes<SiStripQuality, SiStripQualityRcd>(
                               edm::ESInputTag{"", iConfig.getUntrackedParameter<std::string>("qualityLabel", "")})}
                         : decltype(m_qualityToken){}),
      m_mult(0),
      m_collection(iC.consumes<T>(iConfig.getParameter<edm::InputTag>("collectionName"))) {}
template <class T>
SingleMultiplicity<T>::SingleMultiplicity(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC)
    : m_modthr(iConfig.getUntrackedParameter<int>("moduleThreshold")),
      m_useQuality(iConfig.getUntrackedParameter<bool>("useQuality", false)),
      m_qualityToken(m_useQuality
                         ? decltype(m_qualityToken){iC.esConsumes<SiStripQuality, SiStripQualityRcd>(
                               edm::ESInputTag(iConfig.getUntrackedParameter<std::string>("qualityLabel", "")))}
                         : decltype(m_qualityToken){}),
      m_mult(0),
      m_collection(iC.consumes<T>(iConfig.getParameter<edm::InputTag>("collectionName"))) {}

template <class T>
void SingleMultiplicity<T>::getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  m_mult = 0;

  const SiStripQuality* quality = nullptr;
  if (m_useQuality) {
    quality = &iSetup.getData(m_qualityToken);
  }

  edm::Handle<T> digis;
  iEvent.getByToken(m_collection, digis);

  for (typename T::const_iterator it = digis->begin(); it != digis->end(); it++) {
    if (!m_useQuality || !quality->IsModuleBad(it->detId())) {
      if (m_modthr < 0 || int(it->size()) < m_modthr) {
        m_mult += it->size();
        //      mult += it->size();
      }
    }
  }
}

template <class T>
int SingleMultiplicity<T>::mult() const {
  return m_mult;
}

template <class T1, class T2>
class MultiplicityPair {
public:
  MultiplicityPair();
  MultiplicityPair(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
  MultiplicityPair(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);

  void getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  int mult1() const;
  int mult2() const;

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
void MultiplicityPair<T1, T2>::getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  m_multiplicity1.getEvent(iEvent, iSetup);
  m_multiplicity2.getEvent(iEvent, iSetup);
}

template <class T1, class T2>
int MultiplicityPair<T1, T2>::mult1() const {
  return m_multiplicity1.mult();
}

template <class T1, class T2>
int MultiplicityPair<T1, T2>::mult2() const {
  return m_multiplicity2.mult();
}

typedef SingleMultiplicity<edm::DetSetVector<SiStripDigi> > SingleSiStripDigiMultiplicity;
typedef SingleMultiplicity<edmNew::DetSetVector<SiStripCluster> > SingleSiStripClusterMultiplicity;
typedef SingleMultiplicity<edmNew::DetSetVector<SiPixelCluster> > SingleSiPixelClusterMultiplicity;
typedef MultiplicityPair<SingleMultiplicity<edmNew::DetSetVector<SiPixelCluster> >,
                         SingleMultiplicity<edmNew::DetSetVector<SiStripCluster> > >
    SiPixelClusterSiStripClusterMultiplicityPair;
typedef MultiplicityPair<ClusterSummarySingleMultiplicity, ClusterSummarySingleMultiplicity>
    ClusterSummaryMultiplicityPair;

#endif  // DPGAnalysis_SiStripTools_Multiplicities_H
