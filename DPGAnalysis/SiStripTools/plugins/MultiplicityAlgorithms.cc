#include "MultiplicityAlgorithms.h"

using namespace sistriptools::algorithm;

ClusterSummarySingleMultiplicity::ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig,
                                                                   edm::ConsumesCollector&& iC)
    : m_subdetenum((ClusterSummary::CMSTracker)iConfig.getParameter<int>("subDetEnum")),
      m_varenum((ClusterSummary::VariablePlacement)iConfig.getParameter<int>("varEnum")),
      m_collection(iC.consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSummaryCollection"))) {}

ClusterSummarySingleMultiplicity::ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig,
                                                                   edm::ConsumesCollector& iC)
    : m_subdetenum((ClusterSummary::CMSTracker)iConfig.getParameter<int>("subDetEnum")),
      m_varenum((ClusterSummary::VariablePlacement)iConfig.getParameter<int>("varEnum")),
      m_collection(iC.consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSummaryCollection"))) {}

ClusterSummarySingleMultiplicity::value_t ClusterSummarySingleMultiplicity::getEvent(
    const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  int mult = 0;

  edm::Handle<ClusterSummary> clustsumm;
  iEvent.getByToken(m_collection, clustsumm);

  switch (m_varenum) {
    case ClusterSummary::NCLUSTERS:
      mult = int(clustsumm->getNClus(m_subdetenum));
      break;
    case ClusterSummary::CLUSTERSIZE:
      mult = int(clustsumm->getClusSize(m_subdetenum));
      break;
    case ClusterSummary::CLUSTERCHARGE:
      mult = int(clustsumm->getClusCharge(m_subdetenum));
      break;
    default:
      mult = -1;
  }
  return value_t(mult);
}
