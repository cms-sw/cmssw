#include "DPGAnalysis/SiStripTools/interface/Multiplicities.h"

ClusterSummarySingleMultiplicity::ClusterSummarySingleMultiplicity():
  m_subdetenum(ClusterSummary::STRIP),m_varenum(ClusterSummary::NCLUSTERS),m_mult(0),m_collection() { }

ClusterSummarySingleMultiplicity::ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector&& iC):
  m_subdetenum((ClusterSummary::CMSTracker)iConfig.getParameter<int>("subDetEnum")),m_varenum((ClusterSummary::VariablePlacement)iConfig.getParameter<int>("varEnum")),
  m_mult(0),
  m_collection(iC.consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSummaryCollection")))
{}

ClusterSummarySingleMultiplicity::ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
  m_subdetenum((ClusterSummary::CMSTracker)iConfig.getParameter<int>("subDetEnum")),m_varenum((ClusterSummary::VariablePlacement)iConfig.getParameter<int>("varEnum")),
  m_mult(0),
  m_collection(iC.consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSummaryCollection")))
{}

void ClusterSummarySingleMultiplicity::getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  m_mult = 0;

  edm::Handle<ClusterSummary> clustsumm;
  iEvent.getByToken(m_collection,clustsumm);

  switch(m_varenum){
    case ClusterSummary::NCLUSTERS     : m_mult = int(clustsumm->getNClus     (m_subdetenum)); break;
    case ClusterSummary::CLUSTERSIZE   : m_mult = int(clustsumm->getClusSize  (m_subdetenum)); break;
    case ClusterSummary::CLUSTERCHARGE : m_mult = int(clustsumm->getClusCharge(m_subdetenum)); break;
    default : m_mult = -1;
  }
}

int ClusterSummarySingleMultiplicity::mult() const { return m_mult; }

