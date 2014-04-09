#include "DPGAnalysis/SiStripTools/interface/Multiplicities.h"

ClusterSummarySingleMultiplicity::ClusterSummarySingleMultiplicity():
  m_collection(),m_subdetenum(0),m_subdetvar(), m_clustsummvar(), m_mult(0) { }

ClusterSummarySingleMultiplicity::ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector&& iC):
  m_collection(iC.consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSummaryCollection"))),
  m_subdetenum(iConfig.getParameter<int>("subDetEnum")),
  m_subdetvar(iConfig.getParameter<std::string>("subDetVariable")),
  m_clustsummvar(),
  m_mult(0)
{ 

  m_clustsummvar.push_back("cHits");
  m_clustsummvar.push_back("cSize");
  m_clustsummvar.push_back("cCharge");
  m_clustsummvar.push_back("pHits");
  m_clustsummvar.push_back("pSize");
  m_clustsummvar.push_back("pCharge");

}

ClusterSummarySingleMultiplicity::ClusterSummarySingleMultiplicity(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
  m_collection(iC.consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSummaryCollection"))),
  m_subdetenum(iConfig.getParameter<int>("subDetEnum")),
  m_subdetvar(iConfig.getParameter<std::string>("subDetVariable")),
  m_clustsummvar(),
  m_mult(0)
{ 

  m_clustsummvar.push_back("cHits");
  m_clustsummvar.push_back("cSize");
  m_clustsummvar.push_back("cCharge");
  m_clustsummvar.push_back("pHits");
  m_clustsummvar.push_back("pSize");
  m_clustsummvar.push_back("pCharge");

}

void ClusterSummarySingleMultiplicity::getEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  m_mult = 0;

  edm::Handle<ClusterSummary> clustsumm;
  iEvent.getByToken(m_collection,clustsumm);

  clustsumm->SetUserContent(m_clustsummvar);

  m_mult = int(clustsumm->GetGenericVariable(m_subdetvar,m_subdetenum));

}

int ClusterSummarySingleMultiplicity::mult() const { return m_mult; }

