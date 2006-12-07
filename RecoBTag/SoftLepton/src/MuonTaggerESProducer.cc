#include "RecoBTag/SoftLepton/interface/MuonTaggerESProducer.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"

MuonTaggerESProducer::MuonTaggerESProducer(const edm::ParameterSet & pset) : 
  m_pset(pset)
{
  std::string m_name = m_pset.getParameter<std::string>("ComponentName");
  setWhatProduced(this, m_name);
}

MuonTaggerESProducer::~MuonTaggerESProducer() {
}

boost::shared_ptr<LeptonTaggerBase> MuonTaggerESProducer::produce(const SoftLeptonBTagRecord & record) {
  m_softLeptonTagger = boost::shared_ptr<LeptonTaggerBase>(new MuonTagger());
  return m_softLeptonTagger;
}
