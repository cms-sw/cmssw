#include "RecoBTag/SoftLepton/interface/MuonTaggerNoIPESProducer.h"
#include "RecoBTag/SoftLepton/interface/MuonTaggerNoIP.h"

MuonTaggerNoIPESProducer::MuonTaggerNoIPESProducer(const edm::ParameterSet & pset) :
  m_pset(pset)
{
  std::string m_name = m_pset.getParameter<std::string>("ComponentName");
  setWhatProduced(this, m_name);
}

MuonTaggerNoIPESProducer::~MuonTaggerNoIPESProducer() {
}

boost::shared_ptr<LeptonTaggerBase> MuonTaggerNoIPESProducer::produce(const SoftLeptonBTagRecord & record) {
  m_softLeptonTagger = boost::shared_ptr<LeptonTaggerBase>(new MuonTaggerNoIP());
  return m_softLeptonTagger;
}
