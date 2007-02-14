#include "RecoBTag/SoftLepton/interface/ElectronTaggerESProducer.h"
#include "RecoBTag/SoftLepton/interface/ElectronTagger.h"

ElectronTaggerESProducer::ElectronTaggerESProducer(const edm::ParameterSet & pset) : 
  m_pset(pset)
{
  std::string m_name = m_pset.getParameter<std::string>("ComponentName");
  setWhatProduced(this, m_name);
}

ElectronTaggerESProducer::~ElectronTaggerESProducer() {
}

boost::shared_ptr<LeptonTaggerBase> ElectronTaggerESProducer::produce(const SoftLeptonBTagRecord & record) {
  m_softLeptonTagger = boost::shared_ptr<LeptonTaggerBase>(new ElectronTagger());
  return m_softLeptonTagger;
}
