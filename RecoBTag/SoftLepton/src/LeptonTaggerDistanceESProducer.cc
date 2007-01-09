#include "RecoBTag/SoftLepton/interface/LeptonTaggerDistanceESProducer.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerDistance.h"

LeptonTaggerDistanceESProducer::LeptonTaggerDistanceESProducer(const edm::ParameterSet & pset) : 
  m_pset(pset)
{
  std::string m_name = m_pset.getParameter<std::string>("ComponentName");
  setWhatProduced(this, m_name);
}

LeptonTaggerDistanceESProducer::~LeptonTaggerDistanceESProducer() {
}

boost::shared_ptr<LeptonTaggerBase> LeptonTaggerDistanceESProducer::produce(const SoftLeptonBTagRecord & record) {
  m_softLeptonTagger = boost::shared_ptr<LeptonTaggerBase>(new LeptonTaggerDistance());
  return m_softLeptonTagger;
}
