#ifndef RecoBTag_SoftLepton_LeptonTaggerESProducer_h
#define RecoBTag_SoftLepton_LeptonTaggerESProducer_h

#include <string>
#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTag/Records/interface/SoftLeptonBTagRecord.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerBase.h"

template <typename Tagger>
class LeptonTaggerESProducer: public edm::ESProducer {
public:
  LeptonTaggerESProducer(const edm::ParameterSet & pset) : m_pset(pset) {
    setWhatProduced(this, m_pset.retrieve("ComponentName").getString());

    m_softLeptonTagger = boost::shared_ptr<LeptonTaggerBase>(new Tagger());
  }
  
  virtual ~LeptonTaggerESProducer() {
  }

  boost::shared_ptr<LeptonTaggerBase> produce(const SoftLeptonBTagRecord & record) {
    return m_softLeptonTagger;
  }

private:
  boost::shared_ptr<LeptonTaggerBase> m_softLeptonTagger;
  edm::ParameterSet m_pset;
};

#endif // RecoBTag_SoftLepton_LeptonTaggerESProducer_h
