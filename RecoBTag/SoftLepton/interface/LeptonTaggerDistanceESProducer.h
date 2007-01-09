#ifndef RecoBTag_SoftLepton_LeptonTaggerDistanceESProducer_h
#define RecoBTag_SoftLepton_LeptonTaggerDistanceESProducer_h

#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTag/Records/interface/SoftLeptonBTagRecord.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerBase.h"

class LeptonTaggerDistanceESProducer: public edm::ESProducer {
public:
  LeptonTaggerDistanceESProducer(const edm::ParameterSet & pset);
  virtual ~LeptonTaggerDistanceESProducer();

  boost::shared_ptr<LeptonTaggerBase> produce(const SoftLeptonBTagRecord & record);

private:
  boost::shared_ptr<LeptonTaggerBase> m_softLeptonTagger;
  edm::ParameterSet m_pset;
};

#endif // RecoBTag_SoftLepton_LeptonTaggerDistanceESProducer_h
