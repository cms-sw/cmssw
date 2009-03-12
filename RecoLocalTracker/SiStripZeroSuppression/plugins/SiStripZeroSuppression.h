#ifndef SiStripZeroSuppression_h
#define SiStripZeroSuppression_h
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include <memory>

class SiStripDigi;
class SiStripRawDigi;


class SiStripZeroSuppression : public edm::EDProducer
{
  
 public:
  
  explicit SiStripZeroSuppression(const edm::ParameterSet&);  
  virtual void produce(edm::Event& , const edm::EventSetup& );
  
 private:

  void processRaw(const edm::InputTag&, const edm::DetSetVector<SiStripRawDigi>&, std::vector<edm::DetSet<SiStripDigi> >& );
  std::vector<edm::InputTag> inputTags;
  typedef std::vector<edm::InputTag>::const_iterator tag_iterator_t;

  std::auto_ptr<SiStripFedZeroSuppression> suppressor;
  std::auto_ptr<SiStripCommonModeNoiseSubtractor> subtractorCMN;
  std::auto_ptr<SiStripPedestalsSubtractor> subtractorPed;

};
#endif






















