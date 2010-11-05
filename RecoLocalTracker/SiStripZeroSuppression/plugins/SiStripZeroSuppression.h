#ifndef SiStripZeroSuppression_h
#define SiStripZeroSuppression_h
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"

class SiStripDigi;
class SiStripRawDigi;

class SiStripZeroSuppression : public edm::EDProducer
{
  
 public:
  
  explicit SiStripZeroSuppression(const edm::ParameterSet&);  
  virtual void produce(edm::Event& , const edm::EventSetup& );
  
 private:

  void processRaw(const edm::InputTag&, const edm::DetSetVector<SiStripRawDigi>&, std::vector<edm::DetSet<SiStripDigi> >&, std::vector<edm::DetSet<SiStripRawDigi> >& );
  void storeCMN(uint32_t, const std::vector< std::pair<short,float> >&);
  void storeBaseline(uint32_t, const std::vector< std::pair<short,float> >&, std::map< uint16_t, std::vector < int16_t> >&);
  void storeBaselinePoints(uint32_t, std::vector< std::map< uint16_t, int16_t> >&);
  void StandardZeroSuppression(edm::Event&);
  void CollectionMergedZeroSuppression(edm::Event&);
  
  std::vector<edm::InputTag> inputTags;
  typedef std::vector<edm::InputTag>::const_iterator tag_iterator_t;
  std::vector< edm::DetSet<SiStripProcessedRawDigi> > output_apvcm; 
  std::vector< edm::DetSet<SiStripProcessedRawDigi> > output_baseline;
  std::vector< edm::DetSet<SiStripDigi> > output_baseline_points;
  std::auto_ptr<SiStripRawProcessingAlgorithms> algorithms;
  
  
  bool storeCM;
  bool doAPVRestore;
  bool produceRawDigis;
  bool produceCalculatedBaseline;
  bool produceBaselinePoints;
  bool storeInZScollBadAPV;
  bool mergeCollections;
  bool fixCM;
  bool useCMMeanMap;

};
#endif






















