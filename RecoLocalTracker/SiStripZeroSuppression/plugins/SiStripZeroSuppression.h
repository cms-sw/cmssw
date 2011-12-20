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

  void processRaw(const edm::InputTag&, const edm::DetSetVector<SiStripRawDigi>&);
  void storeExtraOutput(uint32_t, int16_t);
  void formatRawDigis(edm::DetSetVector<SiStripRawDigi>::const_iterator, edm::DetSet<SiStripRawDigi>&);
  void storeCMN(uint32_t, const std::vector< std::pair<short,float> >&);
  void storeBaseline(uint32_t, const std::vector< std::pair<short,float> >&);
  void storeBaselinePoints(uint32_t);
  void StandardZeroSuppression(edm::Event&);
  void CollectionMergedZeroSuppression(edm::Event&);
  void MergeCollectionsZeroSuppression(edm::Event&);

  std::vector<edm::InputTag> inputTags;
  edm::InputTag DigisToMergeZS;
  edm::InputTag DigisToMergeVR;

  typedef std::vector<edm::InputTag>::const_iterator tag_iterator_t;
  std::vector<edm::DetSet<SiStripDigi> > output_base; 
  std::vector<edm::DetSet<SiStripRawDigi> > output_base_raw; 
  std::vector< edm::DetSet<SiStripProcessedRawDigi> > output_apvcm; 
  std::vector< edm::DetSet<SiStripProcessedRawDigi> > output_baseline;
  std::vector< edm::DetSet<SiStripDigi> > output_baseline_points;
  std::auto_ptr<SiStripRawProcessingAlgorithms> algorithms;
  
  bool storeCM;
  bool produceRawDigis;
  bool produceCalculatedBaseline;
  bool produceBaselinePoints;
  bool storeInZScollBadAPV;
  bool mergeCollections;
  bool fixCM;
  
};
#endif






















