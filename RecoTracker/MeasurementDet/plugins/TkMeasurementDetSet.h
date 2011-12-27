#ifndef TMeasurementDetSet_H
#define TkMeasurementDetSet_H

#include<vector>
class TkGluedMeasurementDet;
class SiStripRecHitMatcher;
class StripClusterParameterEstimator;
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/* Struct of arrays supporting "members of Tk...MeasurementDet
 * implemented with vectors, to be optimized...
 */


class TkMeasurementDetSet {
public:
  struct BadStripCuts {
     BadStripCuts() : maxBad(9999), maxConsecutiveBad(9999) {}
     BadStripCuts(const edm::ParameterSet &pset) :
        maxBad(pset.getParameter<uint32_t>("maxBad")),
        maxConsecutiveBad(pset.getParameter<uint32_t>("maxConsecutiveBad")) {}
     uint16_t maxBad, maxConsecutiveBad;
  };


  TkMeasurementDetSet(const SiStripRecHitMatcher* matcher,
		      const StripClusterParameterEstimator* cpe,
		      bool regional):
    theMatcher(matcher), theCPE(cpe), skipClusters_(0),isRegional(regional){}
  
  


private:

  vector<TkStripMeasurementDet*> theStripDets;
  
  // globals
  const SiStripRecHitMatcher*       theMatcher;
  const StripClusterParameterEstimator* theCPE;
  const std::vector<bool> * skipClusters_;
  bool isRegional;

  BadStripCuts badStripCuts[4];

  // members of TkStripMeasurementDet
  std::vector<unsigned int> id_;
  std::vector<unsigned char> subId_;

  std::vector<int> totalStrips_;

  const int nbad128 = 6;
  std::vector<bool> bad128Strip_;
  std::vector<bool> hasAny128StripBad_, maskBad128StripBlocks_;

  std::vector<bool> empty_;

  std::vector<bool> activeThisEvent_,activeThisPeriod_;

  // full reco
  std::vector<detset> detSet_;
  std::vector<edm::Handle<edmNew::DetSetVector<SiStripCluster>> > handle_;


  // --- regional unpacking
  std::vector<edm::Handle<edm::LazyGetter<SiStripCluster>> > regionalHandle_;

  std::vector<unsigned int> beginClusterI_;
  std::vector<unsigned int> endClusterI_;

 


};
