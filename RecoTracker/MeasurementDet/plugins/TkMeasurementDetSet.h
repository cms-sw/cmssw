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

  struct BadStripBlock {
      short first;
      short last;
      BadStripBlock(const SiStripBadStrip::data &data) : first(data.firstStrip), last(data.firstStrip+data.range-1) { }
  };


  TkMeasurementDetSet(const SiStripRecHitMatcher* matcher,
		      const StripClusterParameterEstimator* cpe,
		      bool regional):
    theMatcher(matcher), theCPE(cpe), skipClusters_(0),isRegional(regional){}
  

  void init() {

    
    //intialize the detId !
    id_[i] = mdet->gdet->geographicalId().rawId();
    //initalize the total number of strips
    totalStrips_[i] =  mdet->specificGeomDet().specificTopology().nstrips();

  }


  const std::vector<TkStripMeasurementDet*>& stripDets() const {return  theStripDets;}

  void setClusterToSkip(const std::vector<bool>* toSkip){
    skipClusters_ = toSkip;
  }

  void update(int i,
	      const detset &detSet, 
	       const edm::Handle<edmNew::DetSetVector<SiStripCluster> > h ) { 
    detSet_[i] = detSet; 
    handle_[i] = h;

    empty[i] = false;
  }

  void update(int i,
	      std::vector<SiStripCluster>::const_iterator begin ,std::vector<SiStripCluster>::const_iterator end, 
	       const edm::Handle<edm::LazyGetter<SiStripCluster> > h ) { 
    regionalHandle_[i] = h;
    beginClusterI_[i] = begin - regionalHandle_->begin_record();
    endClusterI_[i] = end - regionalHandle_->begin_record();

    empty[i] = false;
    activeThisEvent_[i] = true;
  }


  bool empty(int i) const { return empty_[i];}  
  bool isActive(int i) const { return activeThisEvent_[i] && activeThisPeriod_[i]; }
  void setEmpty(int i) {empty_[i] = true; activeThisEvent_[i] = true; }

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually).
             This also resets the 'setActiveThisEvent' to true */
  void setActive(int it, bool active) { activeThisPeriod_[i] = active; activeThisEvent_[i] = true; if (!active) empty_[i] = true; }
  /** \brief Turn on/off the module for reconstruction for one events.
             This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(int i, bool active) { activeThisEvent_[i] = active;  if (!active) empty[i] = true; }



  unsigned int beginClusterI(int i) const {return beginClusterI[i];}
  unsigned int endClusterI(int i) const {return endClusterI[i];}

  int totalStrips(int) const { return totalStrips_[i];}

  BadStripCuts & badStripCuts(int i) { return  badStripCuts(subId_[i]);}

  bool hasAny128StripBad(int i) const { return  hasAny128StripBad_[i];}

  bool isMasked(int i, const SiStripCluster &cluster) const {
    int offset =  nbad128*i;
    if ( bad128Strip_[offset+cluster.firstStrip() >> 7] ) {
      if ( bad128Strip_[offset+(cluster.firstStrip()+cluster.amplitudes().size())  >> 7] ||
	   bad128Strip_[offset+static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7] ) {
	return true;
      }
    } else {
      if ( bad128Strip_[offset+(cluster.firstStrip()+cluster.amplitudes().size())  >> 7] &&
	   bad128Strip_[offset+static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7] ) {
	return true;
      }
    }
    return false;
  }
 

  void set128StripStatus(int i, bool good, int idx) { 
    int offset =  nbad128*i;
    if (idx == -1) {
      std::fill(bad128Strip_[offset], bad128Strip_[offset+6], !good);
      hasAny128StripBad_[i] = !good;
    } else {
      bad128Strip_[offset+idx] = !good;
      if (good == false) {
	hasAny128StripBad_[i] = false;
      } else { // this should not happen, as usually you turn on all fibers
	// and then turn off the bad ones, and not vice-versa,
	// so I don't care if it's not optimized
	hasAny128StripBad_[i] = true;
	for (int j = 0; i < (totalStrips_[j] >> 7); j++) {
	  if (bad128Strip_[j+offset] == false) hasAny128StripBad_[i] = false;
	}
      }    
    } 
  }

private:

  friend class  MeasurementTrackerImpl;
  mutable vector<TkStripMeasurementDet*> theStripDets;
  
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
