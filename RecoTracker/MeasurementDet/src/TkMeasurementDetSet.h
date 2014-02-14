#ifndef StMeasurementDetSet_H
#define StMeasurementDetSet_H

#include<vector>
class TkStripMeasurementDet;
class TkStripMeasurementDet;
class TkPixelMeasurementDet;
class SiStripRecHitMatcher;
class StripClusterParameterEstimator;
class PixelClusterParameterEstimator;

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefGetter.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// #define VISTAT

#ifdef VISTAT
#include<iostream>
#define COUT std::cout
#else
#define COUT LogDebug("")
#endif

/* Struct of arrays supporting "members of Tk...MeasurementDet
 * implemented with vectors, to be optimized...
   ITEMS THAT DO NOT DEPEND ON THE EVENT
 */
class StMeasurementConditionSet {
public:
  enum QualityFlags { BadModules=1, // for everybody
                       /* Strips: */ BadAPVFibers=2, BadStrips=4, MaskBad128StripBlocks=8, 
                       /* Pixels: */ BadROCs=2 }; 

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
  
  
  StMeasurementConditionSet(const SiStripRecHitMatcher* matcher,
		           const StripClusterParameterEstimator* cpe,
		           bool regional):
    theMatcher(matcher), theCPE(cpe), regional_(regional){}
  
  
  void init(int size);
 
  const SiStripRecHitMatcher*  matcher() const { return theMatcher;}
  const StripClusterParameterEstimator*  stripCPE() const { return theCPE;}


  int nDet() const { return id_.size();}
  unsigned int id(int i) const { return id_[i]; }
  unsigned char subId(int i) const { return subId_[i];}

  int find(unsigned int jd, int i=0) const {
    return std::lower_bound(id_.begin()+i,id_.end(),jd)-id_.begin();
  }
  
  bool isRegional() const { return regional_;}
  bool isActiveThisPeriod(int i) const { return activeThisPeriod_[i]; }

 
  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually) */
  void setActive(int i, bool active) { activeThisPeriod_[i] = active; }
  
  int totalStrips(int i) const { return totalStrips_[i];}
  
  void setMaskBad128StripBlocks(bool maskThem) { maskBad128StripBlocks_ = maskThem; }
  const BadStripCuts & badStripCuts(int i) const { return  badStripCuts_[subId_[i]];}
  
  bool maskBad128StripBlocks() const { return maskBad128StripBlocks_;}
  bool hasAny128StripBad(int i) const { return  hasAny128StripBad_[i];}
  
  std::vector<BadStripBlock> & getBadStripBlocks(int i) { return badStripBlocks_[i]; }
  std::vector<BadStripBlock> const & badStripBlocks(int i) const {return badStripBlocks_[i]; }

  bool isMasked(int i, const SiStripCluster &cluster) const {
    int offset =  nbad128*i;
    if ( bad128Strip_[offset+( cluster.firstStrip() >> 7)] ) {
      if ( bad128Strip_[offset+( (cluster.firstStrip()+cluster.amplitudes().size())  >> 7)] ||
	   bad128Strip_[offset+( static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7)] ) {
	return true;
      }
    } else {
      if ( bad128Strip_[offset+( (cluster.firstStrip()+cluster.amplitudes().size())  >> 7)] &&
	   bad128Strip_[offset+( static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7)] ) {
	return true;
      }
    }
    return false;
  }
  
  
  void set128StripStatus(int i, bool good, int idx=-1);  

private:
  
  friend class  MeasurementTrackerImpl;
  
  // globals
  const SiStripRecHitMatcher*       theMatcher;
  const StripClusterParameterEstimator* theCPE;
  bool  regional_;
  
  bool maskBad128StripBlocks_;
  BadStripCuts badStripCuts_[4];
  
  // members of TkStripMeasurementDet
  std::vector<unsigned int> id_;
  std::vector<unsigned char> subId_;
  
  std::vector<int> totalStrips_;
  
  static const int nbad128 = 6;
  std::vector<bool> bad128Strip_;
  std::vector<bool> hasAny128StripBad_;
  
  std::vector<std::vector<BadStripBlock>> badStripBlocks_;  

  std::vector<bool> activeThisPeriod_;
};

class StMeasurementDetSet {
public:

  typedef edmNew::DetSet<SiStripCluster> StripDetset;
  typedef StripDetset::const_iterator new_const_iterator;
  
  typedef std::vector<SiStripCluster>::const_iterator const_iterator;
  
  typedef edm::LazyGetter<SiStripCluster> LazyGetter;
  typedef edm::RefGetter<SiStripCluster> RefGetter;

  StMeasurementDetSet(const StMeasurementConditionSet & cond) : 
    conditionSet_(&cond),
    empty_(cond.nDet(), true),
    activeThisEvent_(cond.nDet(), true),
    detSet_(!cond.isRegional() ? cond.nDet() : 0),
    detIndex_(!cond.isRegional() ? cond.nDet() : 0,-1),
    ready_(!cond.isRegional() ? cond.nDet() : 0,true),
    clusterI_(cond.isRegional() ? 2*cond.nDet() : 0),
    refGetter_(0),
    theRawInactiveStripDetIds_(),
    stripDefined_(cond.isRegional() ? cond.nDet() : 0), 
    stripUpdated_(cond.isRegional() ? cond.nDet() : 0), 
    stripRegions_(cond.isRegional() ? cond.nDet() : 0) 
  {
  }

  ~StMeasurementDetSet() {
    printStat();
  }

  const StMeasurementConditionSet & conditions() const { return *conditionSet_; } 
 
  void setLazyGetter( edm::Handle<LazyGetter> const & lg) { regionalHandle_=lg;}
 
  void update(int i,const StripDetset & detSet ) { 
    detSet_[i] = detSet;     
    empty_[i] = false;
  }

  void update(int i, int j ) {
    assert(j>=0); assert(empty_[i]); assert(ready_[i]); 
    detIndex_[i] = j;
    empty_[i] = false;
    incReady();
  }

  void update(int i, std::vector<SiStripCluster>::const_iterator begin ,std::vector<SiStripCluster>::const_iterator end) { 
    clusterI_[2*i] = begin - regionalHandle_->begin_record();
    clusterI_[2*i+1] = end - regionalHandle_->begin_record();
    
    empty_[i] = false;
    activeThisEvent_[i] = true;
  }
 
  bool isRegional() const { return conditions().isRegional(); }
 
  int size() const { return conditions().nDet(); }
  int nDet() const { return size();}
  unsigned int id(int i) const { return conditions().id(i); }
  int find(unsigned int jd, int i=0) const  {
    return conditions().find(jd,i);
  }
  
  bool empty(int i) const { return empty_[i];}  
  bool isActive(int i) const { return activeThisEvent_[i] && conditions().isActiveThisPeriod(i); }

  void setEmpty(int i) {empty_[i] = true; activeThisEvent_[i] = true; }
  void setUpdated(int i) { stripUpdated_[i] = true; }
  
  void setEmpty() {
    printStat();
    std::fill(empty_.begin(),empty_.end(),true);
    std::fill(ready_.begin(),ready_.end(),true);
    std::fill(detIndex_.begin(),detIndex_.end(),-1);
    std::fill(activeThisEvent_.begin(), activeThisEvent_.end(),true);
    incTot(size());
  }
  
  /** \brief Turn on/off the module for reconstruction for one events.
      This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(int i, bool active) { activeThisEvent_[i] = active;  if (!active) empty_[i] = true; }
  
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > & handle() {  return handle_; }
  const edm::Handle<edmNew::DetSetVector<SiStripCluster> > & handle() const {  return handle_; }
  // StripDetset & detSet(int i) { return detSet_[i]; }
  const StripDetset & detSet(int i) const { if (ready_[i]) const_cast<StMeasurementDetSet*>(this)->getDetSet(i);     return detSet_[i]; }
  
  edm::Handle<edm::LazyGetter<SiStripCluster> > & regionalHandle() { return regionalHandle_; }
  const edm::Handle<edm::LazyGetter<SiStripCluster> > & regionalHandle() const { return regionalHandle_; }
  unsigned int beginClusterI(int i) const {return clusterI_[2*i];}
  unsigned int endClusterI(int i) const {return clusterI_[2*i+1];}
  
  const edm::RefGetter<SiStripCluster> & refGetter() const { return *refGetter_; }
  void setRefGetter(const edm::RefGetter<SiStripCluster> &getter) { refGetter_ = &getter; }

  //// ------- pieces for on-demand unpacking -------- 
  std::vector<uint32_t> & rawInactiveStripDetIds() { return theRawInactiveStripDetIds_; } 
  const std::vector<uint32_t> & rawInactiveStripDetIds() const { return theRawInactiveStripDetIds_; } 

  void resetOnDemandStrips() { std::fill(stripDefined_.begin(), stripDefined_.end(), false); std::fill(stripUpdated_.begin(), stripUpdated_.end(), false); }
  const bool stripDefined(int i) const { return stripDefined_[i]; }
  const bool stripUpdated(int i) const { return stripUpdated_[i]; }
  void defineStrip(int i, std::pair<unsigned int, unsigned int> range) {
    stripDefined_[i] = true;
    stripUpdated_[i] = false;
    stripRegions_[i] = range; 
  }
  //const bool gluedUpdated(int i) const { return gluedUpdated_(i); }
  const std::pair<unsigned int,unsigned int> & regionRange(int i) const { return stripRegions_[i]; }

private:

  void getDetSet(int i) {
    if(detIndex_[i]>=0) {
      detSet_[i].set(*handle_,handle_->item(detIndex_[i]));
      empty_[i]=false; // better be false already
      incAct();
    }  else { // we should not be here
      detSet_[i] = StripDetset();
      empty_[i]=true;  
    }
    ready_[i]=false;
    incSet();
  }


  friend class  MeasurementTrackerImpl;
  friend class  MeasurementTrackerSiStripRefGetterProducer;

  const StMeasurementConditionSet *conditionSet_; 
 
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > handle_;
  edm::Handle<edm::LazyGetter<SiStripCluster> > regionalHandle_;
  
  std::vector<bool> empty_;
  std::vector<bool> activeThisEvent_;
  
  // full reco
  std::vector<StripDetset> detSet_;
  std::vector<int> detIndex_;
  std::vector<bool> ready_; // to be cleaned
  
  // --- regional unpacking
  // begin,end "pairs"
  std::vector<unsigned int> clusterI_;
 
  //// ------- pieces for on-demand unpacking -------- 
  const edm::RefGetter<SiStripCluster> * refGetter_;
  // note: not aligned to the index
  std::vector<uint32_t> theRawInactiveStripDetIds_;
  // keyed on si-strip index
  std::vector<bool> stripDefined_, stripUpdated_;
  std::vector<std::pair<unsigned int, unsigned int> > stripRegions_;
  // keyed on glued
  // std::vector<bool> gluedUpdated_;



#ifdef VISTAT
  struct Stat {
    int totDet=0; // all dets
    int detReady=0; // dets "updated"
    int detSet=0;  // det actually set not empty
    int detAct=0;  // det actually set with content
  };

  mutable Stat stat;
  void zeroStat() const { stat = Stat(); }
  void incTot(int n) const { stat.totDet=n;}
  void incReady() const { stat.detReady++;}
  void incSet() const { stat.detSet++;}
  void incAct() const { stat.detAct++;}
  void printStat() const {
    COUT << "VI detsets " << stat.totDet <<','<< stat.detReady <<','<< stat.detSet <<','<< stat.detAct << std::endl;
  }

#else
  static void zeroStat(){}
  static void incTot(int){}
  static void incReady() {}
  static void incSet() {}
  static void incAct() {}
  static void printStat(){}
#endif
   
};

class PxMeasurementConditionSet {
public:
  PxMeasurementConditionSet(const PixelClusterParameterEstimator *cpe) :
    theCPE(cpe) {}

  void init(int size);

  int nDet() const { return id_.size();}
  unsigned int id(int i) const { return id_[i]; }
  int find(unsigned int jd, int i=0) const {
    return std::lower_bound(id_.begin()+i,id_.end(),jd)-id_.begin();
  }

  const PixelClusterParameterEstimator*  pixelCPE() const { return theCPE;}
  bool isActiveThisPeriod(int i) const { return activeThisPeriod_[i]; }

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually).
      This also resets the 'setActiveThisEvent' to true */
  void setActive(int i, bool active) { activeThisPeriod_[i] = active; }

 
private:
  friend class MeasurementTrackerImpl;

  // Globals (not-per-event)
  const PixelClusterParameterEstimator* theCPE;

  // Locals, per-event
  std::vector<unsigned int> id_;
  std::vector<bool> activeThisPeriod_;

};


class PxMeasurementDetSet {
public:
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> SiPixelClusterRef;
  typedef edmNew::DetSet<SiPixelCluster> PixelDetSet;

  PxMeasurementDetSet(const PxMeasurementConditionSet &cond) : 
    conditionSet_(&cond),
    detSet_(cond.nDet()),
    empty_(cond.nDet(), true),
    activeThisEvent_(cond.nDet(), true)  {}

  const PxMeasurementConditionSet & conditions() const { return *conditionSet_; } 

  int size() const { return conditions().nDet(); }
  int nDet() const { return size();}
  unsigned int id(int i) const { return conditions().id(i); }
  int find(unsigned int jd, int i=0) const {
    return conditions().find(jd,i);
  }

  void update(int i,const PixelDetSet & detSet ) { 
    detSet_[i] = detSet;     
    empty_[i] = false;
  }

  bool empty(int i) const { return empty_[i];}  
  bool isActive(int i) const { return activeThisEvent_[i] && conditions().isActiveThisPeriod(i); }

  void setEmpty(int i) {empty_[i] = true; activeThisEvent_[i] = true; }
  
  void setEmpty() {
    std::fill(empty_.begin(),empty_.end(),true);
    std::fill(activeThisEvent_.begin(), activeThisEvent_.end(),true);
  }
  void setActiveThisEvent(bool active) {
    std::fill(activeThisEvent_.begin(), activeThisEvent_.end(),active);
  }
  
  /** \brief Turn on/off the module for reconstruction for one events.
      This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(int i, bool active) { activeThisEvent_[i] = active;  if (!active) empty_[i] = true; }
  const edm::Handle<edmNew::DetSetVector<SiPixelCluster> > & handle() const {  return handle_;}
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > & handle() {  return handle_;}
  const PixelDetSet & detSet(int i) const { return detSet_[i];}
private:
  friend class MeasurementTrackerImpl;

  const PxMeasurementConditionSet *conditionSet_;

  // Globals, per-event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > handle_;

  // Locals, per-event
  std::vector<PixelDetSet> detSet_;
  std::vector<bool> empty_;
  std::vector<bool> activeThisEvent_;
};

#endif // StMeasurementDetSet_H
