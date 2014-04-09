#ifndef TkStripMeasurementDet_H
#define TkStripMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

#include<tuple>

class TransientTrackingRecHit;


class TkStripMeasurementDet;

struct TkStripRecHitIter {
  using detset = edmNew::DetSet<SiStripCluster>;
  using new_const_iterator =  detset::const_iterator;
  

  TkStripRecHitIter(){}
  TkStripRecHitIter(const TkStripMeasurementDet & imdet,
		    const TrajectoryStateOnSurface & itsos,
		    const MeasurementTrackerEvent & idata) : mdet(&imdet),tsos(&itsos),data(&idata){}
  
  TkStripRecHitIter(new_const_iterator ci, new_const_iterator ce, 
		    const TkStripMeasurementDet & imdet,
		    const TrajectoryStateOnSurface & itsos,
		    const MeasurementTrackerEvent & idata) 
    : mdet(&imdet),tsos(&itsos),data(&idata), clusterI(ci), clusterE(ce) {}
  
  
  const TkStripMeasurementDet * mdet = 0;
  const TrajectoryStateOnSurface * tsos=0;
  const MeasurementTrackerEvent * data=0;
  
  new_const_iterator clusterI;
  new_const_iterator clusterE;
  
  inline SiStripRecHit2D buildHit() const;
  inline void advance();
  
public:
  
  bool empty() const { return clusterI==clusterE; }
  
  bool operator==(TkStripRecHitIter const & rh) {
    return clusterI==rh.clusterI;
  }
  bool operator!=(TkStripRecHitIter const & rh) {
    return clusterI!=rh.clusterI;
  }
  bool operator<(TkStripRecHitIter const & rh) {
    return clusterI<rh.clusterI;
  }
  
  TkStripRecHitIter & operator++() {
    advance();
    return *this;
  }
  
  SiStripRecHit2D operator*() const {
    return buildHit();
  } 
  
};


class TkStripMeasurementDet GCC11_FINAL : public MeasurementDet {
public:
  
  typedef StripClusterParameterEstimator::LocalValues    LocalValues;
  typedef StripClusterParameterEstimator::VLocalValues    VLocalValues;
  
  typedef SiStripRecHit2D::ClusterRef SiStripClusterRef;
  
  typedef edmNew::DetSet<SiStripCluster> detset;
  typedef detset::const_iterator new_const_iterator;
  
  typedef std::vector<SiStripCluster>::const_iterator const_iterator;
  
  virtual ~TkStripMeasurementDet(){}
  
  TkStripMeasurementDet( const GeomDet* gdet, StMeasurementConditionSet & conditionSet );

  void setIndex(int i) { index_=i;}
  
  void setEmpty(StMeasurementDetSet & theDets) const { theDets.setEmpty(index()); }
  
  bool  isEmpty(const StMeasurementDetSet & theDets) const {return theDets.empty(index());}
  
  int index() const { return index_;}

  unsigned int rawId() const { return conditionSet().id(index()); }
  unsigned char subId() const { return conditionSet().subId(index());}
  
  
  const detset & theSet(const StMeasurementDetSet & theDets) const {return theDets.detSet(index());}
  const detset & detSet(const StMeasurementDetSet & theDets) const {return theDets.detSet(index());}

  
  /** \brief Is this module active in reconstruction? It must be both 'setActiveThisEvent' and 'setActive'. */
  bool isActive(const MeasurementTrackerEvent & data) const { return data.stripData().isActive(index()); }
  
  //TO BE IMPLEMENTED
  bool hasBadComponents( const TrajectoryStateOnSurface &tsos, const MeasurementTrackerEvent & data ) const {return false;}
  
  
  std::tuple<TkStripRecHitIter,TkStripRecHitIter> hitRange(const TrajectoryStateOnSurface&, const MeasurementTrackerEvent & data) const;
  void advance(TkStripRecHitIter & hi ) const;
  SiStripRecHit2D hit(TkStripRecHitIter const & hi ) const;
  
  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&, const MeasurementTrackerEvent & data) const;


  bool empty(const MeasurementTrackerEvent & data) const;

  void simpleRecHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data, std::vector<SiStripRecHit2D> &result) const ;
  bool simpleRecHits( const TrajectoryStateOnSurface& ts, const MeasurementEstimator& est, const MeasurementTrackerEvent & data, std::vector<SiStripRecHit2D> &result) const ;
  
  virtual bool recHits( const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
			RecHitContainer & result, std::vector<float> & diffs) const;
  
  virtual bool measurements( const TrajectoryStateOnSurface& stateOnThisDet,
			     const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
			     TempMeasurements & result) const;
  
  const StripGeomDetUnit& specificGeomDet() const {return static_cast<StripGeomDetUnit const &>(fastGeomDet());}
  

  template<class ClusterRefT>
  TransientTrackingRecHit::RecHitPointer
  buildRecHit( const ClusterRefT &cluster, const TrajectoryStateOnSurface& ltp) const {
    const GeomDetUnit& gdu( specificGeomDet());
    LocalValues lv = cpe()->localParameters( *cluster, gdu, ltp);
    return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, &fastGeomDet(), cluster, cpe());
  }
  
  
  template<class ClusterRefT>
    void
    buildRecHits( const ClusterRefT& cluster, const TrajectoryStateOnSurface& ltp,  const RecHitContainer& _res) const {
    RecHitContainer res = _res;
    const GeomDetUnit& gdu( specificGeomDet());
    VLocalValues vlv = cpe()->localParametersV( *cluster, gdu, ltp);
    for(VLocalValues::const_iterator it=vlv.begin();it!=vlv.end();++it)
      res.push_back(TSiStripRecHit2DLocalPos::build( it->first, it->second, &fastGeomDet(), cluster, cpe()));
  }
  

  template<class ClusterRefT>
  bool filteredRecHits( const ClusterRefT& cluster, const TrajectoryStateOnSurface& ltp,  const MeasurementEstimator& est, const std::vector<bool> & skipClusters,
			RecHitContainer & result, std::vector<float> & diffs) const {
    if (isMasked(*cluster)) return true;
    const GeomDetUnit& gdu( specificGeomDet());
    if (!accept(cluster, skipClusters)) return true;
    VLocalValues const & vlv = cpe()->localParametersV( *cluster, gdu, ltp);
    bool isCompatible(false);
    for(auto vl : vlv) {
      auto && recHit  = TSiStripRecHit2DLocalPos::build( vl.first, vl.second, &fastGeomDet(), cluster, cpe()); 
      std::pair<bool,double> diffEst = est.estimate(ltp, *recHit);
      LogDebug("TkStripMeasurementDet")<<" chi2=" << diffEst.second;
      if ( diffEst.first ) {
	result.push_back(std::move(recHit));
	diffs.push_back(diffEst.second);
	isCompatible = true;
      }
    }
    return isCompatible;
  }


  template<class ClusterRefT>
  bool filteredRecHits( const ClusterRefT& cluster, const TrajectoryStateOnSurface& ltp,  const MeasurementEstimator& est, const std::vector<bool> & skipClusters,
			std::vector<SiStripRecHit2D> & result) const {
    if (isMasked(*cluster)) return true;
    const GeomDetUnit& gdu( specificGeomDet());
    if (!accept(cluster, skipClusters)) return true;
    VLocalValues const & vlv = cpe()->localParametersV( *cluster, gdu, ltp);
    bool isCompatible(false);
    for(auto vl : vlv) {
      auto && recHit  = SiStripRecHit2D( vl.first, vl.second, gdu, cluster);
      std::pair<bool,double> diffEst = est.estimate(ltp, recHit);
      LogDebug("TkStripMeasurementDet")<<" chi2=" << diffEst.second;
      if ( diffEst.first ) {
	result.push_back(std::move(recHit));
	isCompatible = true;
      }
    }
    return isCompatible;
  }



  
  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually). */
  void setActiveThisPeriod(StMeasurementDetSet & theDets, bool active) { conditionSet().setActive(index(),active);}

  /** \brief Turn on/off the module for reconstruction for one events.
      This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(StMeasurementDetSet & theDets, bool active) const {  theDets.setActiveThisEvent(index(),active); }
  
  /** \brief does this module have at least one bad strip, APV or channel? */
  bool hasAllGoodChannels() const { return (!hasAny128StripBad()) && badStripBlocks().empty(); }
  
  /** \brief Sets the status of a block of 128 strips (or all blocks if idx=-1) */
  void set128StripStatus(bool good, int idx=-1) {
    conditionSet().set128StripStatus(index(),good,idx);
  }
  
  typedef StMeasurementConditionSet::BadStripCuts BadStripCuts;
  
  /** \brief return true if there are 'enough' good strips in the utraj +/- 3 uerr range.*/
  bool testStrips(float utraj, float uerr) const;
  
  typedef StMeasurementConditionSet::BadStripBlock BadStripBlock;
  
  std::vector<BadStripBlock> & getBadStripBlocks() { return conditionSet().getBadStripBlocks(index()); }
  std::vector<BadStripBlock> const & badStripBlocks() const { return conditionSet().badStripBlocks(index()); }

  bool maskBad128StripBlocks() const { return conditionSet().maskBad128StripBlocks();}
  
private:
  int index_;
  StMeasurementConditionSet * theDetConditions;
  StMeasurementConditionSet & conditionSet() { return *theDetConditions; }
  const StMeasurementConditionSet & conditionSet() const { return *theDetConditions; }
  
  const StripClusterParameterEstimator* cpe() const { return  conditionSet().stripCPE(); }

  // --- regional unpacking
  int totalStrips() const { return conditionSet().totalStrips(index()); }
  BadStripCuts const & badStripCuts() const { return conditionSet().badStripCuts(index());}
  
  bool hasAny128StripBad() const { return  conditionSet().hasAny128StripBad(index()); } 
  
  
  inline bool isMasked(const SiStripCluster &cluster) const {
    return conditionSet().isMasked(index(), cluster);
  }
  

  template<class ClusterRefT>
  void buildSimpleRecHit( const ClusterRefT& cluster,
			  const TrajectoryStateOnSurface& ltp,
			  std::vector<SiStripRecHit2D>& res) const {
    const GeomDetUnit& gdu( specificGeomDet());
    VLocalValues const & vlv = cpe()->localParametersV( *cluster, gdu, ltp);
    for(VLocalValues::const_iterator it=vlv.begin();it!=vlv.end();++it){
      res.push_back(SiStripRecHit2D( it->first, it->second, gdu, cluster));
    }
  }



 
  
  
  
public:
  inline bool accept(SiStripClusterRef const & r, const std::vector<bool> & skipClusters) const {
    if(skipClusters.empty()) return true;
   if (r.key()>=skipClusters.size()){
      LogDebug("TkStripMeasurementDet")<<r.key()<<" is larger than: "<<skipClusters.size()
				       <<"\n This must be a new cluster, and therefore should not be skiped most likely.";
      // edm::LogError("WrongStripMasking")<<r.key()<<" is larger than: "<<skipClusters.size()<<" no skipping done"; // protect for on demand???
      return true;
    }
    return (not (skipClusters[r.key()]));
  }

};


inline
SiStripRecHit2D TkStripRecHitIter::buildHit() const {
  return mdet->hit(*this);
}
inline
void TkStripRecHitIter::advance() {
  mdet->advance(*this);
}



#endif
