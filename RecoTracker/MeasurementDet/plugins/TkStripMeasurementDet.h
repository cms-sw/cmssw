#ifndef TkStripMeasurementDet_H
#define TkStripMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "RecoTracker/MeasurementDet/interface/ClusterFilterPayload.h"

#include<tuple>

class TrackingRecHit;


class TkStripMeasurementDet;

struct dso_hidden TkStripRecHitIter {
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
  
  
  const TkStripMeasurementDet * mdet = nullptr;
  const TrajectoryStateOnSurface * tsos=nullptr;
  const MeasurementTrackerEvent * data=nullptr;
  
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


class dso_hidden TkStripMeasurementDet final : public MeasurementDet {
public:
  
  typedef StripClusterParameterEstimator::LocalValues    LocalValues;
  typedef StripClusterParameterEstimator::VLocalValues    VLocalValues;
  
  typedef SiStripRecHit2D::ClusterRef SiStripClusterRef;
  
  typedef edmNew::DetSet<SiStripCluster> detset;
  typedef detset::const_iterator new_const_iterator;
  
  typedef std::vector<SiStripCluster>::const_iterator const_iterator;
  
  ~TkStripMeasurementDet() override{}
  
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
  bool isActive(const MeasurementTrackerEvent & data) const override { return data.stripData().isActive(index()); }
  
  //TO BE IMPLEMENTED
  bool hasBadComponents( const TrajectoryStateOnSurface &tsos, const MeasurementTrackerEvent & data ) const override {return false;}
  
  
  std::tuple<TkStripRecHitIter,TkStripRecHitIter> hitRange(const TrajectoryStateOnSurface&, const MeasurementTrackerEvent & data) const;
  void advance(TkStripRecHitIter & hi ) const;
  SiStripRecHit2D hit(TkStripRecHitIter const & hi ) const;
  
  RecHitContainer recHits( const TrajectoryStateOnSurface&, const MeasurementTrackerEvent & data) const override;


  bool empty(const MeasurementTrackerEvent & data) const;

  void simpleRecHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data, std::vector<SiStripRecHit2D> &result) const ;
  bool simpleRecHits( const TrajectoryStateOnSurface& ts, const MeasurementEstimator& est, const MeasurementTrackerEvent & data, std::vector<SiStripRecHit2D> &result) const ;
  
  // simple hits
  bool recHits(SimpleHitContainer & result,  
		       const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator&, const MeasurementTrackerEvent & data) const override;

  // TTRH
  bool recHits( const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
			RecHitContainer & result, std::vector<float> & diffs) const override;
  
  
  bool measurements( const TrajectoryStateOnSurface& stateOnThisDet,
			     const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
			     TempMeasurements & result) const override;
  
  const StripGeomDetUnit& specificGeomDet() const {return static_cast<StripGeomDetUnit const &>(fastGeomDet());}
  

  template<class ClusterRefT>
  TrackingRecHit::RecHitPointer
  buildRecHit( const ClusterRefT &cluster, const TrajectoryStateOnSurface& ltp) const {
    const GeomDetUnit& gdu( specificGeomDet());
    LocalValues lv = cpe()->localParameters( *cluster, gdu, ltp);
    return std::make_shared<SiStripRecHit2D>( lv.first, lv.second, fastGeomDet(), cluster);
  }
  
  
  template<class ClusterRefT>
    void
    buildRecHits( const ClusterRefT& cluster, const TrajectoryStateOnSurface& ltp,  const RecHitContainer& _res) const {
    RecHitContainer res = _res;
    const GeomDetUnit& gdu( specificGeomDet());
    VLocalValues vlv = cpe()->localParametersV( *cluster, gdu, ltp);
    for(VLocalValues::const_iterator it=vlv.begin();it!=vlv.end();++it)
      res.push_back(std::make_shared<SiStripRecHit2D>( it->first, it->second, fastGeomDet(), cluster));
  }
  

  template<class ClusterRefT>
  bool filteredRecHits( const ClusterRefT& cluster, StripCPE::AlgoParam const& cpepar,
			const TrajectoryStateOnSurface& ltp,  const MeasurementEstimator& est, const std::vector<bool> & skipClusters,
			RecHitContainer & result, std::vector<float> & diffs) const {
    if (isMasked(*cluster)) return true;
    if (!accept(cluster, skipClusters)) return true;
    if (!est.preFilter(ltp, ClusterFilterPayload(rawId(),&*cluster) )) return true;  // avoids shadow; consistent with previous statement...
    auto const & vl = cpe()->localParameters( *cluster, cpepar);
    SiStripRecHit2D recHit(vl.first, vl.second, fastGeomDet(), cluster); // FIXME add cluster count in OmniRef (and move again to multiple sub-clusters..)
    std::pair<bool,double> diffEst = est.estimate(ltp, recHit);
    LogDebug("TkStripMeasurementDet")<<" chi2=" << diffEst.second;
    if ( diffEst.first ) {
      result.push_back(std::make_shared<SiStripRecHit2D>(recHit));
      diffs.push_back(diffEst.second);
    }
    return diffEst.first;
  }


  template<class ClusterRefT>
    bool filteredRecHits( const ClusterRefT& cluster, StripCPE::AlgoParam const& cpepar,
			  const TrajectoryStateOnSurface& ltp,  const MeasurementEstimator& est, const std::vector<bool> & skipClusters,
			  std::vector<SiStripRecHit2D> & result) const {
    if (isMasked(*cluster)) return true;
    if (!accept(cluster, skipClusters)) return true;
    if (!est.preFilter(ltp, ClusterFilterPayload(rawId(),&*cluster) )) return true;   // avoids shadow; consistent with previous statement...
    auto const & vl = cpe()->localParameters( *cluster, cpepar);
    result.emplace_back( vl.first, vl.second, fastGeomDet(), cluster);   // FIXME add cluster count in OmniRef
    std::pair<bool,double> diffEst = est.estimate(ltp, result.back());
    LogDebug("TkStripMeasurementDet")<<" chi2=" << diffEst.second;
    if ( !diffEst.first ) result.pop_back();
    return diffEst.first;
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
  using AClusters = StripClusterParameterEstimator::AClusters;
  using ALocalValues  = StripClusterParameterEstimator::ALocalValues;

  
  int index_;
  StMeasurementConditionSet * theDetConditions;
  StMeasurementConditionSet & conditionSet() { return *theDetConditions; }
  const StMeasurementConditionSet & conditionSet() const { return *theDetConditions; }
  
  const StripCPE * cpe() const { return  static_cast<const StripCPE *>(conditionSet().stripCPE()); }

  // --- regional unpacking
  int totalStrips() const { return conditionSet().totalStrips(index()); }
  BadStripCuts const & badStripCuts() const { return conditionSet().badStripCuts(index());}
  
  bool hasAny128StripBad() const { return  conditionSet().hasAny128StripBad(index()); } 
  
  
  inline bool isMasked(const SiStripCluster &cluster) const {
    return conditionSet().isMasked(index(), cluster);
  }
  

  void buildSimpleRecHits(AClusters const & clusters, const MeasurementTrackerEvent & data,
			  const detset & detSet,
			  const TrajectoryStateOnSurface& ltp,
			  std::vector<SiStripRecHit2D>& res) const {
    const GeomDetUnit& gdu( specificGeomDet());
    declareDynArray(LocalValues,clusters.size(),alv);
    cpe()->localParameters(clusters, alv, gdu, ltp.localParameters());
    res.reserve(alv.size());
    for (unsigned int i=0; i< clusters.size(); ++i)
      res.emplace_back( alv[i].first, alv[i].second, gdu, detSet.makeRefTo( data.stripData().handle(), clusters[i]) );
    
  }



 
  
  
  
public:
  inline bool accept(SiStripClusterRef const & r, const std::vector<bool> & skipClusters) const {
    return  accept(r.key(), skipClusters);
  }

  inline bool accept(unsigned int key, const std::vector<bool> & skipClusters) const {
    if(skipClusters.empty()) return true;
    if (key>=skipClusters.size()){
      LogDebug("TkStripMeasurementDet")<<key<<" is larger than: "<<skipClusters.size()
				       <<"\n This must be a new cluster, and therefore should not be skiped most likely.";
      return true;
    }
    return (not (skipClusters[key]));
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
