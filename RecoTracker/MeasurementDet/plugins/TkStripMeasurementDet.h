#ifndef TkStripMeasurementDet_H
#define TkStripMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TkMeasurementDetSet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
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

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TransientTrackingRecHit;

class TkStripMeasurementDet : public MeasurementDet {
public:
  
  typedef StripClusterParameterEstimator::LocalValues    LocalValues;
  typedef StripClusterParameterEstimator::VLocalValues    VLocalValues;
  
  typedef SiStripRecHit2D::ClusterRef SiStripClusterRef;
  
  typedef edm::LazyGetter<SiStripCluster>::value_ref  SiStripRegionalClusterRef;
  
  typedef edmNew::DetSet<SiStripCluster> detset;
  typedef detset::const_iterator new_const_iterator;
  
  typedef std::vector<SiStripCluster>::const_iterator const_iterator;
  
  virtual ~TkStripMeasurementDet(){}
  
  TkStripMeasurementDet( const GeomDet* gdet, TkMeasurementDetSet & dets);
  void setIndex(int i) { index=i;}
  
  void update( const detset &detSet ) { 
    theDets.update(index,detSet);
  }
  void update( std::vector<SiStripCluster>::const_iterator begin ,std::vector<SiStripCluster>::const_iterator end ) { 
    theDets.update(index, begin, end);
  }
  
  bool isRegional() const { return theDets.isRegional();}
  
  void setEmpty(){ theDets.setEmpy(index); }
  
  bool  isEmpty() const {return theDets.empty(index);}
  
  unsigned int rawId() const { return theDets.id(index); }
  unsigned char subId() const { return theDets.subId(index);}
  
  
  const detset& theSet() const {return theDets.detSet(index);}
  detset & detSet() { return theDets.detSet(index);}
  unsigned int beginClusterI() const {return theDets.beginClusterI(index);}
  unsigned int endClusterI() const {return theDets.endClusterI(index);}
  
  int  size() const {return endClusterI() - beginClusterI() ; }
  
  
  /** \brief Is this module active in reconstruction? It must be both 'setActiveThisEvent' and 'setActive'. */
  bool isActive() const { return theDets.isActive(index); }
  
  //TO BE IMPLEMENTED
  bool hasBadComponents( const TrajectoryStateOnSurface &tsos ) const {return false;}
  
  
  
  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&) const;
  void simpleRecHits( const TrajectoryStateOnSurface& ts, std::vector<SiStripRecHit2D> &result) const ;
  
  virtual std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& startingState, 
		    const Propagator&, 
		    const MeasurementEstimator&) const;
  
  const StripGeomDetUnit& specificGeomDet() const {return static_cast<StripGeomDetUnit const &>(fastGeomDet());}
  
  TransientTrackingRecHit::RecHitPointer
  buildRecHit( const SiStripClusterRef&, const TrajectoryStateOnSurface& ltp) const;
  
  TransientTrackingRecHit::RecHitPointer
  buildRecHit( const SiStripRegionalClusterRef&, const TrajectoryStateOnSurface& ltp) const;
  
  
  TkStripMeasurementDet::RecHitContainer 
  buildRecHits( const SiStripClusterRef&, const TrajectoryStateOnSurface& ltp) const;
  
  TkStripMeasurementDet::RecHitContainer 
  buildRecHits( const SiStripRegionalClusterRef&, const TrajectoryStateOnSurface& ltp) const;
  
  
  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually).
      This also resets the 'setActiveThisEvent' to true */
  void setActive(bool active) { theDets.setActive(index,active);}
  /** \brief Turn on/off the module for reconstruction for one events.
      This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(bool active) {  theDets.setActiveThisEvent(index,active); }
  
  /** \brief does this module have at least one bad strip, APV or channel? */
  bool hasAllGoodChannels() const { return !theDets.hasAny128StripBad(index) && badStripBlocks_.empty(); }
  
  /** \brief Sets the status of a block of 128 strips (or all blocks if idx=-1) */
  void set128StripStatus(bool good, int idx=-1) {
    theDets.set128StripStatus(index,good,idx);
  }
  
  typedef TkMeasurementDetSet::BadStripCuts BadStripCuts;
  
  /** \brief return true if there are 'enough' good strips in the utraj +/- 3 uerr range.*/
  bool testStrips(float utraj, float uerr) const;
  
  typedef TkMeasurementDetSet::BadStripBlock BadStripBlock;
  
  std::vector<BadStripBlock> & getBadStripBlocks() { return badStripBlocks_; }
  
  void setMaskBad128StripBlocks(bool maskThem) { maskBad128StripBlocks_ = maskThem; }
  
private:
  
  std::vector<BadStripBlock> badStripBlocks_;  
  
  TkMeasurementDetSet & theDets;
  int index;
  
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > & handle()  { return theDets.handle();}
  edm::Handle<edm::LazyGetter<SiStripCluster> > & regionalHandle() { return theDets.regionalHandle();}
  
  const StripClusterParameterEstimator* cpe() const { return  theDets.stripCpe(); }
  
  
  const std::vector<bool> & skipClusters() const {  return  theDets.skipClusters();}
  
  // --- regional unpacking
  
  int totalStrips() const { return theDets.totalStrips(index); }
  BadStripCuts & badStripCuts() { return theDets.badStripCuts(index);}
  
  bool hasAny128StripBad() const { return  theDets.hasAny128StripBad(index); } 
  bool maskBad128StripBlocks_;
  
  
  
  
  inline bool isMasked(const SiStripCluster &cluster) const {
    return theDets.isMasked(int, cluster);
  }
  
  template<class ClusterRefT>
  void buildSimpleRecHit( const ClusterRefT& cluster,
			  const TrajectoryStateOnSurface& ltp,
			  std::vector<SiStripRecHit2D>& res) const;
  
  
  
public:
  inline bool accept(SiStripClusterRef & r) const {
    if(skipClusters().empty()) return true;
    if (r.key()>=skipClusters().size()){
      edm::LogError("WrongStripMasking")<<r.key()<<" is larger than: "<<skipClusters().size()<<" no skipping done";
      return true;
    }
    return (not (skipClusters())[r.key()]);
  }
  inline bool accept(SiStripRegionalClusterRef &r) const{
    if(0==skipClusters() || skipClusters().empty()) return true;
    if (r.key()>=skipClusters().size()){
      LogDebug("TkStripMeasurementDet")<<r.key()<<" is larger than: "<<skipClusters().size()
				       <<"\n This must be a new cluster, and therefore should not be skiped most likely.";
      return true;
    }
    return (not (skipClusters())[r.key()]);
  }
  
  
  
};

#endif
