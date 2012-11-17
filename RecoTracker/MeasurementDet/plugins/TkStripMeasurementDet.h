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
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"

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
  
  TkStripMeasurementDet( const GeomDet* gdet, StMeasurementDetSet & dets);


  void setIndex(int i) { index_=i;}
  
  void update( const detset &detSet ) { 
    theDets().update(index(),detSet);
  }
  void update( std::vector<SiStripCluster>::const_iterator begin ,std::vector<SiStripCluster>::const_iterator end ) { 
    theDets().update(index(), begin, end);
  }
  
  bool isRegional() const { return theDets().isRegional();}
  
  void setEmpty(){ theDets().setEmpty(index()); }
  
  bool  isEmpty() const {return theDets().empty(index());}
  
  int index() const { return index_;}

  unsigned int rawId() const { return theDets().id(index()); }
  unsigned char subId() const { return theDets().subId(index());}
  
  
  const detset & theSet() const {return theDets().detSet(index());}
  const detset & detSet() const {return theDets().detSet(index());}
  detset & detSet() { return theDets().detSet(index());}
  unsigned int beginClusterI() const {return theDets().beginClusterI(index());}
  unsigned int endClusterI() const {return theDets().endClusterI(index());}
  
  int  size() const {return endClusterI() - beginClusterI() ; }
  
  
  /** \brief Is this module active in reconstruction? It must be both 'setActiveThisEvent' and 'setActive'. */
  bool isActive() const { return theDets().isActive(index()); }
  
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
  

  template<class ClusterRefT>
  TransientTrackingRecHit::RecHitPointer
  buildRecHit( const ClusterRefT &cluster, const TrajectoryStateOnSurface& ltp) const {
    const GeomDetUnit& gdu( specificGeomDet());
    LocalValues lv = cpe()->localParameters( *cluster, gdu, ltp);
    return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, &fastGeomDet(), cluster, cpe());
  }
  
  
  template<class ClusterRefT>
  TkStripMeasurementDet::RecHitContainer 
  buildRecHits( const ClusterRefT& cluster, const TrajectoryStateOnSurface& ltp) const {
    RecHitContainer res;
    const GeomDetUnit& gdu( specificGeomDet());
    VLocalValues vlv = cpe()->localParametersV( *cluster, gdu, ltp);
    for(VLocalValues::const_iterator it=vlv.begin();it!=vlv.end();++it)
      res.push_back(TSiStripRecHit2DLocalPos::build( it->first, it->second, &fastGeomDet(), cluster, cpe()));
    return res;
  }
  
  
  
  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually).
      This also resets the 'setActiveThisEvent' to true */
  void setActive(bool active) { theDets().setActive(index(),active);}
  /** \brief Turn on/off the module for reconstruction for one events.
      This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(bool active) {  theDets().setActiveThisEvent(index(),active); }
  
  /** \brief does this module have at least one bad strip, APV or channel? */
  bool hasAllGoodChannels() const { return (!hasAny128StripBad()) && badStripBlocks().empty(); }
  
  /** \brief Sets the status of a block of 128 strips (or all blocks if idx=-1) */
  void set128StripStatus(bool good, int idx=-1) {
    theDets().set128StripStatus(index(),good,idx);
  }
  
  typedef StMeasurementDetSet::BadStripCuts BadStripCuts;
  
  /** \brief return true if there are 'enough' good strips in the utraj +/- 3 uerr range.*/
  bool testStrips(float utraj, float uerr) const;
  
  typedef StMeasurementDetSet::BadStripBlock BadStripBlock;
  
  std::vector<BadStripBlock> & getBadStripBlocks() { return theDets().getBadStripBlocks(index()); }
  std::vector<BadStripBlock> const & badStripBlocks() const { return theDets().badStripBlocks(index()); }

  bool maskBad128StripBlocks() const { return theDets().maskBad128StripBlocks();}
  

  
private:
  
  StMeasurementDetSet  & theDets() { return *theDets_;}
  StMeasurementDetSet  & theDets() const { return *theDets_;}
  
  StMeasurementDetSet * theDets_;
  int index_;
  


  edm::Handle<edmNew::DetSetVector<SiStripCluster> > const & handle() const { return theDets().handle();}
  edm::Handle<edm::LazyGetter<SiStripCluster> > const & regionalHandle() const { return theDets().regionalHandle();}
  
  const StripClusterParameterEstimator* cpe() const { return  theDets().stripCPE(); }
  
  
  const std::vector<bool> & skipClusters() const {  return  theDets().clusterToSkip();}
  
  // --- regional unpacking
  
  int totalStrips() const { return theDets().totalStrips(index()); }
  BadStripCuts const & badStripCuts() const { return theDets().badStripCuts(index());}
  
  bool hasAny128StripBad() const { return  theDets().hasAny128StripBad(index()); } 
  
  
  
  
  inline bool isMasked(const SiStripCluster &cluster) const {
    return theDets().isMasked(index(), cluster);
  }
  

  template<class ClusterRefT>
  void buildSimpleRecHit( const ClusterRefT& cluster,
			  const TrajectoryStateOnSurface& ltp,
			  std::vector<SiStripRecHit2D>& res) const {
    const GeomDetUnit& gdu( specificGeomDet());
    VLocalValues vlv = cpe()->localParametersV( *cluster, gdu, ltp);
    for(VLocalValues::const_iterator it=vlv.begin();it!=vlv.end();++it){
      res.push_back(SiStripRecHit2D( it->first, it->second, rawId(), cluster));
    }
  }
 
  
  
  
public:
  inline bool accept(SiStripClusterRef & r) const {
    if(skipClusters().empty()) return true;
    if (r.key()>=skipClusters().size()){
      edm::LogError("WrongStripMasking")<<r.key()<<" is larger than: "<<skipClusters().size()<<" no skipping done";
      return true;
    }
    return (not (skipClusters()[r.key()]));
  }
  inline bool accept(SiStripRegionalClusterRef &r) const{
    if(skipClusters().empty()) return true;
    if (r.key()>=skipClusters().size()){
      LogDebug("TkStripMeasurementDet")<<r.key()<<" is larger than: "<<skipClusters().size()
				       <<"\n This must be a new cluster, and therefore should not be skiped most likely.";
      return true;
    }
    return (not (skipClusters()[r.key()]));
  }

};

#endif
