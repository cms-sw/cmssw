#ifndef RecoSelectors_CosmicTrackingParticleSelector_h
#define RecoSelectors_CosmicTrackingParticleSelector_h
/* \class CosmicTrackingParticleSelector
 *
 * \author Yanyan Gao, FNAL
 *
 *  $Date: 2013/06/24 12:25:14 $
 *  $Revision: 1.5 $
 *
 */
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <DataFormats/GeometrySurface/interface/Surface.h>
#include <DataFormats/GeometrySurface/interface/GloballyPositioned.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TrajectoryStateClosestToBeamLineBuilder;

class CosmicTrackingParticleSelector {

public:
  typedef TrackingParticleCollection collection;
  typedef std::vector<const TrackingParticle *> container;
  typedef container::const_iterator const_iterator;
  
  CosmicTrackingParticleSelector(){}

  CosmicTrackingParticleSelector ( double ptMin,double minRapidity,double maxRapidity,
				   double tip,double lip,int minHit, bool chargedOnly, 
				   std::vector<int> pdgId = std::vector<int>()) :
    ptMin_( ptMin ), minRapidity_( minRapidity ), maxRapidity_( maxRapidity ),
    tip_( tip ), lip_( lip ), minHit_( minHit ), chargedOnly_(chargedOnly), pdgId_( pdgId ) { }


    CosmicTrackingParticleSelector  ( const edm::ParameterSet & cfg ) :
      ptMin_(cfg.getParameter<double>("ptMin")),
      minRapidity_(cfg.getParameter<double>("minRapidity")),
      maxRapidity_(cfg.getParameter<double>("maxRapidity")),
      tip_(cfg.getParameter<double>("tip")),
      lip_(cfg.getParameter<double>("lip")),
      minHit_(cfg.getParameter<int>("minHit")),
      chargedOnly_(cfg.getParameter<bool>("chargedOnly")),
      pdgId_(cfg.getParameter<std::vector<int> >("pdgId")) { }
      
      
      void select( const edm::Handle<collection>& c, const edm::Event & event, const edm::EventSetup& setup) {
	selected_.clear();
	edm::Handle<reco::BeamSpot> beamSpot;
	event.getByLabel(edm::InputTag("offlineBeamSpot"), beamSpot); 
	for( TrackingParticleCollection::const_iterator itp = c->begin(); 
	     itp != c->end(); ++ itp )
	  if ( operator()(TrackingParticleRef(c,itp-c->begin()),beamSpot.product(),event,setup) ) {
	    selected_.push_back( & * itp );
	  }
      }
      
    const_iterator begin() const { return selected_.begin(); }
    const_iterator end() const { return selected_.end(); }

    void initEvent(edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssocToSet) const {
      simHitsTPAssoc = simHitsTPAssocToSet;
    }

    // Operator() performs the selection: e.g. if (tPSelector(tp, bs, event, evtsetup)) {...
    bool operator()( const TrackingParticleRef tpr, const reco::BeamSpot* bs, const edm::Event &iEvent, const edm::EventSetup& iSetup ) const { 
      if (chargedOnly_ && tpr->charge()==0) return false;//select only if charge!=0
      //bool testId = false;
      //unsigned int idSize = pdgId_.size();
      //if (idSize==0) testId = true;
      //else for (unsigned int it=0;it!=idSize;++it){
	//if (tpr->pdgId()==pdgId_[it]) testId = true;
      //}
      
      edm::ESHandle<TrackerGeometry> tracker;
      iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
      
      edm::ESHandle<MagneticField> theMF;
      iSetup.get<IdealMagneticFieldRecord>().get(theMF);
      
      GlobalVector finalGV (0,0,0);
      GlobalPoint finalGP(0,0,0);      
      GlobalVector momentum(0,0,0);//At the PCA
      GlobalPoint vertex(0,0,0);//At the PCA
      double radius(9999);
      bool found(0);


      if (simHitsTPAssoc.isValid()==0) {
	edm::LogError("TrackAssociation") << "Invalid handle!";
	return false;
      }
      std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(tpr,TrackPSimHitRef());//SimHit is dummy: for simHitTPAssociationListGreater 
                                                                                                      // sorting only the cluster is needed
      auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(), 
				    clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
      for(auto ip = range.first; ip != range.second; ++ip) {
	TrackPSimHitRef it = ip->second;
	const GeomDet* tmpDet  = tracker->idToDet( DetId(it->detUnitId()) ) ;
	LocalVector  lv = it->momentumAtEntry();
	Local3DPoint lp = it->localPosition ();
	GlobalVector gv = tmpDet->surface().toGlobal( lv );
	GlobalPoint  gp = tmpDet->surface().toGlobal( lp );
	if(gp.perp()<radius){
	  found=true;
	  radius = gp.perp();
	  finalGV = gv;
	  finalGP = gp;
	}
      }
      if(!found) return 0;
      else
	{
	  FreeTrajectoryState ftsAtProduction(finalGP,finalGV,TrackCharge(tpr->charge()),theMF.product());
	  TSCBLBuilderNoMaterial tscblBuilder;
	  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,*bs);//as in TrackProducerAlgorithm
	  if(!tsAtClosestApproach.isValid()){
	    std::cout << "WARNING: tsAtClosestApproach is not valid" << std::endl;
	    return 0;
	  }
	  else
	    {
	      momentum = tsAtClosestApproach.trackStateAtPCA().momentum();
	      vertex = tsAtClosestApproach.trackStateAtPCA().position();
	      return (
		      tpr->numberOfTrackerLayers() >= minHit_ &&
		      sqrt(momentum.perp2()) >= ptMin_ && 
		      momentum.eta() >= minRapidity_ && momentum.eta() <= maxRapidity_ && 
		      sqrt(vertex.perp2()) <= tip_ &&
		      fabs(vertex.z()) <= lip_ 
		      );
	    }
	}
    }
    
    size_t size() const { return selected_.size(); }
    
 private:
    
    double ptMin_;
    double minRapidity_;
    double maxRapidity_;
    double tip_;
    double lip_;
    int    minHit_;
    bool chargedOnly_;
      std::vector<int> pdgId_;
      container selected_;

    mutable edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;      
      
};

#endif
