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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByPositionESProducer.hh"

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
				   const std::vector<int>& pdgId = std::vector<int>()) :
    ptMin_( ptMin ), minRapidity_( minRapidity ), maxRapidity_( maxRapidity ),
    tip_( tip ), lip_( lip ), minHit_( minHit ), chargedOnly_(chargedOnly), pdgId_( pdgId ) { }


    CosmicTrackingParticleSelector  ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
      ptMin_(cfg.getParameter<double>("ptMin")),
      minRapidity_(cfg.getParameter<double>("minRapidity")),
      maxRapidity_(cfg.getParameter<double>("maxRapidity")),
      tip_(cfg.getParameter<double>("tip")),
      lip_(cfg.getParameter<double>("lip")),
      minHit_(cfg.getParameter<int>("minHit")),
      chargedOnly_(cfg.getParameter<bool>("chargedOnly")),
      pdgId_(cfg.getParameter<std::vector<int> >("pdgId")),
      beamSpotToken_(iC.consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"))) { }


      void select( const edm::Handle<collection>& c, const edm::Event & event, const edm::EventSetup& setup) {
	selected_.clear();
	edm::Handle<reco::BeamSpot> beamSpot;
	event.getByToken(beamSpotToken_, beamSpot);
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
      edm::ESHandle<GlobalTrackingGeometry> theGeometry;
      iSetup.get<GlobalTrackingGeometryRecord>().get(theGeometry);

      edm::ESHandle<MagneticField> theMF;
      iSetup.get<IdealMagneticFieldRecord>().get(theMF);

      GlobalVector finalGV (0,0,0);
      GlobalPoint finalGP(0,0,0);
      GlobalVector momentum(0,0,0);//At the PCA
      GlobalPoint vertex(0,0,0);//At the PCA
      double radius(9999);
      bool found(0);

      int ii=0;
      DetId::Detector det;
      int subdet;

      edm::LogVerbatim("CosmicTrackingParticleSelector")
	<<"TOT Number of PSimHits = "<< tpr->numberOfHits() << ", Number of Tracker PSimHits = "<< tpr->numberOfTrackerHits() <<"\n";

      if (simHitsTPAssoc.isValid()==0) {
	edm::LogError("CosmicTrackingParticleSelector") << "Invalid handle!";
	return false;
      }
      std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(tpr,TrackPSimHitRef());//SimHit is dummy: for simHitTPAssociationListGreater
                                                                                                      // sorting only the cluster is needed
      auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(),
				    clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
      for(auto ip = range.first; ip != range.second; ++ip) {
	TrackPSimHitRef it = ip->second;
	++ii;
	const GeomDet* tmpDet  = theGeometry->idToDet( DetId(it->detUnitId()) ) ;
	if (!tmpDet) {
	  edm::LogVerbatim("CosmicTrackingParticleSelector")
	    <<"***WARNING:  PSimHit "<<ii <<", no GeomDet for: "<<it->detUnitId()<<". Skipping it.";
	  continue;
	} else {
	  det = DetId(it->detUnitId()).det();
	  subdet = DetId(it->detUnitId()).subdetId();
	}

	LocalVector  lv = it->momentumAtEntry();
	Local3DPoint lp = it->localPosition ();
	GlobalVector gv = tmpDet->surface().toGlobal( lv );
	GlobalPoint  gp = tmpDet->surface().toGlobal( lp );
	edm::LogVerbatim("CosmicTrackingParticleSelector")
	  <<"PSimHit "<<ii<<", Detector = "<<det<<", subdet = "<<subdet
	  <<"\t Radius = "<< gp.perp() << ", z = "<< gp.z() 
	  <<"\t     pt = "<< gv.perp() << ", pz = "<< gv.z();
	edm::LogVerbatim("CosmicTrackingParticleSelector")
	  <<"\t trackId = "<<it->trackId()<<", particleType = "<<it->particleType()<<", processType = "<<it->processType();

	// discard hits related to low energy debris from the primary particle
	if (it->processType()!=0) continue;

	if(gp.perp()<radius){
	  found=true;
	  radius = gp.perp();
	  finalGV = gv;
	  finalGP = gp;
	}
      }
      edm::LogVerbatim("CosmicTrackingParticleSelector")
	<<"\n"<<"FINAL State at InnerMost Hit:        Radius = "<< finalGP.perp() << ", z = "<< finalGP.z() 
	<<", pt = "<< finalGV.perp() << ", pz = "<< finalGV.z();

      if(!found) return 0;
      else
	{
	  FreeTrajectoryState ftsAtProduction(finalGP,finalGV,TrackCharge(tpr->charge()),theMF.product());
	  TSCBLBuilderNoMaterial tscblBuilder;
	  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,*bs);//as in TrackProducerAlgorithm
	  if(!tsAtClosestApproach.isValid()){
	    edm::LogVerbatim("CosmicTrackingParticleSelector")
	      <<"*** WARNING in CosmicTrackingParticleSelector: tsAtClosestApproach is not valid." <<"\n";
	    return 0;
	  }
	  else
	    {
	      momentum = tsAtClosestApproach.trackStateAtPCA().momentum();
	      vertex = tsAtClosestApproach.trackStateAtPCA().position();

	      edm::LogVerbatim("CosmicTrackingParticleSelector")
		<<"FINAL State extrapolated at PCA: Radius = "<< vertex.perp() << ", z = "<< vertex.z() 
		<<", pt = "<< momentum.perp() << ", pz = "<< momentum.z() <<"\n";

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
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

    mutable edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;

};

#endif
