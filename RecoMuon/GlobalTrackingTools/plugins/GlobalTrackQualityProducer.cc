// -*- C++ -*-
//
// Package:    GlobalTrackingTools
// Class:      GlobalTrackQualityProducer
//
//
// Original Author:  Adam Everett
// $Id: GlobalTrackQualityProducer.cc,v 1.8 2012/12/06 14:45:38 eulisse Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "RecoMuon/GlobalTrackingTools/plugins/GlobalTrackQualityProducer.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/MuonReco/interface/MuonQuality.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

GlobalTrackQualityProducer::GlobalTrackQualityProducer(const edm::ParameterSet& iConfig):
  inputCollection_(iConfig.getParameter<edm::InputTag>("InputCollection")),inputLinksCollection_(iConfig.getParameter<edm::InputTag>("InputLinksCollection")),theService(0),theGlbRefitter(0),theGlbMatcher(0)
{
  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);     
  
  // TrackRefitter parameters
  edm::ParameterSet refitterParameters = iConfig.getParameter<edm::ParameterSet>("RefitterParameters");
  theGlbRefitter = new GlobalMuonRefitter(refitterParameters, theService);

  edm::ParameterSet trackMatcherPSet = iConfig.getParameter<edm::ParameterSet>("GlobalMuonTrackMatcher");
  theGlbMatcher = new GlobalMuonTrackMatcher(trackMatcherPSet,theService);

  double maxChi2 = iConfig.getParameter<double>("MaxChi2");
  double nSigma = iConfig.getParameter<double>("nSigma");
  theEstimator = new Chi2MeasurementEstimator(maxChi2,nSigma);
 
  produces<edm::ValueMap<reco::MuonQuality> >();
}

GlobalTrackQualityProducer::~GlobalTrackQualityProducer() {
  if (theService) delete theService;
  if (theGlbRefitter) delete theGlbRefitter;
  if (theGlbMatcher) delete theGlbMatcher;
  if (theEstimator) delete theEstimator;
}

void
GlobalTrackQualityProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const std::string theCategory = "Muon|RecoMuon|GlobalTrackQualityProducer";
  
  theService->update(iSetup);

  theGlbRefitter->setEvent(iEvent);

  theGlbRefitter->setServices(theService->eventSetup());

  // Take the GLB muon container(s)
  edm::Handle<reco::TrackCollection> glbMuons;
  iEvent.getByLabel(inputCollection_,glbMuons);
  
  edm::Handle<reco::MuonTrackLinksCollection>    linkCollectionHandle;
  iEvent.getByLabel(inputLinksCollection_, linkCollectionHandle);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  iSetup.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();


  // reserve some space
  std::vector<reco::MuonQuality> valuesQual;
  valuesQual.reserve(glbMuons->size());
  
  int trackIndex = 0;
  for (reco::TrackCollection::const_iterator track = glbMuons->begin(); track!=glbMuons->end(); ++track , ++trackIndex) {
    reco::TrackRef glbRef(glbMuons,trackIndex);
    reco::TrackRef staTrack = reco::TrackRef();

    std::vector<Trajectory> refitted=theGlbRefitter->refit(*track,1,tTopo);

    LogTrace(theCategory)<<"GLBQual N refitted " << refitted.size();
    
    std::pair<double,double> thisKink;
    double relative_muon_chi2 = 0.0;
    double relative_tracker_chi2 = 0.0;
    double glbTrackProbability = 0.0;
    if(refitted.size()>0) {
      thisKink = kink(refitted.front()) ;      
      std::pair<double,double> chi = newChi2(refitted.front());
      relative_muon_chi2 = chi.second; //normalized inside to /sum(muHits.dimension)
      relative_tracker_chi2 = chi.first; //normalized inside to /sum(tkHits.dimension)
      glbTrackProbability = trackProbability(refitted.front());
    }

    LogTrace(theCategory)<<"GLBQual: Kink " << thisKink.first << " " << thisKink.second;
    LogTrace(theCategory)<<"GLBQual: Rel Chi2 " << relative_tracker_chi2 << " " << relative_muon_chi2;
    LogTrace(theCategory)<<"GLBQual: trackProbability " << glbTrackProbability;

    // Fill the STA-TK match information
    float chi2, d, dist, Rpos;
    chi2 = d = dist = Rpos = -1.0;
    bool passTight = false;
    typedef MuonTrajectoryBuilder::TrackCand TrackCand;
    if ( linkCollectionHandle.isValid() ) {
      for ( reco::MuonTrackLinksCollection::const_iterator links = linkCollectionHandle->begin();
	    links != linkCollectionHandle->end(); ++links )
	{
	  if ( links->trackerTrack().isNull() ||
	       links->standAloneTrack().isNull() ||
	       links->globalTrack().isNull() ) 
	    {
	      edm::LogWarning(theCategory) << "Global muon links to constituent tracks are invalid. There should be no such object. Muon is skipped.";
	      continue;
	    }
	  if (links->globalTrack() == glbRef) {
	    staTrack = !links->standAloneTrack().isNull() ? links->standAloneTrack() : reco::TrackRef();
	    TrackCand staCand = TrackCand((Trajectory*)(0),links->standAloneTrack());
	    TrackCand tkCand = TrackCand((Trajectory*)(0),links->trackerTrack());
	    chi2 = theGlbMatcher->match(staCand,tkCand,0,0);
	    d    = theGlbMatcher->match(staCand,tkCand,1,0);
	    Rpos = theGlbMatcher->match(staCand,tkCand,2,0);
	    dist = theGlbMatcher->match(staCand,tkCand,3,0);
	    passTight = theGlbMatcher->matchTight(staCand,tkCand);
	  }
	}
    }

    if(!staTrack.isNull()) LogTrace(theCategory)<<"GLBQual: Used UpdatedAtVtx : " <<  (iEvent.getProvenance(staTrack.id()).productInstanceName() == std::string("UpdatedAtVtx"));

    float maxFloat01 = std::numeric_limits<float>::max()*0.1; // a better solution would be to use float above .. m/be not
    reco::MuonQuality muQual;
    if(!staTrack.isNull()) muQual.updatedSta = iEvent.getProvenance(staTrack.id()).productInstanceName() == std::string("UpdatedAtVtx");
    muQual.trkKink    = thisKink.first > maxFloat01 ? maxFloat01 : thisKink.first;
    muQual.glbKink    = thisKink.second > maxFloat01 ? maxFloat01 : thisKink.second;
    muQual.trkRelChi2 = relative_tracker_chi2 > maxFloat01 ? maxFloat01 : relative_tracker_chi2;
    muQual.staRelChi2 = relative_muon_chi2 > maxFloat01 ? maxFloat01 : relative_muon_chi2;
    muQual.tightMatch = passTight;
    muQual.chi2LocalPosition = dist;
    muQual.chi2LocalMomentum = chi2;
    muQual.localDistance = d;
    muQual.globalDeltaEtaPhi = Rpos;
    muQual.glbTrackProbability = glbTrackProbability;
    valuesQual.push_back(muQual);
  }

  /*
  for(int i = 0; i < valuesTkRelChi2.size(); i++) {
    LogTrace(theCategory)<<"value " << valuesTkRelChi2[i] ;
  }
  */

  // create and fill value maps
  std::auto_ptr<edm::ValueMap<reco::MuonQuality> > outQual(new edm::ValueMap<reco::MuonQuality>());
  edm::ValueMap<reco::MuonQuality>::Filler fillerQual(*outQual);
  fillerQual.insert(glbMuons, valuesQual.begin(), valuesQual.end());
  fillerQual.fill();
  
  // put value map into event
  iEvent.put(outQual);
}

std::pair<double,double> GlobalTrackQualityProducer::kink(Trajectory& muon) const {

  const std::string theCategory = "Muon|RecoMuon|GlobalTrackQualityProducer";
   
  using namespace std;
  using namespace edm;
  using namespace reco;
 
  double result = 0.0;
  double resultGlb = 0.0;

     
  typedef TransientTrackingRecHit::ConstRecHitPointer 	ConstRecHitPointer;
  typedef ConstRecHitPointer RecHit;
  typedef std::vector<TrajectoryMeasurement>::const_iterator TMI;


  vector<TrajectoryMeasurement> meas = muon.measurements();
  
  for ( TMI m = meas.begin(); m != meas.end(); m++ ) {
    TransientTrackingRecHit::ConstRecHitPointer hit = m->recHit();

    //not used    double estimate = 0.0;

    RecHit rhit = (*m).recHit();
    bool ok = false;
    if ( rhit->isValid() ) {
      if(DetId::Tracker == rhit->geographicalId().det()) ok = true;
    }

    //if ( !ok ) continue;
    
    const TrajectoryStateOnSurface& tsos = (*m).predictedState();


    if ( tsos.isValid() && rhit->isValid() && rhit->hit()->isValid()
	 && !edm::isNotFinite(rhit->localPositionError().xx()) //this is paranoia induced by reported case
	 && !edm::isNotFinite(rhit->localPositionError().xy()) //it's better to track down the origin of bad numbers
	 && !edm::isNotFinite(rhit->localPositionError().yy())
	 ) {

      double phi1 = tsos.globalPosition().phi();
      if ( phi1 < 0 ) phi1 = 2*M_PI + phi1;

      double phi2 = rhit->globalPosition().phi();
      if ( phi2 < 0 ) phi2 = 2*M_PI + phi2;

      double diff = fabs(phi1 - phi2);
      if ( diff > M_PI ) diff = 2*M_PI - diff;

      GlobalPoint hitPos = rhit->globalPosition();

      GlobalError hitErr = rhit->globalPositionError();
      //LogDebug(theCategory)<<"hitPos " << hitPos;
      double error = hitErr.phierr(hitPos);  // error squared

      double s = ( error > 0.0 ) ? (diff*diff)/error : (diff*diff);

      if(ok) result += s;
      resultGlb += s;
    }
    
  }
  
  
  return std::pair<double,double>(result,resultGlb);
  
}

std::pair<double,double> GlobalTrackQualityProducer::newChi2(Trajectory& muon) const {
  const std::string theCategory = "Muon|RecoMuon|GlobalTrackQualityProducer";

  using namespace std;
  using namespace edm;
  using namespace reco;

  double muChi2 = 0.0;
  double tkChi2 = 0.0;
  unsigned int muNdof = 0;
  unsigned int tkNdof = 0;
  
  typedef TransientTrackingRecHit::ConstRecHitPointer 	ConstRecHitPointer;
  typedef ConstRecHitPointer RecHit;
  typedef vector<TrajectoryMeasurement>::const_iterator TMI;

  vector<TrajectoryMeasurement> meas = muon.measurements();

  for ( TMI m = meas.begin(); m != meas.end(); m++ ) {
    TransientTrackingRecHit::ConstRecHitPointer hit = m->recHit();
    const TrajectoryStateOnSurface& uptsos = (*m).updatedState();
    TransientTrackingRecHit::RecHitPointer preciseHit = hit->clone(uptsos);
    double estimate = 0.0;
    if (preciseHit->isValid() && uptsos.isValid()) {
      estimate = theEstimator->estimate(uptsos, *preciseHit ).second;
    }
    
    //LogTrace(theCategory) << "estimate " << estimate << " TM.est " << m->estimate();
    //UNUSED:    double tkDiff = 0.0;
    //UNUSED:    double staDiff = 0.0;
    if ( hit->isValid() &&  (hit->geographicalId().det()) == DetId::Tracker ) {
      tkChi2 += estimate;
      //UNUSED:      tkDiff = estimate - m->estimate();
      tkNdof += hit->dimension();
    }
    if ( hit->isValid() &&  (hit->geographicalId().det()) == DetId::Muon ) {
      muChi2 += estimate;
      //UNUSED      staDiff = estimate - m->estimate();
      muNdof += hit->dimension();
    }
  }
  
  if (tkNdof < 6 ) tkChi2 = tkChi2; // or should I set it to  a large number ?
  else tkChi2 /= (tkNdof-5.);

  if (muNdof < 6 ) muChi2 = muChi2; // or should I set it to  a large number ?
  else muChi2 /= (muNdof-5.);

  return std::pair<double,double>(tkChi2,muChi2);
       
}

//
// calculate the tail probability (-ln(P)) of a fit
//
double 
GlobalTrackQualityProducer::trackProbability(Trajectory& track) const {

  if ( track.ndof() > 0 && track.chiSquared() > 0 ) { 
    return -LnChiSquaredProbability(track.chiSquared(), track.ndof());
  } else {
    return 0.0;
  }

}

//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_FWK_MODULE(GlobalTrackQualityProducer);
