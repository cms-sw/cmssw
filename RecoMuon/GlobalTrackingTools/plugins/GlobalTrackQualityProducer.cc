// -*- C++ -*-
//
// Package:    GlobalTrackingTools
// Class:      GlobalTrackQualityProducer
//
//
// Original Author:  Adam Everett
// $Id: GlobalTrackQualityProducer.cc,v 1.2 2009/09/12 20:33:33 aeverett Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"


#include "RecoMuon/GlobalTrackingTools/plugins/GlobalTrackQualityProducer.h"
//#include "FWCore/Framework/interface/MakerMacros.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/MuonReco/interface/MuonQuality.h"


GlobalTrackQualityProducer::GlobalTrackQualityProducer(const edm::ParameterSet& iConfig):
  inputCollection_(iConfig.getParameter<edm::InputTag>("InputCollection")),theService(0),theGlbRefitter(0)
{
  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);     
  
  // TrackRefitter parameters
  edm::ParameterSet refitterParameters = iConfig.getParameter<edm::ParameterSet>("RefitterParameters");
  theGlbRefitter = new GlobalMuonRefitter(refitterParameters, theService);

  double maxChi2 = iConfig.getParameter<double>("MaxChi2");
  double nSigma = iConfig.getParameter<double>("nSigma");
  theEstimator = new Chi2MeasurementEstimator(maxChi2,nSigma);
 
  produces<edm::ValueMap<reco::MuonQuality> >();
}

GlobalTrackQualityProducer::~GlobalTrackQualityProducer() {
  if (theService) delete theService;
  if (theGlbRefitter) delete theGlbRefitter;
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
  
  // reserve some space
  std::vector<reco::MuonQuality> valuesQual;
  valuesQual.reserve(glbMuons->size());

  int trackIndex = 0;
  for (reco::TrackCollection::const_iterator track = glbMuons->begin(); track!=glbMuons->end(); track++ , ++trackIndex) {
    reco::TrackRef glbRef(glbMuons,trackIndex);
    
    std::vector<Trajectory> refitted=theGlbRefitter->refit(*track,1);

    LogDebug(theCategory)<<"N refitted " << refitted.size();
    
    std::pair<double,double> thisKink;
    double relative_muon_chi2 = 0.0;
    double relative_tracker_chi2 = 0.0;
    
    if(refitted.size()>0) {
      thisKink = kink(refitted.front()) ;      
      std::pair<double,double> chi = newChi2(refitted.front());
      relative_muon_chi2 = chi.second; //normalized inside to /sum(muHits.dimension)
      relative_tracker_chi2 = chi.first; //normalized inside to /sum(tkHits.dimension)
    }
    LogDebug(theCategory)<<"Kink " << thisKink.first << " " << thisKink.second;
    LogDebug(theCategory)<<"Rel Chi2 " << relative_tracker_chi2 << " " << relative_muon_chi2;
    reco::MuonQuality muQual;
    muQual.trkKink    = thisKink.first;
    muQual.glbKink    = thisKink.second;
    muQual.trkRelChi2 = relative_tracker_chi2;
    muQual.staRelChi2 = relative_muon_chi2;
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
	 && !std::isinf(rhit->localPositionError().xx()) //this is paranoia induced by reported case
	 && !std::isinf(rhit->localPositionError().xy()) //it's better to track down the origin of bad numbers
	 && !std::isinf(rhit->localPositionError().yy())
	 && !std::isnan(rhit->localPositionError().xx()) //this is paranoia induced by reported case
	 && !std::isnan(rhit->localPositionError().xy()) //it's better to track down the origin of bad numbers
	 && !std::isnan(rhit->localPositionError().yy())
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
    double tkDiff = 0.0;
    double staDiff = 0.0;
    if ( hit->isValid() &&  (hit->geographicalId().det()) == DetId::Tracker ) {
      tkChi2 += estimate;
      tkDiff = estimate - m->estimate();
      tkNdof += hit->dimension();
    }
    if ( hit->isValid() &&  (hit->geographicalId().det()) == DetId::Muon ) {
      muChi2 += estimate;
      staDiff = estimate - m->estimate();
      muNdof += hit->dimension();
    }
  }
  
  if (tkNdof < 6 ) tkChi2 = tkChi2; // or should I set it to  a large number ?
  else tkChi2 /= (tkNdof-5.);

  if (muNdof < 6 ) muChi2 = muChi2; // or should I set it to  a large number ?
  else muChi2 /= (muNdof-5.);

  return std::pair<double,double>(tkChi2,muChi2);
       
}

//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_FWK_MODULE(GlobalTrackQualityProducer);
