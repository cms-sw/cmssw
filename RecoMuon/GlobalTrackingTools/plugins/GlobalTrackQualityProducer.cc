// -*- C++ -*-
//
// Package:    GlobalTrackingTools
// Class:      GlobalTrackQualityProducer
//
//
// Original Author:  Adam Everett
// $Id: $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"


#include "RecoMuon/GlobalTrackingTools/plugins/GlobalTrackQualityProducer.h"
//#include "FWCore/Framework/interface/MakerMacros.h"

#include <TrackingTools/PatternTools/interface/TrajectoryMeasurement.h>
#include <TrackingTools/PatternTools/interface/Trajectory.h>

GlobalTrackQualityProducer::GlobalTrackQualityProducer(const edm::ParameterSet& iConfig):
  inputCollection_(iConfig.getParameter<edm::InputTag>("InputCollection")),baseLabel_(iConfig.getParameter<std::string>("BaseLabel")),theService(0),theGlbRefitter(0)
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
 
  produces<edm::ValueMap<float> >("muqual"+baseLabel_+"GlbKink");
  produces<edm::ValueMap<float> >("muqual"+baseLabel_+"TkKink");
  produces<edm::ValueMap<float> >("muqual"+baseLabel_+"StaRelChi2");
  produces<edm::ValueMap<float> >("muqual"+baseLabel_+"TkRelChi2");
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
  std::vector<float> valuesGlbKink;
  valuesGlbKink.reserve(glbMuons->size());
  std::vector<float> valuesTkKink;
  valuesTkKink.reserve(glbMuons->size());
  std::vector<float> valuesStaRelChi2;
  valuesStaRelChi2.reserve(glbMuons->size());
  std::vector<float> valuesTkRelChi2;
  valuesTkRelChi2.reserve(glbMuons->size());

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
      relative_muon_chi2 = chi.second; //chi.second/staTrack->ndof();
      relative_tracker_chi2 = chi.first; // chi.first/tkTrack->ndof();
    }
    LogDebug(theCategory)<<"Kink " << thisKink.first << " " << thisKink.second;
    LogDebug(theCategory)<<"Rel Chi2 " << relative_tracker_chi2 << " " << relative_muon_chi2;
    valuesTkKink.push_back(thisKink.first);
    valuesGlbKink.push_back(thisKink.second);
    valuesTkRelChi2.push_back(relative_tracker_chi2);
    valuesStaRelChi2.push_back(relative_muon_chi2);
  }

  /*
  for(int i = 0; i < valuesTkRelChi2.size(); i++) {
    LogTrace(theCategory)<<"value " << valuesTkRelChi2[i] ;
  }
  */

  // create and fill value maps
  std::auto_ptr<edm::ValueMap<float> > outTkKink(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerTkKink(*outTkKink);
  fillerTkKink.insert(glbMuons, valuesTkKink.begin(), valuesTkKink.end());
  fillerTkKink.fill();
  
  std::auto_ptr<edm::ValueMap<float> > outGlbKink(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerGlbKink(*outGlbKink);
  fillerGlbKink.insert(glbMuons, valuesGlbKink.begin(), valuesGlbKink.end());
  fillerGlbKink.fill();

  std::auto_ptr<edm::ValueMap<float> > outTkRelChi2(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerTkRelChi2(*outTkRelChi2);
  fillerTkRelChi2.insert(glbMuons, valuesTkRelChi2.begin(), valuesTkRelChi2.end());
  fillerTkRelChi2.fill();

  std::auto_ptr<edm::ValueMap<float> > outStaRelChi2(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerStaRelChi2(*outStaRelChi2);
  fillerStaRelChi2.insert(glbMuons, valuesStaRelChi2.begin(), valuesStaRelChi2.end());
  fillerStaRelChi2.fill();
  
  // put value map into event
  iEvent.put(outTkKink,"muqual"+baseLabel_+"TkKink");
  iEvent.put(outGlbKink,"muqual"+baseLabel_+"GlbKink");
  iEvent.put(outTkRelChi2,"muqual"+baseLabel_+"TkRelChi2");
  iEvent.put(outStaRelChi2,"muqual"+baseLabel_+"StaRelChi2");
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

    double estimate = 0.0;

    RecHit rhit = (*m).recHit();
    bool ok = false;
    if ( rhit->isValid() ) {
      if(DetId::Tracker == rhit->geographicalId().det()) ok = true;
    }

    //if ( !ok ) continue;
    
    const TrajectoryStateOnSurface& tsos = (*m).predictedState();


    if ( tsos.isValid() ) {

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
    }
    if ( hit->isValid() &&  (hit->geographicalId().det()) == DetId::Muon ) {
      muChi2 += estimate;
      staDiff = estimate - m->estimate();
    }
  }
  

  return std::pair<double,double>(tkChi2,muChi2);
       
}

//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_FWK_MODULE(GlobalTrackQualityProducer);
