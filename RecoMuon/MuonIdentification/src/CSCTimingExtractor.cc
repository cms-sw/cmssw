// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      CSCTimingExtractor
// 
/**\class CSCTimingExtractor CSCTimingExtractor.cc RecoMuon/MuonIdentification/src/CSCTimingExtractor.cc
 *
 * Description: <one line class summary>
 *
 */
//
// Original Author:  Traczyk Piotr
//         Created:  Thu Oct 11 15:01:28 CEST 2007
//
//

#include "RecoMuon/MuonIdentification/interface/CSCTimingExtractor.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"


// system include files
#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace edm {
  class ParameterSet;
  class EventSetup;
  class InputTag;
}

class MuonServiceProxy;

//
// constructors and destructor
//
CSCTimingExtractor::CSCTimingExtractor(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC)
  :
  thePruneCut_(iConfig.getParameter<double>("PruneCut")),
  theStripTimeOffset_(iConfig.getParameter<double>("CSCStripTimeOffset")),
  theWireTimeOffset_(iConfig.getParameter<double>("CSCWireTimeOffset")),
  theStripError_(iConfig.getParameter<double>("CSCStripError")),
  theWireError_(iConfig.getParameter<double>("CSCWireError")),
  UseWireTime(iConfig.getParameter<bool>("UseWireTime")),
  UseStripTime(iConfig.getParameter<bool>("UseStripTime")),
  debug(iConfig.getParameter<bool>("debug"))
{
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);
  
  edm::ParameterSet matchParameters = iConfig.getParameter<edm::ParameterSet>("MatchParameters");

  theMatcher = new MuonSegmentMatcher(matchParameters, iC);
}


CSCTimingExtractor::~CSCTimingExtractor()
{
  if (theService) delete theService;
  if (theMatcher) delete theMatcher;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CSCTimingExtractor::fillTiming(TimeMeasurementSequence &tmSequence, reco::TrackRef muonTrack, const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  if (debug) 
    std::cout << " *** CSC Timimng Extractor ***" << std::endl;

  theService->update(iSetup);

  const GlobalTrackingGeometry *theTrackingGeometry = &*theService->trackingGeometry();
  
  edm::ESHandle<Propagator> propagator;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
  const Propagator *propag = propagator.product();

  double invbeta=0;
  double invbetaerr=0;
  double totalWeightInvbeta=0;
  double totalWeightVertex=0;
  std::vector<TimeMeasurement> tms;

  math::XYZPoint  pos=muonTrack->innerPosition();
  math::XYZVector mom=muonTrack->innerMomentum();

  if (sqrt(muonTrack->innerPosition().mag2()) > sqrt(muonTrack->outerPosition().mag2())){
     pos=muonTrack->outerPosition();
     mom=-1*muonTrack->outerMomentum();
  }

  GlobalPoint  posp(pos.x(), pos.y(), pos.z());
  GlobalVector momv(mom.x(), mom.y(), mom.z());
  FreeTrajectoryState muonFTS(posp, momv, (TrackCharge)muonTrack->charge(), theService->magneticField().product());

  // get the CSC segments that were used to construct the muon
  std::vector<const CSCSegment*> range = theMatcher->matchCSC(*muonTrack,iEvent);

  // create a collection on TimeMeasurements for the track        
  for (std::vector<const CSCSegment*>::iterator rechit = range.begin(); rechit!=range.end();++rechit) {

    // Create the ChamberId
    DetId id = (*rechit)->geographicalId();
    CSCDetId chamberId(id.rawId());
    //    int station = chamberId.station();

    if (!(*rechit)->specificRecHits().size()) continue;

    const std::vector<CSCRecHit2D> hits2d = (*rechit)->specificRecHits();

    // store all the hits from the segment
    for (std::vector<CSCRecHit2D>::const_iterator hiti=hits2d.begin(); hiti!=hits2d.end(); hiti++) {

      const GeomDet* cscDet = theTrackingGeometry->idToDet(hiti->geographicalId());
      TimeMeasurement thisHit;

      std::pair< TrajectoryStateOnSurface, double> tsos;
      tsos=propag->propagateWithPath(muonFTS,cscDet->surface());

      double dist;            
      if (tsos.first.isValid()) dist = tsos.second+posp.mag(); 
        else dist = cscDet->toGlobal(hiti->localPosition()).mag();

      thisHit.distIP = dist;
      if (UseStripTime) {
        thisHit.weightInvbeta = dist*dist/(theStripError_*theStripError_*30.*30.);
        thisHit.weightVertex = 1./(theStripError_*theStripError_);
        thisHit.timeCorr = hiti->tpeak()-theStripTimeOffset_;
        tms.push_back(thisHit);
      }

      if (UseWireTime) {
	thisHit.weightInvbeta = dist*dist/(theWireError_*theWireError_*30.*30.);
        thisHit.weightVertex = 1./(theWireError_*theWireError_);
        thisHit.timeCorr = hiti->wireTime()-theWireTimeOffset_;
        tms.push_back(thisHit);
      }

      
//      std::cout << " CSC Hit. Dist= " << dist << "    Time= " << thisHit.timeCorr 
//           << "   invBeta= " << (1.+thisHit.timeCorr/dist*30.) << std::endl;
    }

  } // rechit
      
  bool modified = false;
  std::vector <double> dstnc, dsegm, dtraj, hitWeightInvbeta, hitWeightVertex;

  // Now loop over the measurements, calculate 1/beta and cut away outliers
  do {    

    modified = false;
    dstnc.clear();
    dsegm.clear();
    dtraj.clear();
    hitWeightInvbeta.clear();
    hitWeightVertex.clear();
      
    totalWeightInvbeta=0;
    totalWeightVertex=0;
      
	for (std::vector<TimeMeasurement>::iterator tm=tms.begin(); tm!=tms.end(); ++tm) {
	  dstnc.push_back(tm->distIP);
	  dsegm.push_back(tm->timeCorr);
	  hitWeightInvbeta.push_back(tm->weightInvbeta);
          hitWeightVertex.push_back(tm->weightVertex);
	  totalWeightInvbeta+=tm->weightInvbeta;
	  totalWeightVertex+=tm->weightVertex;
	}
          
    if (totalWeightInvbeta==0) break;        

    // calculate the value and error of 1/beta from the complete set of 1D hits
    if (debug)
      std::cout << " Points for global fit: " << dstnc.size() << std::endl;

    // inverse beta - weighted average of the contributions from individual hits
    invbeta=0;
    for (unsigned int i=0;i<dstnc.size();i++) 
      invbeta+=(1.+dsegm.at(i)/dstnc.at(i)*30.)*hitWeightInvbeta.at(i)/totalWeightInvbeta;

    double chimax=0.;
    std::vector<TimeMeasurement>::iterator tmmax;
    
    // the dispersion of inverse beta
    double diff;
    for (unsigned int i=0;i<dstnc.size();i++) {
      diff=(1.+dsegm.at(i)/dstnc.at(i)*30.)-invbeta;
      diff=diff*diff*hitWeightInvbeta.at(i);
      invbetaerr+=diff;
      if (diff>chimax) { 
	tmmax=tms.begin()+i;
	chimax=diff;
      }
    }
    
    invbetaerr=sqrt(invbetaerr/totalWeightInvbeta); 
 
    // cut away the outliers
    if (chimax>thePruneCut_) {
      tms.erase(tmmax);
      modified=true;
    }    

    if (debug)
      std::cout << " Measured 1/beta: " << invbeta << " +/- " << invbetaerr << std::endl;

  } while (modified);

  // std::cout << " *** FINAL Measured 1/beta: " << invbeta << " +/- " << invbetaerr << std::endl;

  for (unsigned int i=0;i<dstnc.size();i++) {
    tmSequence.dstnc.push_back(dstnc.at(i));
    tmSequence.local_t0.push_back(dsegm.at(i));
    tmSequence.weightInvbeta.push_back(hitWeightInvbeta.at(i));
    tmSequence.weightVertex.push_back(hitWeightVertex.at(i));
  }

  tmSequence.totalWeightInvbeta=totalWeightInvbeta;
  tmSequence.totalWeightVertex=totalWeightVertex;

}

//define this as a plug-in
//DEFINE_FWK_MODULE(CSCTimingExtractor);
