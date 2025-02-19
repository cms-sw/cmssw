// -*- C++ -*-
//
// Package:    TrackIPProducer
// Class:      TrackIPProducer
// 
/**\class TrackIPProducer TrackIPProducer.cc RecoBTau/TrackIPProducer/src/TrackIPProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
// $Id: TrackIPProducer.cc,v 1.24 2009/11/24 10:30:50 saout Exp $
//
//

// system include files
#include <cmath>
#include <memory>
#include <iostream>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"

#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"
#include "RecoBTag/ImpactParameter/plugins/TrackIPProducer.h"


using namespace std;
using namespace reco;
using namespace edm;
using boost::bind;

//
// constructors and destructor
//
TrackIPProducer::TrackIPProducer(const edm::ParameterSet& iConfig) : 
  m_config(iConfig)
{
  m_calibrationCacheId3D = 0;
  m_calibrationCacheId2D = 0;

  m_associator              = m_config.getParameter<InputTag>("jetTracks");
  m_primaryVertexProducer   = m_config.getParameter<InputTag>("primaryVertex");
  m_computeProbabilities    = m_config.getParameter<bool>("computeProbabilities");
  m_computeGhostTrack       = m_config.getParameter<bool>("computeGhostTrack");
  m_ghostTrackPriorDeltaR   = m_config.getParameter<double>("ghostTrackPriorDeltaR");
  m_cutPixelHits            = m_config.getParameter<int>("minimumNumberOfPixelHits");
  m_cutTotalHits            = m_config.getParameter<int>("minimumNumberOfHits");
  m_cutMaxTIP               = m_config.getParameter<double>("maximumTransverseImpactParameter");
  m_cutMinPt                = m_config.getParameter<double>("minimumTransverseMomentum");
  m_cutMaxChiSquared        = m_config.getParameter<double>("maximumChiSquared");
  m_cutMaxLIP               = m_config.getParameter<double>("maximumLongitudinalImpactParameter");
  m_directionWithTracks     = m_config.getParameter<bool>("jetDirectionUsingTracks");
  m_directionWithGhostTrack = m_config.getParameter<bool>("jetDirectionUsingGhostTrack");
  m_useTrackQuality         = m_config.getParameter<bool>("useTrackQuality");

  if (m_computeGhostTrack)
    produces<reco::TrackCollection>("ghostTracks");
  produces<reco::TrackIPTagInfoCollection>();
}

TrackIPProducer::~TrackIPProducer()
{
}

//
// member functions
//
// ------------ method called to produce the data  ------------
void
TrackIPProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // Update probability estimator if event setup is changed
   if (m_computeProbabilities)
     checkEventSetup(iSetup);
 
   //input objects 
   Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
   iEvent.getByLabel(m_associator, jetTracksAssociation);
   
   Handle<reco::VertexCollection> primaryVertex;
   iEvent.getByLabel(m_primaryVertexProducer, primaryVertex);

   edm::ESHandle<TransientTrackBuilder> builder;
   iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
   // m_algo.setTransientTrackBuilder(builder.product());

   // output collections 
   auto_ptr<reco::TrackIPTagInfoCollection> result(new reco::TrackIPTagInfoCollection);

   auto_ptr<reco::TrackCollection> ghostTracks;
   TrackRefProd ghostTrackRefProd;
   if (m_computeGhostTrack) {
     ghostTracks.reset(new reco::TrackCollection);
     ghostTrackRefProd = iEvent.getRefBeforePut<TrackCollection>("ghostTracks");
   }

   // use first pv of the collection
   Vertex dummy;
   const Vertex *pv = &dummy;
   edm::Ref<VertexCollection> pvRef;
   if (primaryVertex->size() != 0) {
     pv = &*primaryVertex->begin();
     // we always use the first vertex (at the moment)
     pvRef = edm::Ref<VertexCollection>(primaryVertex, 0);
   } else { // create a dummy PV
     Vertex::Error e;
     e(0, 0) = 0.0015 * 0.0015;
     e(1, 1) = 0.0015 * 0.0015;
     e(2, 2) = 15. * 15.;
     Vertex::Point p(0, 0, 0);
     dummy = Vertex(p, e, 0, 0, 0);
   }

   int i = 0;
   for(JetTracksAssociationCollection::const_iterator it =
       					jetTracksAssociation->begin();
       it != jetTracksAssociation->end(); it++, i++) {
     TrackRefVector tracks = it->second;
     math::XYZVector jetMomentum = it->first->momentum();

     if (m_directionWithTracks) {
       jetMomentum *= 0.5;
       for(TrackRefVector::const_iterator itTrack = tracks.begin();
           itTrack != tracks.end(); ++itTrack)
         if ((**itTrack).numberOfValidHits() >= m_cutTotalHits)
           //minimal quality cuts
           jetMomentum += (*itTrack)->momentum();
     }

     TrackRefVector selectedTracks;
     vector<TransientTrack> transientTracks;

     for(TrackRefVector::const_iterator itTrack = tracks.begin();
         itTrack != tracks.end(); ++itTrack) {
       const Track & track = **itTrack;
       TransientTrack transientTrack = builder->build(*itTrack);
/*     cout << " pt " <<  track.pt() <<
               " d0 " <<  fabs(track.d0()) <<
               " #hit " <<    track.hitPattern().numberOfValidHits()<<
               " ipZ " <<   fabs(track.dz()-pvZ)<<
               " chi2 " <<  track.normalizedChi2()<<
               " #pixel " <<    track.hitPattern().numberOfValidPixelHits()<< endl;
*/
       if (track.pt() > m_cutMinPt &&
           track.hitPattern().numberOfValidHits() >= m_cutTotalHits &&         // min num tracker hits
           track.hitPattern().numberOfValidPixelHits() >= m_cutPixelHits &&
           track.normalizedChi2() < m_cutMaxChiSquared &&
           std::abs(track.dxy(pv->position())) < m_cutMaxTIP &&
           std::abs(track.dz(pv->position())) < m_cutMaxLIP) {
         selectedTracks.push_back(*itTrack);
         transientTracks.push_back(transientTrack);
       }
     }

     GlobalVector direction(jetMomentum.x(), jetMomentum.y(), jetMomentum.z());

     auto_ptr<GhostTrack> ghostTrack;
     TrackRef ghostTrackRef;
     if (m_computeGhostTrack) {
       GhostTrackFitter fitter;
       GlobalPoint origin = RecoVertex::convertPos(pv->position());
       GlobalError error = RecoVertex::convertError(pv->error());
       ghostTrack.reset(new GhostTrack(fitter.fit(origin, error, direction,
                                                  m_ghostTrackPriorDeltaR,
                                                  transientTracks)));

/*
	if (std::sqrt(jetMomentum.Perp2()) > 30) {
		double offset = ghostTrack->prediction().lambda(origin);
		std::cout << "------------------ jet pt " << std::sqrt(jetMomentum.Perp2()) << std::endl;
		const std::vector<GhostTrackState> *states = &ghostTrack->states();
		for(std::vector<GhostTrackState>::const_iterator state = states->begin();
		    state != states->end(); ++state) {
			double dist = state->lambda() - offset;
			double err = state->lambdaError(ghostTrack->prediction(), error);
			double ipSig = IPTools::signedImpactParameter3D(state->track(), direction, *pv).second.significance();
			double axisDist = state->axisDistance(ghostTrack->prediction());
			std::cout << state->track().impactPointState().freeState()->momentum().perp()
			          << ": " << dist << "/" << err << " [" << (dist / err) << "], ipsig = " << ipSig << ", dist = " << axisDist << ", w = " << state->weight() << std::endl;
		}
	}
*/
       ghostTrackRef = TrackRef(ghostTrackRefProd, ghostTracks->size());
       ghostTracks->push_back(*ghostTrack);

       if (m_directionWithGhostTrack) { 
         const GhostTrackPrediction &pred = ghostTrack->prediction();
         double lambda = pred.lambda(origin);
         dummy = Vertex(RecoVertex::convertPos(pred.position(lambda)),
                        RecoVertex::convertError(pred.positionError(lambda)),
                        0, 0, 0);
         pv = &dummy;
         direction = pred.direction();
       }
     }

     vector<float> prob2D, prob3D;
     vector<TrackIPTagInfo::TrackIPData> ipData;

     for(unsigned int ind = 0; ind < transientTracks.size(); ind++) {
       const Track & track = *selectedTracks[ind];
       const TransientTrack &transientTrack = transientTracks[ind];

       TrackIPTagInfo::TrackIPData trackIP;
       trackIP.ip3d = IPTools::signedImpactParameter3D(transientTrack, direction, *pv).second;
       trackIP.ip2d = IPTools::signedTransverseImpactParameter(transientTrack, direction, *pv).second;

       TrajectoryStateOnSurface closest =
               IPTools::closestApproachToJet(transientTrack.impactPointState(),
                                             *pv, direction,
                                             transientTrack.field());
       if (closest.isValid())
         trackIP.closestToJetAxis = closest.globalPosition();

       // TODO: cross check if it is the same using other methods
       trackIP.distanceToJetAxis = IPTools::jetTrackDistance(transientTrack, direction, *pv).second;

       if (ghostTrack.get()) {
         const std::vector<GhostTrackState> &states = ghostTrack->states();
         std::vector<GhostTrackState>::const_iterator pos =
                std::find_if(states.begin(), states.end(),
                             bind(equal_to<TransientTrack>(),
                                  bind(&GhostTrackState::track, _1),
                                  transientTrack));

         if (pos != states.end() && pos->isValid()) {
           VertexDistance3D dist;
           const GhostTrackPrediction &pred = ghostTrack->prediction();
           GlobalPoint p1 = pos->tsos().globalPosition();
           GlobalError e1 = pos->tsos().cartesianError().position();
           GlobalPoint p2 = pred.position(pos->lambda());
           GlobalError e2 = pred.positionError(pos->lambda());
           trackIP.closestToGhostTrack = p1;
           trackIP.distanceToGhostTrack = dist.distance(VertexState(p1, e1),
                                                        VertexState(p2, e2));
           trackIP.ghostTrackWeight = pos->weight();
         } else {
           trackIP.distanceToGhostTrack = Measurement1D(-1. -1.);
           trackIP.ghostTrackWeight = 0.;
         }
       } else {
         trackIP.distanceToGhostTrack = Measurement1D(-1. -1.);
         trackIP.ghostTrackWeight = 1.;
       }

       ipData.push_back(trackIP);

       if (m_computeProbabilities) {
         //probability with 3D ip
         pair<bool,double> probability = m_probabilityEstimator->probability(m_useTrackQuality, 0,ipData.back().ip3d.significance(),track,*(it->first),*pv);
         prob3D.push_back(probability.first ? probability.second : -1.);

         //probability with 2D ip
         probability = m_probabilityEstimator->probability(m_useTrackQuality,1,ipData.back().ip2d.significance(),track,*(it->first),*pv);
         prob2D.push_back(probability.first ? probability.second : -1.);
       } 
     }

     Ref<JetTracksAssociationCollection> jtaRef(jetTracksAssociation, i);
     result->push_back(
             TrackIPTagInfo(ipData, prob2D, prob3D, selectedTracks,
                            jtaRef, pvRef, direction, ghostTrackRef));
   }
 
   if (m_computeGhostTrack)
     iEvent.put(ghostTracks, "ghostTracks");
   iEvent.put(result);
}


#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

void TrackIPProducer::checkEventSetup(const EventSetup & iSetup)
 {
  using namespace edm;
  using namespace edm::eventsetup;

   const EventSetupRecord & re2D= iSetup.get<BTagTrackProbability2DRcd>();
   const EventSetupRecord & re3D= iSetup.get<BTagTrackProbability3DRcd>();
   unsigned long long cacheId2D= re2D.cacheIdentifier();
   unsigned long long cacheId3D= re3D.cacheIdentifier();

   if(cacheId2D!=m_calibrationCacheId2D || cacheId3D!=m_calibrationCacheId3D  )  //Calibration changed
   {
     //iSetup.get<BTagTrackProbabilityRcd>().get(calib);
     ESHandle<TrackProbabilityCalibration> calib2DHandle;
     iSetup.get<BTagTrackProbability2DRcd>().get(calib2DHandle);
     ESHandle<TrackProbabilityCalibration> calib3DHandle;
     iSetup.get<BTagTrackProbability3DRcd>().get(calib3DHandle);

     const TrackProbabilityCalibration *  ca2D= calib2DHandle.product();
     const TrackProbabilityCalibration *  ca3D= calib3DHandle.product();

     m_probabilityEstimator.reset(new HistogramProbabilityEstimator(ca3D,ca2D));

   }
   m_calibrationCacheId3D=cacheId3D;
   m_calibrationCacheId2D=cacheId2D;
}
