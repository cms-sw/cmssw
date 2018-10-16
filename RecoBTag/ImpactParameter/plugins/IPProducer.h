#ifndef RecoBTag_IPProducer
#define RecoBTag_IPProducer

// system include files
#include <cmath>
#include <memory>
#include <iostream>
#include <algorithm>

#include "boost/bind.hpp"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/Candidate/interface/Candidate.h"

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
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class HistogramProbabilityEstimator;
using boost::bind;

namespace IPProducerHelpers {
      class FromJTA{
	      public:
		      FromJTA(const edm::ParameterSet& iConfig, edm::ConsumesCollector && iC) : token_associator(iC.consumes<reco::JetTracksAssociationCollection>(iConfig.getParameter<edm::InputTag>("jetTracks"))) 
			{}
		      reco::TrackRefVector tracks(const reco::JTATagInfo & it)
		      {
			      return it.tracks();
		      }
		      std::vector<reco::JTATagInfo>  makeBaseVector(const edm::Event& iEvent){
			      edm::Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
			      iEvent.getByToken(token_associator, jetTracksAssociation);
			      std::vector<reco::JTATagInfo> bases;
			      size_t i = 0;
			      for(reco::JetTracksAssociationCollection::const_iterator it =
					      jetTracksAssociation->begin();
					      it != jetTracksAssociation->end(); it++, i++) {
				      edm::Ref<reco::JetTracksAssociationCollection> jtaRef(jetTracksAssociation, i);
				      bases.push_back(reco::JTATagInfo(jtaRef));
			      }
			      return bases;
		      }

		      edm::EDGetTokenT<reco::JetTracksAssociationCollection> token_associator;
      };
      class FromJetAndCands{
              public:
		      FromJetAndCands(const edm::ParameterSet& iConfig,  edm::ConsumesCollector && iC, const std::string & jets = "jets"): token_jets(iC.consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>(jets))),
		      token_cands(iC.consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("candidates"))), maxDeltaR(iConfig.getParameter<double>("maxDeltaR")),
		      explicitJTA(iConfig.existsAs<bool>("explicitJTA") ? iConfig.getParameter<bool>("explicitJTA") : false) {}

                      const std::vector<reco::CandidatePtr> & tracks(const reco::JetTagInfo & it)
                      {
                              return m_map[it.jet().key()];
                      }
                      std::vector<reco::JetTagInfo>  makeBaseVector(const edm::Event& iEvent){
                              edm::Handle<edm::View<reco::Jet> > jets;
                              iEvent.getByToken(token_jets, jets);
                              std::vector<reco::JetTagInfo> bases;
			      
                              edm::Handle<edm::View<reco::Candidate> > cands;
                              iEvent.getByToken(token_cands, cands);
			      m_map.clear();
			      m_map.resize(jets->size());
			      double maxDeltaR2 = maxDeltaR*maxDeltaR;
                              size_t i = 0;
                              for(edm::View<reco::Jet>::const_iterator it = jets->begin();
                                              it != jets->end(); it++, i++) {
                                      edm::RefToBase<reco::Jet> jRef(jets, i);
                                      bases.push_back(reco::JetTagInfo(jRef));
				      if( explicitJTA )
				      {
					  for(size_t j=0;j<it->numberOfDaughters();++j) {
						  if( it->daughterPtr(j)->bestTrack()!=nullptr && it->daughterPtr(j)->charge() !=0 ){
							  m_map[i].push_back(it->daughterPtr(j));
						  }
					  }
				      }
				      else
				      {
					  for(size_t j=0;j<cands->size();++j) {
						  if( (*cands)[j].bestTrack()!=nullptr && (*cands)[j].charge() !=0 && (*cands)[j].pt() > 0 && Geom::deltaR2((*cands)[j],(*jets)[i]) < maxDeltaR2  ){
							  m_map[i].push_back(cands->ptrAt(j));
						  }
					  }
				      }
                              }
                              return bases;
                      }
		      std::vector<std::vector<reco::CandidatePtr> > m_map;	
                      edm::EDGetTokenT<edm::View<reco::Jet> > token_jets;
                      edm::EDGetTokenT<edm::View<reco::Candidate> >token_cands;
		      double maxDeltaR;
		      bool   explicitJTA;
      };
}
template <class Container, class Base, class Helper> 
class IPProducer : public edm::stream::EDProducer<> {
   public:
      typedef std::vector<reco::IPTagInfo<Container,Base> > Product;	
      

      explicit IPProducer(const edm::ParameterSet&);
      ~IPProducer() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      void produce(edm::Event&, const edm::EventSetup&) override;
   private:
    void  checkEventSetup(const edm::EventSetup & iSetup);

    const edm::ParameterSet& m_config;
    edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;

    bool m_computeProbabilities;
    bool m_computeGhostTrack;
    double m_ghostTrackPriorDeltaR;
    std::unique_ptr<HistogramProbabilityEstimator> m_probabilityEstimator;
    unsigned long long  m_calibrationCacheId2D; 
    unsigned long long  m_calibrationCacheId3D;
    bool m_useDB;

    int  m_cutPixelHits;
    int  m_cutTotalHits;
    double  m_cutMaxTIP;
    double  m_cutMinPt;
    double  m_cutMaxChiSquared;
    double  m_cutMaxLIP;
    bool  m_directionWithTracks;
    bool  m_directionWithGhostTrack;
    bool  m_useTrackQuality;
    Helper m_helper;
};

// -*- C++ -*-
//
// Package:    IPProducer
// Class:      IPProducer
// 
/**\class IPProducer IPProducer.cc RecoBTau/IPProducer/src/IPProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
//
//





//
// constructors and destructor
//
template <class Container, class Base, class Helper> IPProducer<Container,Base,Helper>::IPProducer(const edm::ParameterSet& iConfig) : 
  m_config(iConfig),m_helper(iConfig,consumesCollector())
{
  m_calibrationCacheId3D = 0;
  m_calibrationCacheId2D = 0;

  token_primaryVertex       = consumes<reco::VertexCollection>(m_config.getParameter<edm::InputTag>("primaryVertex"));

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
  produces<Product>();
}

template <class Container, class Base, class Helper> IPProducer<Container,Base,Helper>::~IPProducer()
{
}

//
// member functions
//
// ------------ method called to produce the data  ------------
template <class Container, class Base, class Helper> void
IPProducer<Container,Base,Helper>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // Update probability estimator if event setup is changed
   if (m_computeProbabilities)
     checkEventSetup(iSetup);
 
   
   edm::Handle<reco::VertexCollection> primaryVertex;
   iEvent.getByToken(token_primaryVertex, primaryVertex);

   edm::ESHandle<TransientTrackBuilder> builder;
   iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
   // m_algo.setTransientTrackBuilder(builder.product());

   // output collections 
   auto result = std::make_unique<Product>();

   std::unique_ptr<reco::TrackCollection> ghostTracks;
   reco::TrackRefProd ghostTrackRefProd;
   if (m_computeGhostTrack) {
     ghostTracks = std::make_unique<reco::TrackCollection>();
     ghostTrackRefProd = iEvent.getRefBeforePut<reco::TrackCollection>("ghostTracks");
   }

   // use first pv of the collection
   reco::Vertex dummy;
   const reco::Vertex *pv = &dummy;
   edm::Ref<reco::VertexCollection> pvRef;
   if (!primaryVertex->empty()) {
     pv = &*primaryVertex->begin();
     // we always use the first vertex (at the moment)
     pvRef = edm::Ref<reco::VertexCollection>(primaryVertex, 0);
   } else { // create a dummy PV
     reco::Vertex::Error e;
     e(0, 0) = 0.0015 * 0.0015;
     e(1, 1) = 0.0015 * 0.0015;
     e(2, 2) = 15. * 15.;
     reco::Vertex::Point p(0, 0, 0);
     dummy = reco::Vertex(p, e, 0, 0, 0);
   }

   std::vector<Base> baseTagInfos = m_helper.makeBaseVector(iEvent);
   for(typename std::vector<Base>::const_iterator it = baseTagInfos.begin();  it != baseTagInfos.end(); it++) {
     Container tracks = m_helper.tracks(*it);
     math::XYZVector jetMomentum = it->jet()->momentum();

     if (m_directionWithTracks) {
       jetMomentum *= 0.5;
       for(typename Container::const_iterator itTrack = tracks.begin();
           itTrack != tracks.end(); ++itTrack)
           if (reco::btag::toTrack(*itTrack)->numberOfValidHits() >= m_cutTotalHits)           //minimal quality cuts
	           jetMomentum += (*itTrack)->momentum();
     }

     Container selectedTracks;
     std::vector<reco::TransientTrack> transientTracks;

     for(typename Container::const_iterator itTrack = tracks.begin();
         itTrack != tracks.end(); ++itTrack) {
       reco::TransientTrack transientTrack = builder->build(*itTrack);
       const reco::Track & track = transientTrack.track(); //**itTrack;
 /*    cout << " pt " <<  track.pt() <<
               " d0 " <<  fabs(track.d0()) <<
               " #hit " <<    track.hitPattern().numberOfValidHits()<<
               " ipZ " <<   fabs(track.dz()-pv->z())<<
               " chi2 " <<  track.normalizedChi2()<<
               " #pixel " <<    track.hitPattern().numberOfValidPixelHits()<< endl;
*/
       if (track.pt() > m_cutMinPt &&
           track.hitPattern().numberOfValidHits() >= m_cutTotalHits &&         // min num tracker hits
           track.hitPattern().numberOfValidPixelHits() >= m_cutPixelHits &&
           track.normalizedChi2() < m_cutMaxChiSquared &&
           std::abs(track.dxy(pv->position())) < m_cutMaxTIP &&
           std::abs(track.dz(pv->position())) < m_cutMaxLIP) {
//	 std::cout << "selected" << std::endl; 	
         selectedTracks.push_back(*itTrack);
         transientTracks.push_back(transientTrack);
       }
     }
//	std::cout <<"SIZE: " << transientTracks.size() << std::endl;
     GlobalVector direction(jetMomentum.x(), jetMomentum.y(), jetMomentum.z());

     std::unique_ptr<reco::GhostTrack> ghostTrack;
     reco::TrackRef ghostTrackRef;
     if (m_computeGhostTrack) {
       reco::GhostTrackFitter fitter;
       GlobalPoint origin = RecoVertex::convertPos(pv->position());
       GlobalError error = RecoVertex::convertError(pv->error());
       ghostTrack.reset(new reco::GhostTrack(fitter.fit(origin, error, direction,
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
       ghostTrackRef = reco::TrackRef(ghostTrackRefProd, ghostTracks->size());
       ghostTracks->push_back(*ghostTrack);

       if (m_directionWithGhostTrack) { 
         const reco::GhostTrackPrediction &pred = ghostTrack->prediction();
         double lambda = pred.lambda(origin);
         dummy = reco::Vertex(RecoVertex::convertPos(pred.position(lambda)),
                        RecoVertex::convertError(pred.positionError(lambda)),
                        0, 0, 0);
         pv = &dummy;
         direction = pred.direction();
       }
     }

     std::vector<float> prob2D, prob3D;
     std::vector<reco::btag::TrackIPData> ipData;

     for(unsigned int ind = 0; ind < transientTracks.size(); ind++) {
       const reco::TransientTrack &transientTrack = transientTracks[ind];
       const reco::Track & track = transientTrack.track();

       reco::btag::TrackIPData trackIP;
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
         const std::vector<reco::GhostTrackState> &states = ghostTrack->states();
         std::vector<reco::GhostTrackState>::const_iterator pos =
                std::find_if(states.begin(), states.end(),
                             bind(std::equal_to<reco::TransientTrack>(),
                                  bind(&reco::GhostTrackState::track, _1),
                                  transientTrack));

         if (pos != states.end() && pos->isValid()) {
           VertexDistance3D dist;
           const reco::GhostTrackPrediction &pred = ghostTrack->prediction();
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
         std::pair<bool,double> probability = m_probabilityEstimator->probability(m_useTrackQuality, 0,ipData.back().ip3d.significance(),track,*(it->jet()),*pv);
         prob3D.push_back(probability.first ? probability.second : -1.);

         //probability with 2D ip
         probability = m_probabilityEstimator->probability(m_useTrackQuality,1,ipData.back().ip2d.significance(),track,*(it->jet()),*pv);
         prob2D.push_back(probability.first ? probability.second : -1.);
       } 
     }

     result->push_back(typename Product::value_type(ipData, prob2D, prob3D, selectedTracks,
                             *it, pvRef, direction, ghostTrackRef));
   }
 
   if (m_computeGhostTrack)
     iEvent.put(std::move(ghostTracks), "ghostTracks");
   iEvent.put(std::move(result));
}


#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

template <class Container, class Base, class Helper> void IPProducer<Container,Base,Helper>::checkEventSetup(const edm::EventSetup & iSetup)
 {
  
   const edm::eventsetup::EventSetupRecord & re2D= iSetup.get<BTagTrackProbability2DRcd>();
   const edm::eventsetup::EventSetupRecord & re3D= iSetup.get<BTagTrackProbability3DRcd>();
   unsigned long long cacheId2D= re2D.cacheIdentifier();
   unsigned long long cacheId3D= re3D.cacheIdentifier();

   if(cacheId2D!=m_calibrationCacheId2D || cacheId3D!=m_calibrationCacheId3D  )  //Calibration changed
   {
     //iSetup.get<BTagTrackProbabilityRcd>().get(calib);
     edm::ESHandle<TrackProbabilityCalibration> calib2DHandle;
     iSetup.get<BTagTrackProbability2DRcd>().get(calib2DHandle);
     edm::ESHandle<TrackProbabilityCalibration> calib3DHandle;
     iSetup.get<BTagTrackProbability3DRcd>().get(calib3DHandle);

     const TrackProbabilityCalibration *  ca2D= calib2DHandle.product();
     const TrackProbabilityCalibration *  ca3D= calib3DHandle.product();

     m_probabilityEstimator.reset(new HistogramProbabilityEstimator(ca3D,ca2D));

   }
   m_calibrationCacheId3D=cacheId3D;
   m_calibrationCacheId2D=cacheId2D;
}

// Specialized templates used to fill 'descriptions'
// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
template <>
void IPProducer<reco::TrackRefVector, reco::JTATagInfo, IPProducerHelpers::FromJTA>::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<double>("maximumTransverseImpactParameter",0.2);
  desc.add<int>("minimumNumberOfHits",8);
  desc.add<double>("minimumTransverseMomentum",1.0);
  desc.add<edm::InputTag>("primaryVertex",edm::InputTag("offlinePrimaryVertices"));
  desc.add<double>("maximumLongitudinalImpactParameter",17.0);
  desc.add<bool>("computeGhostTrack",true);
  desc.add<double>("ghostTrackPriorDeltaR",0.03);
  desc.add<edm::InputTag>("jetTracks",edm::InputTag("ak4JetTracksAssociatorAtVertexPF"));
  desc.add<bool>("jetDirectionUsingGhostTrack",false);
  desc.add<int>("minimumNumberOfPixelHits",2);
  desc.add<bool>("jetDirectionUsingTracks",false);
  desc.add<bool>("computeProbabilities",true);
  desc.add<bool>("useTrackQuality",false);
  desc.add<double>("maximumChiSquared",5.0);
  descriptions.addDefault(desc);
}

template <>
void IPProducer<std::vector<reco::CandidatePtr>,reco::JetTagInfo,  IPProducerHelpers::FromJetAndCands>::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<double>("maximumTransverseImpactParameter",0.2);
  desc.add<int>("minimumNumberOfHits",8);
  desc.add<double>("minimumTransverseMomentum",1.0);
  desc.add<edm::InputTag>("primaryVertex",edm::InputTag("offlinePrimaryVertices"));
  desc.add<double>("maximumLongitudinalImpactParameter",17.0);
  desc.add<bool>("computeGhostTrack",true);
  desc.add<double>("maxDeltaR",0.4);
  desc.add<edm::InputTag>("candidates",edm::InputTag("particleFlow"));
  desc.add<bool>("jetDirectionUsingGhostTrack",false);
  desc.add<int>("minimumNumberOfPixelHits",2);
  desc.add<bool>("jetDirectionUsingTracks",false);
  desc.add<bool>("computeProbabilities",true);
  desc.add<bool>("useTrackQuality",false);
  desc.add<edm::InputTag>("jets",edm::InputTag("ak4PFJetsCHS"));
  desc.add<double>("ghostTrackPriorDeltaR",0.03);
  desc.add<double>("maximumChiSquared",5.0);
  desc.addOptional<bool>("explicitJTA",false);
  descriptions.addDefault(desc);
}

#endif
