
#include "HLTrigger/special/interface/HLTSingleVertexPixelTrackFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors and destructor
//

HLTSingleVertexPixelTrackFilter::HLTSingleVertexPixelTrackFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
    pixelVerticesTag_ (iConfig.getParameter<edm::InputTag>("vertexCollection")),
    pixelTracksTag_ (iConfig.getParameter<edm::InputTag>("trackCollection")),
    min_Pt_  (iConfig.getParameter<double>("MinPt")),
    max_Pt_  (iConfig.getParameter<double>("MaxPt")),
    max_Eta_  (iConfig.getParameter<double>("MaxEta")),
    max_Vz_  (iConfig.getParameter<double>("MaxVz")),
    min_trks_  (iConfig.getParameter<int>("MinTrks")),
    min_sep_  (iConfig.getParameter<double>("MinSep"))
{
  pixelVerticesToken_ = consumes<reco::VertexCollection>(pixelVerticesTag_);
  pixelTracksToken_ = consumes<reco::RecoChargedCandidateCollection>(pixelTracksTag_);
}

HLTSingleVertexPixelTrackFilter::~HLTSingleVertexPixelTrackFilter()
{
}

void
HLTSingleVertexPixelTrackFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("vertexCollection",edm::InputTag("hltPixelVerticesForMinBias"));
  desc.add<edm::InputTag>("trackCollection",edm::InputTag("hltPixelCands"));
  desc.add<double>("MinPt",0.2);
  desc.add<double>("MaxPt",10000.0);
  desc.add<double>("MaxEta",1.0);
  desc.add<double>("MaxVz",10.0);
  desc.add<int>("MinTrks",30);
  desc.add<double>("MinSep",0.12);
  descriptions.add("hltSingleVertexPixelTrackFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTSingleVertexPixelTrackFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.
   if (saveTags())
      filterproduct.addCollectionTag(pixelTracksTag_);

   // Ref to Candidate object to be recorded in filter object
   edm::Ref<reco::RecoChargedCandidateCollection> candref;

   // Specific filter code
   bool accept = false;

   int nTrackCandidate = 0;

   // get hold of products from Event
   edm::Handle<reco::VertexCollection> vertexCollection;
   iEvent.getByToken( pixelVerticesToken_, vertexCollection );
   if(vertexCollection.isValid())
   {
     const reco::VertexCollection * vertices = vertexCollection.product();
     int npixelvertices = vertices->size();
     if (npixelvertices!=0)
     {
       double vzmax = 0;
       int   nmax = 0;
       reco::VertexCollection::const_iterator verticesItr;
       for (verticesItr=vertices->begin(); verticesItr!=vertices->end(); ++verticesItr)
       {
            int ntracksize = verticesItr->tracksSize();
            double vz = verticesItr->z();
            if(fabs(vz) > max_Vz_) continue;
            if( ntracksize > nmax)
            {
              vzmax = vz;
              nmax = ntracksize;
            }
       }

       edm::Handle<reco::RecoChargedCandidateCollection> trackCollection;
       iEvent.getByToken(pixelTracksToken_,trackCollection);
       if(trackCollection.isValid())
       {
          const reco::RecoChargedCandidateCollection * tracks = trackCollection.product();
          reco::RecoChargedCandidateCollection::const_iterator tracksItr;
          int icount=-1;
          for (tracksItr=tracks->begin(); tracksItr!=tracks->end(); ++tracksItr)
          {
            icount++;
            double eta = tracksItr->eta();
            if(fabs(eta) > max_Eta_) continue;
            double pt  = tracksItr->pt();
            if(pt < min_Pt_ || pt > max_Pt_) continue;
            double vz = tracksItr->vz();
            if(fabs(vz-vzmax) > min_sep_) continue;

            candref = edm::Ref<reco::RecoChargedCandidateCollection>(trackCollection, icount);
            filterproduct.addObject(trigger::TriggerTrack, candref);
            nTrackCandidate++;
          }
       }
     }
   }

   accept = ( nTrackCandidate >= min_trks_ );

   return accept;
}

DEFINE_FWK_MODULE(HLTSingleVertexPixelTrackFilter);
