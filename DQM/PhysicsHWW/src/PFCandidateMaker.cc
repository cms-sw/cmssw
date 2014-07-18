#include "CommonTools/ParticleFlow/interface/PFPileUpAlgo.h"

#include "DQM/PhysicsHWW/interface/PFCandidateMaker.h"

typedef math::XYZTLorentzVectorF LorentzVector;
typedef math::XYZPoint Point;
using namespace reco;
using namespace edm;
using namespace std;

PFCandidateMaker::PFCandidateMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector){

  PFCandidateCollection_ = iCollector.consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandsInputTag"));
  PFElectrons_           = iCollector.consumes<edm::ValueMap<reco::PFCandidatePtr> >(iConfig.getParameter<edm::InputTag>("pfElectronsTag"));
  TrackCollection_       = iCollector.consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackInputTag"));
  thePVCollection_       = iCollector.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexInputTag"));

}

void PFCandidateMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  hww.Load_pfcands_p4();
  hww.Load_pfcands_charge();
  hww.Load_pfcands_particleId();
  hww.Load_pfcands_vtxidx();
  hww.Load_pfcands_trkidx();
  hww.Load_pfcands_pfelsidx();

  PFPileUpAlgo *pfPileUpAlgo_ = new PFPileUpAlgo();

  bool validToken;

  //get pfcandidates
  const reco::PFCandidateCollection *pfCandidates;
  Handle<PFCandidateCollection> pfCandidatesHandle;
  validToken = iEvent.getByToken(PFCandidateCollection_, pfCandidatesHandle);
  if(!validToken) return;
  pfCandidates  = pfCandidatesHandle.product();

  //get pfelectrons
  typedef edm::ValueMap<reco::PFCandidatePtr> PFCandMap;
  Handle<PFCandMap> pfElectronsHandle;
  validToken = iEvent.getByToken(PFElectrons_, pfElectronsHandle);
  if(!validToken) return;
  const PFCandMap *pfElectrons  = pfElectronsHandle.product();

  // get tracks
  Handle<reco::TrackCollection>  track_h;
  validToken = iEvent.getByToken(TrackCollection_, track_h);
  if(!validToken) return;

  // get vertices
  Handle<reco::VertexCollection> vertex_h;
  validToken = iEvent.getByToken(thePVCollection_, vertex_h);
  if(!validToken) return;
  const reco::VertexCollection *vertices = vertex_h.product();


  int iCand = 0;
  for( PFCandidateCollection::const_iterator pf_it = pfCandidates->begin(); pf_it != pfCandidates->end(); pf_it++ ) {

    int pfflags = 0;
    for( unsigned int i = 0; i < 17; i++ ) {
      if(pf_it->flag((PFCandidate::Flags)i)) pfflags |= (1<<i);
    }

    iCand++;

    hww.pfcands_p4()			          .push_back( LorentzVector(pf_it->px(), pf_it->py(), pf_it->pz(), pf_it->p())	  );
    hww.pfcands_charge()		        .push_back( pf_it->charge()						                                        );
    hww.pfcands_particleId()		    .push_back( pf_it->translateTypeToPdgId(pf_it->particleId())	                  );

    //for charged pfcandidates, find corresponding track index
    //here we take the track directly from PFCandidate::trackRef()
    if( pf_it->charge() != 0 ){

      // match pf cand to a vertex
      // using the no pileup algo
      hww.pfcands_vtxidx().push_back(pfPileUpAlgo_->chargedHadronVertex(*vertices, *pf_it));

      reco::TrackRef pftrack = pf_it->trackRef();

      int trkidx = 0;
      bool foundTrack = false;

      reco::TrackCollection::const_iterator tracks_end = track_h->end();

      for (reco::TrackCollection::const_iterator itrk = track_h->begin(); itrk != tracks_end; ++itrk) {
        
        reco::TrackRef trkref( track_h , itrk - track_h->begin() );

        if( pftrack.key() == trkref.key() ){
        
          //sanity check
          float dpt = pftrack->pt() - trkref->pt();
          if( fabs( dpt ) > 0.1 ){
            edm::LogWarning("SanityCheck") << "Warning: pfcandidate track pt - matched track pt = " << dpt << ", possible mismatch";
          }

          //found corresponding track
          hww.pfcands_trkidx().push_back( trkidx );
          foundTrack = true;
          break;
        }
        
        ++trkidx;
      }

      if( !foundTrack ){
        //no matched track found, set trkidx to -1
        hww.pfcands_trkidx().push_back(-1);
      }
      
    }else{
      //neutral particle, set trkidx to -2
      hww.pfcands_trkidx().push_back(-2);
      // no vertex match
      hww.pfcands_vtxidx().push_back(-2);
    }


    //find corresponding PFElectron index
    if (pf_it->particleId() == PFCandidate::e){

     int index = -1; 

     if (pf_it->gsfTrackRef().isNonnull()) {

       int pfGsfTkId = pf_it->gsfTrackRef().key();
       unsigned int elsIndex = 0;
       PFCandMap::const_iterator el_pit = pfElectrons->begin();
       unsigned int nE = el_pit.size();
       for(unsigned int iE = 0; iE < nE; ++iE){
         const PFCandidatePtr& el_it = el_pit[iE];
         if (el_it.isNull()) continue;
         int elGsfTkId = -1;
         if (el_it->gsfTrackRef().isNonnull()) elGsfTkId = el_it->gsfTrackRef().key();
         if (elGsfTkId==pfGsfTkId) {
           index = elsIndex;
           break;

         }
         elsIndex++;
       }            
     }

    hww.pfcands_pfelsidx().push_back(index);

    } else {
            hww.pfcands_pfelsidx().push_back(-2);
          }

        
  }//loop over candidate collection

}
