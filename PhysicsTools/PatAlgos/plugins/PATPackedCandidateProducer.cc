#include <string>


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"

/*#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
*/
//#define CRAZYSORT 

namespace pat {
    ///conversion map from quality flags used in PV association and miniAOD one
    static int qualityMap[8]  = {1,0,1,1,4,4,5,6};

    class PATPackedCandidateProducer : public edm::EDProducer {
        public:
            explicit PATPackedCandidateProducer(const edm::ParameterSet&);
            ~PATPackedCandidateProducer();

            virtual void produce(edm::Event&, const edm::EventSetup&);

            //sorting of cands to maximize the zlib compression
            bool candsOrdering(pat::PackedCandidate i,pat::PackedCandidate j) {
                if (std::abs(i.charge()) == std::abs(j.charge())) {
                    if(i.charge()!=0){
                        if(i.pt() > minPtForTrackProperties_ and j.pt() <= minPtForTrackProperties_ ) return true;
                        if(i.pt() <= minPtForTrackProperties_ and j.pt() > minPtForTrackProperties_ ) return false;
                  }
                   if(i.vertexRef() == j.vertexRef()) 
                      return i.eta() > j.eta();
                   else 
                      return i.vertexRef().key() < j.vertexRef().key();
                }
                return std::abs(i.charge()) > std::abs(j.charge());
            }
            template <typename T>
            std::vector<size_t> sort_indexes(const std::vector<T> &v ) {
              std::vector<size_t> idx(v.size());
              for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
              std::sort(idx.begin(), idx.end(),[&v,this](size_t i1, size_t i2) { return candsOrdering(v[i1],v[i2]);});
              return idx;
           }

        private:
            edm::EDGetTokenT<reco::PFCandidateCollection>    Cands_;
            edm::EDGetTokenT<reco::VertexCollection>         PVs_;
            edm::EDGetTokenT<edm::Association<reco::VertexCollection> > PVAsso_;
            edm::EDGetTokenT<edm::ValueMap<int> >            PVAssoQuality_;
            edm::EDGetTokenT<reco::VertexCollection>         PVOrigs_;
            edm::EDGetTokenT<reco::TrackCollection>          TKOrigs_;
            edm::EDGetTokenT< edm::ValueMap<float> >         PuppiWeight_;
            edm::EDGetTokenT< edm::ValueMap<float> >         PuppiWeightNoLep_;            
            edm::EDGetTokenT<edm::ValueMap<reco::CandidatePtr> >    PuppiCandsMap_;
            edm::EDGetTokenT<std::vector< reco::PFCandidate >  >    PuppiCands_;
            edm::EDGetTokenT<std::vector< reco::PFCandidate >  >    PuppiCandsNoLep_;            
            edm::EDGetTokenT<edm::View<reco::CompositePtrCandidate> > SVWhiteList_;

            double minPtForTrackProperties_;
            // for debugging
            float calcDxy(float dx, float dy, float phi) {
                return - dx * std::sin(phi) + dy * std::cos(phi);
            }
            float calcDz(reco::Candidate::Point p, reco::Candidate::Point v, const reco::Candidate &c) {
                return p.Z()-v.Z() - ((p.X()-v.X()) * c.px() + (p.Y()-v.Y())*c.py()) * c.pz()/(c.pt()*c.pt());
            }
    };
}

pat::PATPackedCandidateProducer::PATPackedCandidateProducer(const edm::ParameterSet& iConfig) :
  Cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCollection"))),
  PVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("inputVertices"))),
  PVAsso_(consumes<edm::Association<reco::VertexCollection> >(iConfig.getParameter<edm::InputTag>("vertexAssociator"))),
  PVAssoQuality_(consumes<edm::ValueMap<int> >(iConfig.getParameter<edm::InputTag>("vertexAssociator"))),
  PVOrigs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("originalVertices"))),
  TKOrigs_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("originalTracks"))),
  PuppiWeight_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("PuppiSrc"))),
  PuppiWeightNoLep_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("PuppiNoLepSrc"))),  
  PuppiCandsMap_(consumes<edm::ValueMap<reco::CandidatePtr> >(iConfig.getParameter<edm::InputTag>("PuppiSrc"))),
  PuppiCands_(consumes<std::vector< reco::PFCandidate > >(iConfig.getParameter<edm::InputTag>("PuppiSrc"))),
  PuppiCandsNoLep_(consumes<std::vector< reco::PFCandidate > >(iConfig.getParameter<edm::InputTag>("PuppiNoLepSrc"))),  
  SVWhiteList_(consumes<edm::View< reco::CompositePtrCandidate > >(iConfig.getParameter<edm::InputTag>("secondaryVerticesForWhiteList"))),
  minPtForTrackProperties_(iConfig.getParameter<double>("minPtForTrackProperties"))
{
  produces< std::vector<pat::PackedCandidate> > ();
  produces< edm::Association<pat::PackedCandidateCollection> > ();
  produces< edm::Association<reco::PFCandidateCollection> > ();
}

pat::PATPackedCandidateProducer::~PATPackedCandidateProducer() {}



void pat::PATPackedCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Handle<reco::PFCandidateCollection> cands;
    iEvent.getByToken( Cands_, cands );
    std::vector<reco::Candidate>::const_iterator cand;

    edm::Handle< edm::ValueMap<float> > puppiWeight;
    iEvent.getByToken( PuppiWeight_, puppiWeight );
    edm::Handle<edm::ValueMap<reco::CandidatePtr> > puppiCandsMap;
    iEvent.getByToken( PuppiCandsMap_, puppiCandsMap );
    edm::Handle<std::vector< reco::PFCandidate > > puppiCands;
    iEvent.getByToken( PuppiCands_, puppiCands );
    std::vector<int> mappingPuppi(puppiCands->size());

    edm::Handle< edm::ValueMap<float> > puppiWeightNoLep;
    iEvent.getByToken( PuppiWeightNoLep_, puppiWeightNoLep );
    edm::Handle<std::vector< reco::PFCandidate > > puppiCandsNoLep;
    iEvent.getByToken( PuppiCandsNoLep_, puppiCandsNoLep );  

    std::vector<reco::CandidatePtr> puppiCandsNoLepPtrs;
    if (puppiCandsNoLep.isValid()){
      for (auto pup : *puppiCandsNoLep){
        puppiCandsNoLepPtrs.push_back(pup.sourceCandidatePtr(0));
      }
    }
    auto const& puppiCandsNoLepV = puppiCandsNoLep.product();

    edm::Handle<reco::VertexCollection> PVOrigs;
    iEvent.getByToken( PVOrigs_, PVOrigs );

    edm::Handle<edm::Association<reco::VertexCollection> > assoHandle;
    iEvent.getByToken(PVAsso_,assoHandle);
    edm::Handle<edm::ValueMap<int> > assoQualityHandle;
    iEvent.getByToken(PVAssoQuality_,assoQualityHandle);
    const edm::Association<reco::VertexCollection> &  associatedPV=*(assoHandle.product());
    const edm::ValueMap<int>  &  associationQuality=*(assoQualityHandle.product());
           
    edm::Handle<edm::View<reco::CompositePtrCandidate > > svWhiteListHandle;
    iEvent.getByToken(SVWhiteList_,svWhiteListHandle);
    const edm::View<reco::CompositePtrCandidate > &  svWhiteList=*(svWhiteListHandle.product());
    std::set<unsigned int> whiteList;
    for(unsigned int i=0; i<svWhiteList.size();i++)
    {
      for(unsigned int j=0; j< svWhiteList[i].numberOfSourceCandidatePtrs(); j++) {
          const edm::Ptr<reco::Candidate> & c = svWhiteList[i].sourceCandidatePtr(j);
          if(c.id() == cands.id()) whiteList.insert(c.key());
      }
    }
 

    edm::Handle<reco::VertexCollection> PVs;
    iEvent.getByToken( PVs_, PVs );
    reco::VertexRef PV(PVs.id());
    math::XYZPoint  PVpos;


    edm::Handle<reco::TrackCollection> TKOrigs;
    iEvent.getByToken( TKOrigs_, TKOrigs );
    std::auto_ptr< std::vector<pat::PackedCandidate> > outPtrP( new std::vector<pat::PackedCandidate> );
    std::vector<int> mapping(cands->size());
    std::vector<int> mappingReverse(cands->size());
    std::vector<int> mappingTk(TKOrigs->size(), -1);

    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        const reco::PFCandidate &cand=(*cands)[ic];
        float phiAtVtx = cand.phi();
        const reco::Track *ctrack = 0;
        if ((abs(cand.pdgId()) == 11 || cand.pdgId() == 22) && cand.gsfTrackRef().isNonnull()) {
            ctrack = &*cand.gsfTrackRef();
        } else if (cand.trackRef().isNonnull()) {
            ctrack = &*cand.trackRef();
        }
        if (ctrack) {
          float dist=1e99;
          int pvi=-1;
          for(size_t ii=0;ii<PVs->size();ii++){
            float dz=std::abs(ctrack->dz( ((*PVs)[ii]).position()));
            if(dz<dist) {pvi=ii;dist=dz; }
          }
          PV = reco::VertexRef(PVs, pvi);
          math::XYZPoint vtx = cand.vertex();
          pat::PackedCandidate::LostInnerHits lostHits = pat::PackedCandidate::noLostInnerHits;
          const reco::VertexRef & PVOrig = associatedPV[reco::CandidatePtr(cands,ic)];
          if(PVOrig.isNonnull()) PV = reco::VertexRef(PVs, PVOrig.key()); // WARNING: assume the PV slimmer is keeping same order
          int quality=associationQuality[reco::CandidatePtr(cands,ic)];
//          if ((size_t)pvi!=PVOrig.key()) std::cout << "not closest in Z" << pvi << " " << PVOrig.key() << " " << cand.pt() << " " << quality << std::endl;
          //          TrajectoryStateOnSurface tsos = extrapolator.extrapolate(trajectoryStateTransform::initialFreeState(*ctrack,&*magneticField), RecoVertex::convertPos(PV->position()));
          //   vtx = tsos.globalPosition();
          //          phiAtVtx = tsos.globalDirection().phi();
          vtx = ctrack->referencePoint();
          phiAtVtx = ctrack->phi();
          int nlost = ctrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
          if (nlost == 0) { 
            if (ctrack->hitPattern().hasValidHitInFirstPixelBarrel()) {
              lostHits = pat::PackedCandidate::validHitInFirstPixelBarrelLayer;
            }
          } else {
            lostHits = ( nlost == 1 ? pat::PackedCandidate::oneLostInnerHit : pat::PackedCandidate::moreLostInnerHits);
          }


          outPtrP->push_back( pat::PackedCandidate(cand.polarP4(), vtx, phiAtVtx, cand.pdgId(), PV));
          outPtrP->back().setAssociationQuality(pat::PackedCandidate::PVAssociationQuality(qualityMap[quality]));
          if(cand.trackRef().isNonnull() && PVOrig->trackWeight(cand.trackRef()) > 0.5 && quality == 7) {
                  outPtrP->back().setAssociationQuality(pat::PackedCandidate::UsedInFitTight);
          }
          // properties of the best track 
          outPtrP->back().setLostInnerHits( lostHits );
          if(outPtrP->back().pt() > minPtForTrackProperties_ || whiteList.find(ic)!=whiteList.end()) {
            outPtrP->back().setTrackProperties(*ctrack);
            //outPtrP->back().setTrackProperties(*ctrack,tsos.curvilinearError());
          }

          // these things are always for the CKF track
          outPtrP->back().setTrackHighPurity( cand.trackRef().isNonnull() && cand.trackRef()->quality(reco::Track::highPurity) );
          if (cand.muonRef().isNonnull()) {
            outPtrP->back().setMuonID(cand.muonRef()->isStandAloneMuon(), cand.muonRef()->isGlobalMuon());
          }
        } else {

          if (!PVs->empty()) {
            PV = reco::VertexRef(PVs, 0);
            PVpos = PV->position();
          }

          outPtrP->push_back( pat::PackedCandidate(cand.polarP4(), PVpos, cand.phi(), cand.pdgId(), PV));
          outPtrP->back().setAssociationQuality(pat::PackedCandidate::PVAssociationQuality(pat::PackedCandidate::UsedInFitTight));
        }
	
        if (puppiWeight.isValid()){
           reco::PFCandidateRef pkref( cands, ic );
                 // outPtrP->back().setPuppiWeight( (*puppiWeight)[pkref]);
           
           float puppiWeightVal = (*puppiWeight)[pkref];
           float puppiWeightNoLepVal = 0.0;

           // Check the "no lepton" puppi weights. 
           // If present, then it is not a lepton, use stored weight
           // If absent, it is a lepton, so set the weight to 1.0
           if ( puppiWeightNoLep.isValid() ) {
             // Look for the pointer inside the "no lepton" candidate collection.
             auto pkrefPtr = pkref->sourceCandidatePtr(0);

             bool foundNoLep = false;
             for ( size_t ipcnl = 0; ipcnl < puppiCandsNoLepPtrs.size(); ipcnl++){
              if (puppiCandsNoLepPtrs[ipcnl] == pkrefPtr){
                foundNoLep = true;
                  puppiWeightNoLepVal = puppiCandsNoLepV->at(ipcnl).pt()/cand.pt(); // a hack for now, should use the value map
                  break;
                }
              }
              if ( !foundNoLep || puppiWeightNoLepVal > 1 ) {
                puppiWeightNoLepVal = 1.0;
              }
            }
          outPtrP->back().setPuppiWeight( puppiWeightVal, puppiWeightNoLepVal );

          mappingPuppi[((*puppiCandsMap)[pkref]).key()]=ic;
        }
	
        mapping[ic] = ic; // trivial at the moment!
        if (cand.trackRef().isNonnull() && cand.trackRef().id() == TKOrigs.id()) {
	  mappingTk[cand.trackRef().key()] = ic;	    
        }

    }

    std::auto_ptr< std::vector<pat::PackedCandidate> > outPtrPSorted( new std::vector<pat::PackedCandidate> );
    std::vector<size_t> order=sort_indexes(*outPtrP);
    std::vector<size_t> reverseOrder(order.size());
    for(size_t i=0,nc=cands->size();i<nc;i++) {
        outPtrPSorted->push_back((*outPtrP)[order[i]]);
        reverseOrder[order[i]] = i;
        mappingReverse[order[i]]=i;
    }

    // Fix track association for sorted candidates
    for(size_t i=0,ntk=mappingTk.size();i<ntk;i++){
        int origInd = mappingTk[i];
        if (origInd>=0 ) mappingTk[i]=reverseOrder[origInd]; 
    }

    for(size_t i=0,ntk=mappingPuppi.size();i<ntk;i++){
        mappingPuppi[i]=reverseOrder[mappingPuppi[i]];
    }

    edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put( outPtrPSorted );

    // now build the two maps
    std::auto_ptr<edm::Association<pat::PackedCandidateCollection> > pf2pc(new edm::Association<pat::PackedCandidateCollection>(oh   ));
    std::auto_ptr<edm::Association<reco::PFCandidateCollection   > > pc2pf(new edm::Association<reco::PFCandidateCollection   >(cands));
    edm::Association<pat::PackedCandidateCollection>::Filler pf2pcFiller(*pf2pc);
    edm::Association<reco::PFCandidateCollection   >::Filler pc2pfFiller(*pc2pf);
    pf2pcFiller.insert(cands, mappingReverse.begin(), mappingReverse.end());
    pc2pfFiller.insert(oh   , order.begin(), order.end());
    // include also the mapping track -> packed PFCand
    pf2pcFiller.insert(TKOrigs, mappingTk.begin(), mappingTk.end());
    pf2pcFiller.insert(puppiCands, mappingPuppi.begin(), mappingPuppi.end());

    pf2pcFiller.fill();
    pc2pfFiller.fill();
    iEvent.put(pf2pc);
    iEvent.put(pc2pf);

}


using pat::PATPackedCandidateProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedCandidateProducer);
