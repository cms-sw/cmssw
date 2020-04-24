#include <string>


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
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
    const static int qualityMap[8]  = {1,0,1,1,4,4,5,6};

    class PATPackedCandidateProducer : public edm::global::EDProducer<> {
        public:
            explicit PATPackedCandidateProducer(const edm::ParameterSet&);
            ~PATPackedCandidateProducer() override;

            void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

            //sorting of cands to maximize the zlib compression
            bool candsOrdering(pat::PackedCandidate i,pat::PackedCandidate j) const {
                if (std::abs(i.charge()) == std::abs(j.charge())) {
                    if(i.charge()!=0){
                        if(i.hasTrackDetails() and ! j.hasTrackDetails() ) return true;
                        if(! i.hasTrackDetails() and  j.hasTrackDetails() ) return false;
                        if(i.covarianceSchema() >  j.covarianceSchema() ) return true;
                        if(i.covarianceSchema() <  j.covarianceSchema() ) return false;

                  }
                   if(i.vertexRef() == j.vertexRef()) 
                      return i.eta() > j.eta();
                   else 
                      return i.vertexRef().key() < j.vertexRef().key();
                }
                return std::abs(i.charge()) > std::abs(j.charge());
            }
            template <typename T>
            std::vector<size_t> sort_indexes(const std::vector<T> &v ) const {
              std::vector<size_t> idx(v.size());
              for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
              std::sort(idx.begin(), idx.end(),[&v,this](size_t i1, size_t i2) { return candsOrdering(v[i1],v[i2]);});
              return idx;
           }

        private: 
            //if PuppiSrc && PuppiNoLepSrc are empty, usePuppi is false
            //otherwise assumes that if they are set, you wanted to use puppi and will throw an exception
            //if the puppis are not found
            const bool usePuppi_;  
            
            const edm::EDGetTokenT<reco::PFCandidateCollection>    Cands_;
            const edm::EDGetTokenT<reco::VertexCollection>         PVs_;
            const edm::EDGetTokenT<edm::Association<reco::VertexCollection> > PVAsso_;
            const edm::EDGetTokenT<edm::ValueMap<int> >            PVAssoQuality_;
            const edm::EDGetTokenT<reco::VertexCollection>         PVOrigs_;
            const edm::EDGetTokenT<reco::TrackCollection>          TKOrigs_;
            const edm::EDGetTokenT< edm::ValueMap<float> >         PuppiWeight_;
            const edm::EDGetTokenT< edm::ValueMap<float> >         PuppiWeightNoLep_;
            const edm::EDGetTokenT<edm::ValueMap<reco::CandidatePtr> >    PuppiCandsMap_;
            const edm::EDGetTokenT<std::vector< reco::PFCandidate >  >    PuppiCands_;
            const edm::EDGetTokenT<std::vector< reco::PFCandidate >  >    PuppiCandsNoLep_;
            std::vector< edm::EDGetTokenT<edm::View<reco::Candidate> > > SVWhiteLists_;
            const bool storeChargedHadronIsolation_;
            const edm::EDGetTokenT<edm::ValueMap<bool> >            ChargedHadronIsolation_;

            const double minPtForChargedHadronProperties_;
            const double minPtForTrackProperties_;
            const int covarianceVersion_;
            const std::vector<int> covariancePackingSchemas_;
      
            const bool storeTiming_;
      
            // for debugging
            float calcDxy(float dx, float dy, float phi) const {
                return - dx * std::sin(phi) + dy * std::cos(phi);
            }
            float calcDz(reco::Candidate::Point p, reco::Candidate::Point v, const reco::Candidate &c) const {
                return p.Z()-v.Z() - ((p.X()-v.X()) * c.px() + (p.Y()-v.Y())*c.py()) * c.pz()/(c.pt()*c.pt());
            }
    };
}

pat::PATPackedCandidateProducer::PATPackedCandidateProducer(const edm::ParameterSet& iConfig) :
  usePuppi_(!iConfig.getParameter<edm::InputTag>("PuppiSrc").encode().empty() || 
	    !iConfig.getParameter<edm::InputTag>("PuppiNoLepSrc").encode().empty()),
  Cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCollection"))),
  PVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("inputVertices"))),
  PVAsso_(consumes<edm::Association<reco::VertexCollection> >(iConfig.getParameter<edm::InputTag>("vertexAssociator"))),
  PVAssoQuality_(consumes<edm::ValueMap<int> >(iConfig.getParameter<edm::InputTag>("vertexAssociator"))),
  PVOrigs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("originalVertices"))),
  TKOrigs_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("originalTracks"))),
  PuppiWeight_(usePuppi_ ? consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("PuppiSrc")) : edm::EDGetTokenT< edm::ValueMap<float> >()),
  PuppiWeightNoLep_(usePuppi_ ? consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("PuppiNoLepSrc")) : edm::EDGetTokenT< edm::ValueMap<float> >()),
  PuppiCandsMap_(usePuppi_ ? consumes<edm::ValueMap<reco::CandidatePtr> >(iConfig.getParameter<edm::InputTag>("PuppiSrc")) : edm::EDGetTokenT<edm::ValueMap<reco::CandidatePtr> >() ),
  PuppiCands_(usePuppi_ ? consumes<std::vector< reco::PFCandidate > >(iConfig.getParameter<edm::InputTag>("PuppiSrc")) : edm::EDGetTokenT<std::vector< reco::PFCandidate > >() ),
  PuppiCandsNoLep_(usePuppi_ ? consumes<std::vector< reco::PFCandidate > >(iConfig.getParameter<edm::InputTag>("PuppiNoLepSrc")) : edm::EDGetTokenT<std::vector< reco::PFCandidate > >()),
  storeChargedHadronIsolation_(!iConfig.getParameter<edm::InputTag>("chargedHadronIsolation").encode().empty()),
  ChargedHadronIsolation_(consumes<edm::ValueMap<bool> >(iConfig.getParameter<edm::InputTag>("chargedHadronIsolation"))),
  minPtForChargedHadronProperties_(iConfig.getParameter<double>("minPtForChargedHadronProperties")),
  minPtForTrackProperties_(iConfig.getParameter<double>("minPtForTrackProperties")),
  covarianceVersion_(iConfig.getParameter<int >("covarianceVersion")),
  covariancePackingSchemas_(iConfig.getParameter<std::vector<int> >("covariancePackingSchemas")),
  storeTiming_(iConfig.getParameter<bool>("storeTiming"))  
{
  std::vector<edm::InputTag> sv_tags = iConfig.getParameter<std::vector<edm::InputTag> >("secondaryVerticesForWhiteList");
  for(auto itag : sv_tags){
    SVWhiteLists_.push_back(
      consumes<edm::View< reco::Candidate > >(itag)
      );
  }

  produces< std::vector<pat::PackedCandidate> > ();
  produces< edm::Association<pat::PackedCandidateCollection> > ();
  produces< edm::Association<reco::PFCandidateCollection> > ();
}

pat::PATPackedCandidateProducer::~PATPackedCandidateProducer() {}



void pat::PATPackedCandidateProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

    edm::Handle<reco::PFCandidateCollection> cands;
    iEvent.getByToken( Cands_, cands );
    std::vector<reco::Candidate>::const_iterator cand;

    edm::Handle<edm::ValueMap<float> > puppiWeight;
    edm::Handle<edm::ValueMap<reco::CandidatePtr> > puppiCandsMap;
    edm::Handle<std::vector< reco::PFCandidate > > puppiCands;
    edm::Handle<edm::ValueMap<float> > puppiWeightNoLep;
    edm::Handle<std::vector< reco::PFCandidate > > puppiCandsNoLep;
    std::vector<reco::CandidatePtr> puppiCandsNoLepPtrs;
    if(usePuppi_){
      iEvent.getByToken( PuppiWeight_, puppiWeight );
      iEvent.getByToken( PuppiCandsMap_, puppiCandsMap );
      iEvent.getByToken( PuppiCands_, puppiCands );
      iEvent.getByToken( PuppiWeightNoLep_, puppiWeightNoLep );
      iEvent.getByToken( PuppiCandsNoLep_, puppiCandsNoLep );  
      for (auto pup : *puppiCandsNoLep){
        puppiCandsNoLepPtrs.push_back(pup.sourceCandidatePtr(0));
      }
    }  
    std::vector<int> mappingPuppi(usePuppi_ ? puppiCands->size() : 0);

    edm::Handle<reco::VertexCollection> PVOrigs;
    iEvent.getByToken( PVOrigs_, PVOrigs );

    edm::Handle<edm::Association<reco::VertexCollection> > assoHandle;
    iEvent.getByToken(PVAsso_,assoHandle);
    edm::Handle<edm::ValueMap<int> > assoQualityHandle;
    iEvent.getByToken(PVAssoQuality_,assoQualityHandle);
    const edm::Association<reco::VertexCollection> &  associatedPV=*(assoHandle.product());
    const edm::ValueMap<int>  &  associationQuality=*(assoQualityHandle.product());
           
    edm::Handle<edm::ValueMap<bool> > chargedHadronIsolationHandle;
    if(storeChargedHadronIsolation_)
      iEvent.getByToken(ChargedHadronIsolation_,chargedHadronIsolationHandle);

    std::set<unsigned int> whiteList;
    std::set<reco::TrackRef> whiteListTk;
    for(auto itoken : SVWhiteLists_) {
      edm::Handle<edm::View<reco::Candidate > > svWhiteListHandle;
      iEvent.getByToken(itoken, svWhiteListHandle);
      const edm::View<reco::Candidate > &  svWhiteList=*(svWhiteListHandle.product());
      for(unsigned int i=0; i<svWhiteList.size();i++) {
	//Whitelist via Ptrs
        for(unsigned int j=0; j< svWhiteList[i].numberOfSourceCandidatePtrs(); j++) {
          const edm::Ptr<reco::Candidate> & c = svWhiteList[i].sourceCandidatePtr(j);
          if(c.id() == cands.id()) whiteList.insert(c.key());

        }
	//Whitelist via RecoCharged
	for(auto dau = svWhiteList[i].begin(); dau != svWhiteList[i].end() ; dau++){
            const reco::RecoChargedCandidate * chCand=dynamic_cast<const reco::RecoChargedCandidate *>(&(*dau));
	    if(chCand!=nullptr) {
		whiteListTk.insert(chCand->track());
	    }
	}
      }
    }
 

    edm::Handle<reco::VertexCollection> PVs;
    iEvent.getByToken( PVs_, PVs );
    reco::VertexRef PV(PVs.id());
    reco::VertexRefProd PVRefProd(PVs);
    math::XYZPoint  PVpos;


    edm::Handle<reco::TrackCollection> TKOrigs;
    iEvent.getByToken( TKOrigs_, TKOrigs );
    auto outPtrP = std::make_unique<std::vector<pat::PackedCandidate>>();
    std::vector<int> mapping(cands->size());
    std::vector<int> mappingReverse(cands->size());
    std::vector<int> mappingTk(TKOrigs->size(), -1);

    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        const reco::PFCandidate &cand=(*cands)[ic];
        const reco::Track *ctrack = nullptr;
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
	  float ptTrk = ctrack->pt();
	  float etaAtVtx = ctrack->eta();
          float phiAtVtx = ctrack->phi();

          int nlost = ctrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
          if (nlost == 0) { 
            if (ctrack->hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1)) {
              lostHits = pat::PackedCandidate::validHitInFirstPixelBarrelLayer;
            }
          } else {
            lostHits = ( nlost == 1 ? pat::PackedCandidate::oneLostInnerHit : pat::PackedCandidate::moreLostInnerHits);
          }

	  
          outPtrP->push_back( pat::PackedCandidate(cand.polarP4(), vtx, ptTrk, etaAtVtx, phiAtVtx, cand.pdgId(), PVRefProd, PV.key()));
          outPtrP->back().setAssociationQuality(pat::PackedCandidate::PVAssociationQuality(qualityMap[quality]));
          outPtrP->back().setCovarianceVersion(covarianceVersion_);
          if(cand.trackRef().isNonnull() && PVOrig.isNonnull() && PVOrig->trackWeight(cand.trackRef()) > 0.5 && quality == 7) {
                  outPtrP->back().setAssociationQuality(pat::PackedCandidate::UsedInFitTight);
          }
          // properties of the best track 
          outPtrP->back().setLostInnerHits( lostHits );
          if(outPtrP->back().pt() > minPtForTrackProperties_ || 
	     outPtrP->back().ptTrk() > minPtForTrackProperties_ ||
	     whiteList.find(ic)!=whiteList.end() || 
             (cand.trackRef().isNonnull() &&  whiteListTk.find(cand.trackRef())!=whiteListTk.end())
	    ) {
	      outPtrP->back().setFirstHit(ctrack->hitPattern().getHitPattern(reco::HitPattern::TRACK_HITS, 0));
              if(abs(outPtrP->back().pdgId())==22) {
                  outPtrP->back().setTrackProperties(*ctrack,covariancePackingSchemas_[4],covarianceVersion_);
	      } else { 
                  if( ctrack->hitPattern().numberOfValidPixelHits() >0) {
		      outPtrP->back().setTrackProperties(*ctrack,covariancePackingSchemas_[0],covarianceVersion_); //high quality 
		  }  else { 
		      outPtrP->back().setTrackProperties(*ctrack,covariancePackingSchemas_[1],covarianceVersion_);
		  } 
              }            
            //outPtrP->back().setTrackProperties(*ctrack,tsos.curvilinearError());
          } else {
            if(outPtrP->back().pt() > 0.5 ){ 
                if(ctrack->hitPattern().numberOfValidPixelHits() >0)  outPtrP->back().setTrackProperties(*ctrack,covariancePackingSchemas_[2],covarianceVersion_); //low quality, with pixels
                  else       outPtrP->back().setTrackProperties(*ctrack,covariancePackingSchemas_[3],covarianceVersion_); //low quality, without pixels
            }
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
	
          outPtrP->push_back( pat::PackedCandidate(cand.polarP4(), PVpos, cand.pt(), cand.eta(), cand.phi(), cand.pdgId(), PVRefProd, PV.key()));
          outPtrP->back().setAssociationQuality(pat::PackedCandidate::PVAssociationQuality(pat::PackedCandidate::UsedInFitTight));
        }
    
	// neutrals and isolated charged hadrons

        bool isIsolatedChargedHadron = false;
        if(storeChargedHadronIsolation_) {
          const edm::ValueMap<bool>  &  chargedHadronIsolation=*(chargedHadronIsolationHandle.product());
          isIsolatedChargedHadron=((cand.pt()>minPtForChargedHadronProperties_)&&(chargedHadronIsolation[reco::PFCandidateRef(cands,ic)]));
          outPtrP->back().setIsIsolatedChargedHadron(isIsolatedChargedHadron);
        }

	if(abs(cand.pdgId()) == 1 || abs(cand.pdgId()) == 130) {
	  outPtrP->back().setHcalFraction(cand.hcalEnergy()/(cand.ecalEnergy()+cand.hcalEnergy()));
        } else if(isIsolatedChargedHadron) {
          outPtrP->back().setRawCaloFraction((cand.rawEcalEnergy()+cand.rawHcalEnergy())/cand.energy());
          outPtrP->back().setHcalFraction(cand.rawHcalEnergy()/(cand.rawEcalEnergy()+cand.rawHcalEnergy()));
	} else {
	  outPtrP->back().setHcalFraction(0);
	}
	
	//specifically this is the PFLinker requirements to apply the e/gamma regression
	if(cand.particleId() == reco::PFCandidate::e || (cand.particleId() == reco::PFCandidate::gamma && cand.mva_nothing_gamma()>0.)) { 
	  outPtrP->back().setGoodEgamma();
	}
       
        if (usePuppi_){
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
                  puppiWeightNoLepVal = puppiCandsNoLep->at(ipcnl).pt()/cand.pt(); // a hack for now, should use the value map
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
	
        if (storeTiming_ && cand.isTimeValid())  {
          outPtrP->back().setTime(cand.time(), cand.timeError());
        }

        mapping[ic] = ic; // trivial at the moment!
        if (cand.trackRef().isNonnull() && cand.trackRef().id() == TKOrigs.id()) {
	  mappingTk[cand.trackRef().key()] = ic;	    
        }

    }

    auto outPtrPSorted = std::make_unique<std::vector<pat::PackedCandidate>>();
    std::vector<size_t> order=sort_indexes(*outPtrP);
    std::vector<size_t> reverseOrder(order.size());
    for(size_t i=0,nc=cands->size();i<nc;i++) {
        outPtrPSorted->push_back((*outPtrP)[order[i]]);
        reverseOrder[order[i]] = i;
        mappingReverse[order[i]]=i;
    }

    // Fix track association for sorted candidates
    for(size_t i=0,ntk=mappingTk.size();i<ntk;i++){
        if(mappingTk[i] >= 0)
          mappingTk[i]=reverseOrder[mappingTk[i]];
    }

    for(size_t i=0,ntk=mappingPuppi.size();i<ntk;i++){
        mappingPuppi[i]=reverseOrder[mappingPuppi[i]];
    }

    edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put(std::move(outPtrPSorted));

    // now build the two maps
    auto pf2pc = std::make_unique<edm::Association<pat::PackedCandidateCollection>>(oh);
    auto pc2pf = std::make_unique<edm::Association<reco::PFCandidateCollection>>(cands);
    edm::Association<pat::PackedCandidateCollection>::Filler pf2pcFiller(*pf2pc);
    edm::Association<reco::PFCandidateCollection   >::Filler pc2pfFiller(*pc2pf);
    pf2pcFiller.insert(cands, mappingReverse.begin(), mappingReverse.end());
    pc2pfFiller.insert(oh   , order.begin(), order.end());
    // include also the mapping track -> packed PFCand
    pf2pcFiller.insert(TKOrigs, mappingTk.begin(), mappingTk.end());
    if(usePuppi_) pf2pcFiller.insert(puppiCands, mappingPuppi.begin(), mappingPuppi.end());

    pf2pcFiller.fill();
    pc2pfFiller.fill();
    iEvent.put(std::move(pf2pc));
    iEvent.put(std::move(pc2pf));

}


using pat::PATPackedCandidateProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedCandidateProducer);
