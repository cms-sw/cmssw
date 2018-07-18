#include <string>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace pat {

    class PATIsolatedTrackProducer : public edm::stream::EDProducer<> {
      public:
          typedef pat::IsolatedTrack::LorentzVector LorentzVector;

          explicit PATIsolatedTrackProducer(const edm::ParameterSet&);
          ~PATIsolatedTrackProducer() override;

          void produce(edm::Event&, const edm::EventSetup&) override;
        
          // compute iso/miniiso
          void getIsolation(const LorentzVector& p4, const pat::PackedCandidateCollection* pc, int pc_idx,
                            pat::PFIsolation &iso, pat::PFIsolation &miniiso) const;

          bool getPFLeptonOverlap(const LorentzVector& p4, const pat::PackedCandidateCollection* pc) const;

          float getPFNeutralSum(const LorentzVector& p4, const pat::PackedCandidateCollection* pc, int pc_idx) const;

          void getNearestPCRef(const LorentzVector& p4, const pat::PackedCandidateCollection *pc, int pc_idx, int& pc_ref_idx) const;
      
          float getDeDx(const reco::DeDxHitInfo *hitInfo, bool doPixel, bool doStrip) const ;

          TrackDetMatchInfo getTrackDetMatchInfo(const edm::Event&, const edm::EventSetup&, const reco::Track&);

          void getCaloJetEnergy(const LorentzVector&, const reco::CaloJetCollection*, float&, float&) const;

      private:             
          const edm::EDGetTokenT<pat::PackedCandidateCollection>    pc_;
          const edm::EDGetTokenT<pat::PackedCandidateCollection>    lt_;
          const edm::EDGetTokenT<reco::TrackCollection>    gt_;
          const edm::EDGetTokenT<reco::VertexCollection>    pv_;
          const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > gt2pc_;
          const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > gt2lt_;
          const edm::EDGetTokenT<edm::Association<reco::PFCandidateCollection> > pc2pf_;
          const edm::EDGetTokenT<reco::CaloJetCollection> caloJets_;
          const edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > gt2dedxStrip_;
          const edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > gt2dedxPixel_;
          const edm::EDGetTokenT<reco::DeDxHitInfoAss> gt2dedxHitInfo_;
          const bool addPrescaledDeDxTracks_;
          const edm::EDGetTokenT<edm::ValueMap<int> >  gt2dedxHitInfoPrescale_;
          const bool usePrecomputedDeDxStrip_;
          const bool usePrecomputedDeDxPixel_;
          const float pT_cut_;  // only save cands with pT>pT_cut_
          const float pT_cut_noIso_;  // above this pT, don't apply any iso cut
          const float pfIsolation_DR_;  // isolation radius
          const float pfIsolation_DZ_;  // used in determining if pfcand is from PV or PU
          const float absIso_cut_;  // save if ANY of absIso, relIso, or miniRelIso pass the cuts 
          const float relIso_cut_;
          const float miniRelIso_cut_;
          const float caloJet_DR_;  // save energy of nearest calojet within caloJet_DR_
          const float pflepoverlap_DR_;  // pf lepton overlap radius
          const float pflepoverlap_pTmin_;  // pf lepton overlap min pT (only look at PF candidates with pT>pflepoverlap_pTmin_)
          const float pcRefNearest_DR_;  // radius for nearest charged packed candidate
          const float pcRefNearest_pTmin_;  // min pT for nearest charged packed candidate
          const float pfneutralsum_DR_;  // pf lepton overlap radius
          const bool saveDeDxHitInfo_;
          StringCutObjectSelector<pat::IsolatedTrack> saveDeDxHitInfoCut_;

          std::vector<double> miniIsoParams_;

          TrackDetectorAssociator trackAssociator_;
          TrackAssociatorParameters trackAssocParameters_;
    };
}

pat::PATIsolatedTrackProducer::PATIsolatedTrackProducer(const edm::ParameterSet& iConfig) :
  pc_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
  lt_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("lostTracks"))),
  gt_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("generalTracks"))),
  pv_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertices"))),
  gt2pc_(consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
  gt2lt_(consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("lostTracks"))),
  pc2pf_(consumes<edm::Association<reco::PFCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
  caloJets_(consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("caloJets"))),
  gt2dedxStrip_(consumes<edm::ValueMap<reco::DeDxData> >(iConfig.getParameter<edm::InputTag>("dEdxDataStrip"))),
  gt2dedxPixel_(consumes<edm::ValueMap<reco::DeDxData> >(iConfig.getParameter<edm::InputTag>("dEdxDataPixel"))),
  gt2dedxHitInfo_(consumes<reco::DeDxHitInfoAss>(iConfig.getParameter<edm::InputTag>("dEdxHitInfo"))),
  addPrescaledDeDxTracks_(iConfig.getParameter<bool>("addPrescaledDeDxTracks")),
  gt2dedxHitInfoPrescale_(addPrescaledDeDxTracks_ ? consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("dEdxHitInfoPrescale")) : edm::EDGetTokenT<edm::ValueMap<int>>()),
  usePrecomputedDeDxStrip_(iConfig.getParameter<bool>("usePrecomputedDeDxStrip")),
  usePrecomputedDeDxPixel_(iConfig.getParameter<bool>("usePrecomputedDeDxPixel")),
  pT_cut_         (iConfig.getParameter<double>("pT_cut")),
  pT_cut_noIso_   (iConfig.getParameter<double>("pT_cut_noIso")),
  pfIsolation_DR_ (iConfig.getParameter<double>("pfIsolation_DR")),
  pfIsolation_DZ_ (iConfig.getParameter<double>("pfIsolation_DZ")),
  absIso_cut_     (iConfig.getParameter<double>("absIso_cut")),
  relIso_cut_     (iConfig.getParameter<double>("relIso_cut")),
  miniRelIso_cut_ (iConfig.getParameter<double>("miniRelIso_cut")),
  caloJet_DR_     (iConfig.getParameter<double>("caloJet_DR")),
  pflepoverlap_DR_ (iConfig.getParameter<double>("pflepoverlap_DR")),
  pflepoverlap_pTmin_ (iConfig.getParameter<double>("pflepoverlap_pTmin")),
  pcRefNearest_DR_ (iConfig.getParameter<double>("pcRefNearest_DR")),
  pcRefNearest_pTmin_ (iConfig.getParameter<double>("pcRefNearest_pTmin")),
  pfneutralsum_DR_ (iConfig.getParameter<double>("pfneutralsum_DR")),
  saveDeDxHitInfo_(iConfig.getParameter<bool>("saveDeDxHitInfo")),
  saveDeDxHitInfoCut_(iConfig.getParameter<std::string>("saveDeDxHitInfoCut"))
{
    // TrackAssociator parameters
    edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
    edm::ConsumesCollector iC = consumesCollector();
    trackAssocParameters_.loadParameters( parameters, iC );

    trackAssociator_.useDefaultPropagator();

    miniIsoParams_ = iConfig.getParameter<std::vector<double> >("miniIsoParams");
    if(miniIsoParams_.size() != 3)
        throw cms::Exception("ParameterError") << "miniIsoParams must have exactly 3 elements.\n";

    produces< pat::IsolatedTrackCollection > ();

    if (saveDeDxHitInfo_) {
       produces<reco::DeDxHitInfoCollection>();
       produces<reco::DeDxHitInfoAss>();
    }
}

pat::PATIsolatedTrackProducer::~PATIsolatedTrackProducer() {}

void pat::PATIsolatedTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // packedPFCandidate collection
    edm::Handle<pat::PackedCandidateCollection> pc_h;
    iEvent.getByToken( pc_, pc_h );
    const pat::PackedCandidateCollection *pc = pc_h.product();

    // lostTracks collection
    edm::Handle<pat::PackedCandidateCollection> lt_h;
    iEvent.getByToken( lt_, lt_h );
    const pat::PackedCandidateCollection *lt = lt_h.product();

    // generalTracks collection
    edm::Handle<reco::TrackCollection> gt_h;
    iEvent.getByToken( gt_, gt_h );
    const reco::TrackCollection *generalTracks = gt_h.product();

    // get the primary vertex
    edm::Handle<reco::VertexCollection> pvs;
    iEvent.getByToken( pv_, pvs );
    const reco::Vertex & pv = (*pvs)[0];

    // generalTracks-->packedPFCandidate association
    edm::Handle<edm::Association<pat::PackedCandidateCollection> > gt2pc;
    iEvent.getByToken(gt2pc_, gt2pc);

    // generalTracks-->lostTracks association
    edm::Handle<edm::Association<pat::PackedCandidateCollection> > gt2lt;
    iEvent.getByToken(gt2lt_, gt2lt);

    // packedPFCandidates-->particleFlow(reco::PFCandidate) association
    edm::Handle<edm::Association<reco::PFCandidateCollection> > pc2pf;
    iEvent.getByToken(pc2pf_, pc2pf);

    edm::Handle<reco::CaloJetCollection> caloJets;
    iEvent.getByToken(caloJets_, caloJets);

    // associate generalTracks with their DeDx data (estimator for strip dE/dx)
    edm::Handle<edm::ValueMap<reco::DeDxData> > gt2dedxStrip;
    iEvent.getByToken(gt2dedxStrip_, gt2dedxStrip);

    // associate generalTracks with their DeDx data (estimator for pixel dE/dx)
    edm::Handle<edm::ValueMap<reco::DeDxData> > gt2dedxPixel;
    iEvent.getByToken(gt2dedxPixel_, gt2dedxPixel);

    // associate generalTracks with their DeDx hit info (used to estimate pixel dE/dx)
    edm::Handle<reco::DeDxHitInfoAss> gt2dedxHitInfo;
    iEvent.getByToken(gt2dedxHitInfo_, gt2dedxHitInfo);
    edm::Handle<edm::ValueMap<int>> gt2dedxHitInfoPrescale;
    if (addPrescaledDeDxTracks_) {
        iEvent.getByToken(gt2dedxHitInfoPrescale_, gt2dedxHitInfoPrescale);
    }

    edm::ESHandle<HcalChannelQuality> hcalQ_h;
    iSetup.get<HcalChannelQualityRcd>().get("withTopo", hcalQ_h);
    const HcalChannelQuality *hcalQ = hcalQ_h.product();
    
    edm::ESHandle<EcalChannelStatus> ecalS_h;
    iSetup.get<EcalChannelStatusRcd>().get(ecalS_h);
    const EcalChannelStatus *ecalS = ecalS_h.product();

    auto outDeDxC = std::make_unique<reco::DeDxHitInfoCollection>();
    std::vector<int> dEdXass;

    auto outPtrP = std::make_unique<std::vector<pat::IsolatedTrack>>();

    //add general tracks
    for(unsigned int igt=0; igt<generalTracks->size(); igt++){
        const reco::Track &gentk = (*gt_h)[igt];
        reco::TrackRef tkref = reco::TrackRef(gt_h, igt);
        pat::PackedCandidateRef pcref = (*gt2pc)[tkref];
        pat::PackedCandidateRef ltref = (*gt2lt)[tkref];
        const pat::PackedCandidate & pfCand = *(pcref.get());
        const pat::PackedCandidate & lostTrack = *(ltref.get());

        // Determine if this general track is associated with anything in packedPFCandidates or lostTracks
        // Sometimes, a track gets associated w/ a neutral pfCand.
        // In this case, ignore the pfCand and take from lostTracks
        bool isInPackedCands = (pcref.isNonnull() && pcref.id()==pc_h.id() && pfCand.charge()!=0);
        bool isInLostTracks  = (ltref.isNonnull() && ltref.id()==lt_h.id());

        LorentzVector p4;
        pat::PackedCandidateRef refToCand;
        int pdgId, charge, fromPV;
        float dz, dxy, dzError, dxyError;
        int pfCandInd; //to avoid counting packedPFCands in their own isolation
	int ltCandInd; //to avoid pointing lost track to itself when looking for closest

        // get the four-momentum and charge
        if(isInPackedCands){
            p4        = pfCand.p4();
            charge    = pfCand.charge();
            pfCandInd = pcref.key();
	    ltCandInd = -1;
        }else if(isInLostTracks){
            p4        = lostTrack.p4();
            charge    = lostTrack.charge();
            pfCandInd = -1;
	    ltCandInd = ltref.key();
        }else{
            double m = 0.13957018; //assume pion mass
            double E = sqrt(m*m + gentk.p()*gentk.p());
            p4.SetPxPyPzE(gentk.px(), gentk.py(), gentk.pz(), E);
            charge = gentk.charge();
            pfCandInd = -1;
	    ltCandInd = -1;
        }

        int prescaled = 0;
        if (addPrescaledDeDxTracks_) {
            const auto &dedxRef = (*gt2dedxHitInfo)[tkref];
            if (dedxRef.isNonnull()) {
                prescaled = (*gt2dedxHitInfoPrescale)[dedxRef];
            }
        }

        if(p4.pt() < pT_cut_ && prescaled <= 1) 
            continue;
        if(charge == 0)
            continue;

        // get the isolation of the track
        pat::PFIsolation isolationDR03;
        pat::PFIsolation miniIso;
        getIsolation(p4, pc, pfCandInd, isolationDR03, miniIso);
        
        // isolation cut
        if( p4.pt() < pT_cut_noIso_ && prescaled <= 1 && 
            !(isolationDR03.chargedHadronIso() < absIso_cut_ ||
              isolationDR03.chargedHadronIso()/p4.pt() < relIso_cut_ ||
              miniIso.chargedHadronIso()/p4.pt() < miniRelIso_cut_))
            continue;

        // get the rest after the pt/iso cuts. Saves some runtime
        if(isInPackedCands){
            pdgId     = pfCand.pdgId();
            dz        = pfCand.dz();
            dxy       = pfCand.dxy();
            dzError   = pfCand.hasTrackDetails() ? pfCand.dzError()  : gentk.dzError();
            dxyError  = pfCand.hasTrackDetails() ? pfCand.dxyError() : gentk.dxyError();
            fromPV    = pfCand.fromPV();
            refToCand = pcref;
        }else if(isInLostTracks){
            pdgId     = lostTrack.pdgId();
            dz        = lostTrack.dz();
            dxy       = lostTrack.dxy();
            dzError   = lostTrack.hasTrackDetails() ? lostTrack.dzError()  : gentk.dzError();
            dxyError  = lostTrack.hasTrackDetails() ? lostTrack.dxyError() : gentk.dxyError();
            fromPV    = lostTrack.fromPV();
            refToCand = ltref;
        }else{
            pdgId     = 0;
            dz        = gentk.dz(pv.position());
            dxy       = gentk.dxy(pv.position());
            dzError   = gentk.dzError();
            dxyError  = gentk.dxyError();
            fromPV    = -1;
            refToCand = pat::PackedCandidateRef();  //NULL reference
        }

        float caloJetEm, caloJetHad;
        getCaloJetEnergy(p4, caloJets.product(), caloJetEm, caloJetHad);

	bool pfLepOverlap = getPFLeptonOverlap(p4, pc);
	float pfNeutralSum = getPFNeutralSum(p4, pc, pfCandInd);
	
	pat::PackedCandidateRef refToNearestPF = pat::PackedCandidateRef();
	int refToNearestPF_idx=-1;
	getNearestPCRef(p4, pc, pfCandInd, refToNearestPF_idx);
	if(refToNearestPF_idx!=-1)
	  refToNearestPF = pat::PackedCandidateRef(pc_h, refToNearestPF_idx);
	
	pat::PackedCandidateRef refToNearestLostTrack = pat::PackedCandidateRef();
	int refToNearestLostTrack_idx=-1;
	getNearestPCRef(p4, lt, ltCandInd, refToNearestLostTrack_idx);
	if(refToNearestLostTrack_idx!=-1)
	  refToNearestLostTrack = pat::PackedCandidateRef(lt_h, refToNearestLostTrack_idx);
	
        // if no dEdx info exists, just store -1
        float dEdxPixel=-1, dEdxStrip=-1;        
        if(usePrecomputedDeDxStrip_ && gt2dedxStrip.isValid() && gt2dedxStrip->contains(tkref.id())){
            dEdxStrip = (*gt2dedxStrip)[tkref].dEdx();
        }else if(gt2dedxHitInfo.isValid() && gt2dedxHitInfo->contains(tkref.id())){
            const reco::DeDxHitInfo* hitInfo = (*gt2dedxHitInfo)[tkref].get();
            dEdxStrip = getDeDx(hitInfo, false, true);
        }
        if(usePrecomputedDeDxPixel_ && gt2dedxPixel.isValid() && gt2dedxPixel->contains(tkref.id())){
            dEdxPixel = (*gt2dedxPixel)[tkref].dEdx();
        }else if(gt2dedxHitInfo.isValid() && gt2dedxHitInfo->contains(tkref.id())){
            const reco::DeDxHitInfo* hitInfo = (*gt2dedxHitInfo)[tkref].get();
            dEdxPixel = getDeDx(hitInfo, true, false);
        }

        int trackQuality = gentk.qualityMask();

        // get the associated ecal/hcal detectors
        TrackDetMatchInfo trackDetInfo = getTrackDetMatchInfo(iEvent, iSetup, gentk);

        // fill ecal/hcal status vectors
        std::vector<uint32_t> crossedHcalStatus;
        for(auto const & did : trackDetInfo.crossedHcalIds){
            crossedHcalStatus.push_back(hcalQ->getValues(did.rawId())->getValue());
        }
        std::vector<uint16_t> crossedEcalStatus;
        for(auto const & did : trackDetInfo.crossedEcalIds){
            crossedEcalStatus.push_back(ecalS->find(did.rawId())->getStatusCode());
        }

        int deltaEta = int((trackDetInfo.trkGlobPosAtEcal.eta() - gentk.eta())/0.5 * 250);
        int deltaPhi = int((trackDetInfo.trkGlobPosAtEcal.phi() - gentk.phi())/0.5 * 250);
        if(deltaEta < -250) deltaEta = -250;
        if(deltaEta > 250)  deltaEta = 250;
        if(deltaPhi < -250) deltaPhi = -250;
        if(deltaPhi > 250)  deltaPhi = 250;

        outPtrP->push_back(pat::IsolatedTrack(isolationDR03, miniIso, caloJetEm, caloJetHad, pfLepOverlap, pfNeutralSum, p4,
                                              charge, pdgId, dz, dxy, dzError, dxyError,
                                              gentk.hitPattern(), dEdxStrip, dEdxPixel, fromPV, trackQuality,
                                              crossedEcalStatus, crossedHcalStatus,
                                              deltaEta, deltaPhi, refToCand,
					      refToNearestPF, refToNearestLostTrack));
        outPtrP->back().setStatus(prescaled);

        if (saveDeDxHitInfo_) {
            const auto &dedxRef = (*gt2dedxHitInfo)[tkref];
            if (saveDeDxHitInfoCut_(outPtrP->back()) && dedxRef.isNonnull()) {
                outDeDxC->push_back( *dedxRef );
                dEdXass.push_back(outDeDxC->size()-1);
            } else {
                dEdXass.push_back(-1);
            }
        }

    }

    // there are some number of pfcandidates with no associated track
    // (mostly electrons, with a handful of muons)
    // here we find these and store. Track-specific variables get some default values
    for(unsigned int ipc=0; ipc<pc->size(); ipc++){
        const pat::PackedCandidate& pfCand = pc->at(ipc);
        pat::PackedCandidateRef pcref = pat::PackedCandidateRef(pc_h, ipc);
        reco::PFCandidateRef pfref = (*pc2pf)[pcref];

        // already counted if it has a track reference in the generalTracks collection
        if(pfref.get()->trackRef().isNonnull() && pfref.get()->trackRef().id() == gt_h.id())
            continue;

        LorentzVector p4;
        pat::PackedCandidateRef refToCand;
        int pdgId, charge, fromPV;
        float dz, dxy, dzError, dxyError;

        p4 = pfCand.p4();
        charge = pfCand.charge();

        if(p4.pt() < pT_cut_)
            continue;
        if(charge == 0)
            continue;

        // get the isolation of the track
        pat::PFIsolation isolationDR03;
        pat::PFIsolation miniIso;
        getIsolation(p4, pc, ipc, isolationDR03, miniIso);
        
        // isolation cut
        if( p4.pt() < pT_cut_noIso_ && 
            !(isolationDR03.chargedHadronIso() < absIso_cut_ ||
              isolationDR03.chargedHadronIso()/p4.pt() < relIso_cut_ ||
              miniIso.chargedHadronIso()/p4.pt() < miniRelIso_cut_))
            continue;

        pdgId     = pfCand.pdgId();
        dz        = pfCand.dz();
        dxy       = pfCand.dxy();
        if (pfCand.hasTrackDetails()){
            dzError   = pfCand.dzError();
            dxyError  = pfCand.dxyError();
        } else {
            dzError = 0;
            dxyError = 0;
        }
        fromPV    = pfCand.fromPV();
        refToCand = pcref;

        float caloJetEm, caloJetHad;
        getCaloJetEnergy(p4, caloJets.product(), caloJetEm, caloJetHad);

	bool pfLepOverlap = getPFLeptonOverlap(p4, pc);
	float pfNeutralSum = getPFNeutralSum(p4, pc, ipc);

	pat::PackedCandidateRef refToNearestPF = pat::PackedCandidateRef();
	int refToNearestPF_idx=-1;
	getNearestPCRef(p4, pc, ipc, refToNearestPF_idx);
	if(refToNearestPF_idx!=-1)
	  refToNearestPF = pat::PackedCandidateRef(pc_h, refToNearestPF_idx);
	
	pat::PackedCandidateRef refToNearestLostTrack = pat::PackedCandidateRef();
	int refToNearestLostTrack_idx=-1;
	getNearestPCRef(p4, lt, -1, refToNearestLostTrack_idx);
	if(refToNearestLostTrack_idx!=-1)
	  refToNearestLostTrack = pat::PackedCandidateRef(lt_h, refToNearestLostTrack_idx);

        // fill with default values
        reco::HitPattern hp;
        float dEdxPixel=-1, dEdxStrip=-1;
        int trackQuality=0;
        std::vector<uint16_t> ecalStatus;
        std::vector<uint32_t> hcalStatus;
        int deltaEta=0;
        int deltaPhi=0;

        outPtrP->push_back(pat::IsolatedTrack(isolationDR03, miniIso, caloJetEm, caloJetHad, pfLepOverlap, pfNeutralSum, p4,
                                              charge, pdgId, dz, dxy, dzError, dxyError,
                                              hp, dEdxStrip, dEdxPixel, fromPV, trackQuality,
                                              ecalStatus, hcalStatus, deltaEta, deltaPhi, refToCand,
					      refToNearestPF, refToNearestLostTrack));

        dEdXass.push_back(-1); // these never have dE/dx hit info, as there's no track
    }
    
    auto orphHandle = iEvent.put(std::move(outPtrP));
    if (saveDeDxHitInfo_) {
        auto dedxOH = iEvent.put(std::move(outDeDxC));
        auto dedxMatch = std::make_unique<reco::DeDxHitInfoAss>(dedxOH);
        reco::DeDxHitInfoAss::Filler filler(*dedxMatch);  
        filler.insert(orphHandle, dEdXass.begin(), dEdXass.end()); 
        filler.fill();
        iEvent.put(std::move(dedxMatch));
    }
    
}


void pat::PATIsolatedTrackProducer::getIsolation(const LorentzVector& p4, 
                                                 const pat::PackedCandidateCollection *pc, int pc_idx,
                                                 pat::PFIsolation &iso, pat::PFIsolation &miniiso) const
{
        float chiso=0, nhiso=0, phiso=0, puiso=0;   // standard isolation
        float chmiso=0, nhmiso=0, phmiso=0, pumiso=0;  // mini isolation
        float miniDR = std::max(miniIsoParams_[0], std::min(miniIsoParams_[1], miniIsoParams_[2]/p4.pt()));
        for(pat::PackedCandidateCollection::const_iterator pf_it = pc->begin(); pf_it != pc->end(); pf_it++){
            if(int(pf_it - pc->begin()) == pc_idx)  //don't count itself
                continue;
            int id = std::abs(pf_it->pdgId());
            bool fromPV = (pf_it->fromPV()>1 || fabs(pf_it->dz()) < pfIsolation_DZ_);
            float pt = pf_it->p4().pt();
            float dr = deltaR(p4, pf_it->p4());

            if(dr < pfIsolation_DR_){
                // charged cands from PV get added to trackIso
                if(id==211 && fromPV)
                    chiso += pt;
                // charged cands not from PV get added to pileup iso
                else if(id==211)
                    puiso += pt;
                // neutral hadron iso
                if(id==130)
                    nhiso += pt;
                // photon iso
                if(id==22)
                    phiso += pt;
            }
            // same for mini isolation
            if(dr < miniDR){
                if(id == 211 && fromPV)
                    chmiso += pt;
                else if(id == 211)
                    pumiso += pt;
                if(id == 130)
                    nhmiso += pt;
                if(id == 22)
                    phmiso += pt;
            }
        }

        iso = pat::PFIsolation(chiso,nhiso,phiso,puiso);
        miniiso = pat::PFIsolation(chmiso,nhmiso,phmiso,pumiso);

}

//get overlap of isolated track with a PF lepton
bool pat::PATIsolatedTrackProducer::getPFLeptonOverlap(const LorentzVector& p4, const pat::PackedCandidateCollection *pc) const
{
        bool isOverlap = false;
	float dr_min = pflepoverlap_DR_;
	int id_drmin = 0;
        for (const auto & pf : *pc) {
            int id = std::abs(pf.pdgId());
	    int charge = std::abs(pf.charge());
            bool fromPV = (pf.fromPV()>1 || std::abs(pf.dz()) < pfIsolation_DZ_);
            float pt = pf.pt();
	    if(charge==0) // exclude neutral candidates
	      continue;
	    if(!(fromPV)) // exclude candidates not from PV
	      continue;
	    if(pt < pflepoverlap_pTmin_) // exclude pf candidates w/ pT below threshold
	      continue;

            float dr = deltaR(p4, pf.p4());
            if(dr > pflepoverlap_DR_) // exclude pf candidates far from isolated track
	      continue;

	    if(dr < dr_min){
	      dr_min = dr;
	      id_drmin = id;
	    }
	    
        }

	if(dr_min<pflepoverlap_DR_ && (id_drmin==11 || id_drmin==13))
	  isOverlap = true;
	
	return isOverlap;
}

//get ref to nearest pf packed candidate
void pat::PATIsolatedTrackProducer::getNearestPCRef(const LorentzVector& p4, const pat::PackedCandidateCollection *pc, int pc_idx, int& pc_ref_idx) const
{
	float dr_min = pcRefNearest_DR_;
	float dr_min_pu = pcRefNearest_DR_;
	int pc_ref_idx_pu=-1;
        for(pat::PackedCandidateCollection::const_iterator pf_it = pc->begin(); pf_it != pc->end(); pf_it++){
            if(int(pf_it - pc->begin()) == pc_idx)  //don't count itself
                continue;
	    int charge = std::abs(pf_it->charge());
            bool fromPV = (pf_it->fromPV()>1 || fabs(pf_it->dz()) < pfIsolation_DZ_);
            float pt = pf_it->p4().pt();
            float dr = deltaR(p4, pf_it->p4());
	    if(charge==0) // exclude neutral candidates
	      continue;
	    if(pt < pcRefNearest_pTmin_) // exclude candidates w/ pT below threshold
	      continue;
	    if(dr > dr_min && dr > dr_min_pu) // exclude too far candidates
	      continue;

	    if(fromPV){ // Priority to candidates from PV
	      if(dr < dr_min){
		dr_min = dr;
		pc_ref_idx = int(pf_it - pc->begin());
	      }
	    }
	    else{ // Otherwise, store candidate from non-PV if no candidate from PV found
	      if(dr < dr_min_pu){
		dr_min_pu = dr;
		pc_ref_idx_pu = int(pf_it - pc->begin());
	      }
	    }
        }

	if(pc_ref_idx == -1 && pc_ref_idx_pu != -1) // If no candidate from PV was found, store candidate from non-PV (if found)
	  pc_ref_idx = pc_ref_idx_pu;
}

// get PF neutral pT sum around isolated track
float pat::PATIsolatedTrackProducer::getPFNeutralSum(const LorentzVector& p4, const pat::PackedCandidateCollection *pc, int pc_idx) const
{
        float nsum=0;
        float nhsum=0, phsum=0;
        for(pat::PackedCandidateCollection::const_iterator pf_it = pc->begin(); pf_it != pc->end(); pf_it++){
            if(int(pf_it - pc->begin()) == pc_idx)  //don't count itself
                continue;
            int id = std::abs(pf_it->pdgId());
            float pt = pf_it->p4().pt();
            float dr = deltaR(p4, pf_it->p4());

            if(dr < pfneutralsum_DR_){
	      // neutral hadron sum
	      if(id==130)
		nhsum += pt;
	      // photon iso
	      if(id==22)
		phsum += pt;
            }
        }

	nsum = nhsum + phsum;
	
	return nsum;
}

// get the estimated DeDx in either the pixels or strips (or both)
float pat::PATIsolatedTrackProducer::getDeDx(const reco::DeDxHitInfo *hitInfo, bool doPixel, bool doStrip) const
{
    if(hitInfo == nullptr){
        return -1;
    }

    std::vector<float> charge_vec;
    for(unsigned int ih=0; ih<hitInfo->size(); ih++){

        bool isPixel = (hitInfo->pixelCluster(ih) != nullptr);
        bool isStrip = (hitInfo->stripCluster(ih) != nullptr);

        if(isPixel && !doPixel) continue;
        if(isStrip && !doStrip) continue;

        // probably shouldn't happen
        if(!isPixel && !isStrip) continue;

        // shape selection for strips
        if(isStrip && !DeDxTools::shapeSelection(*(hitInfo->stripCluster(ih))))
            continue;
        
        float Norm=0;       
        if(isPixel)
            Norm = 3.61e-06; //compute the normalization factor to get the energy in MeV/mm
        if(isStrip)
            Norm = 3.61e-06 * 265;
            
        //save the dE/dx in MeV/mm to a vector.  
        charge_vec.push_back(Norm*hitInfo->charge(ih)/hitInfo->pathlength(ih));           
    }

    int size = charge_vec.size();
    float result = 0.0;

    //build the harmonic 2 dE/dx estimator
    float expo = -2;
    for(int i=0; i<size; i++){
        result += pow(charge_vec[i], expo);
    }
    result = (size>0) ? pow(result/size, 1./expo) : 0.0;

    return result;
}

TrackDetMatchInfo pat::PATIsolatedTrackProducer::getTrackDetMatchInfo(const edm::Event& iEvent, 
                                                                      const edm::EventSetup& iSetup,
                                                                      const reco::Track& track)
{
    edm::ESHandle<MagneticField> bField;
    iSetup.get<IdealMagneticFieldRecord>().get(bField);
    FreeTrajectoryState initialState = trajectoryStateTransform::initialFreeState(track,&*bField);

    // can't use the associate() using reco::Track directly, since 
    // track->extra() is non-null but segfaults when trying to use it
    return trackAssociator_.associate(iEvent, iSetup, trackAssocParameters_, &initialState);
}

void pat::PATIsolatedTrackProducer::getCaloJetEnergy(const LorentzVector& p4, const reco::CaloJetCollection* cJets,
                                                     float &caloJetEm, float& caloJetHad) const
{
    float nearestDR = 999;
    int ind = -1;
    for(unsigned int i=0; i<cJets->size(); i++){
        float dR = deltaR(cJets->at(i).p4(), p4);
        if(dR < caloJet_DR_ && dR < nearestDR){
            nearestDR = dR;
            ind = i;
        }
    }

    if(ind==-1){
        caloJetEm = 0;
        caloJetHad = 0;
    }else{
        const reco::CaloJet & cJet = cJets->at(ind);
        caloJetEm = cJet.emEnergyInEB() + cJet.emEnergyInEE() + cJet.emEnergyInHF();
        caloJetHad = cJet.hadEnergyInHB() + cJet.hadEnergyInHE() + cJet.hadEnergyInHF();
    }

}


using pat::PATIsolatedTrackProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATIsolatedTrackProducer);
