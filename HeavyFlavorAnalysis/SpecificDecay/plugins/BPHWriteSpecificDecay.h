#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHWriteSpecificDecay_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHWriteSpecificDecay_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexCompositePtrCandidate.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include <string>
#include <vector>
#include <map>

class TH1F;
class BPHRecoCandidate;

class BPHWriteSpecificDecay : public BPHAnalyzerWrapper<BPHModuleWrapper::stream_producer> {
public:
  explicit BPHWriteSpecificDecay(const edm::ParameterSet& ps);
  ~BPHWriteSpecificDecay() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& ev, const edm::EventSetup& es) override;
  virtual void fill(edm::Event& ev, const BPHEventSetupWrapper& es);

private:
  std::string pVertexLabel;
  std::string patMuonLabel;
  std::string ccCandsLabel;
  std::string pfCandsLabel;
  std::string pcCandsLabel;
  std::string gpCandsLabel;
  std::string k0CandsLabel;
  std::string l0CandsLabel;
  std::string kSCandsLabel;
  std::string lSCandsLabel;

  // token wrappers to allow running both on "old" and "new" CMSSW versions
  BPHESTokenWrapper<MagneticField, IdealMagneticFieldRecord> magFieldToken;
  BPHESTokenWrapper<TransientTrackBuilder, TransientTrackRecord> ttBToken;
  BPHTokenWrapper<std::vector<reco::Vertex> > pVertexToken;
  BPHTokenWrapper<pat::MuonCollection> patMuonToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > ccCandsToken;
  BPHTokenWrapper<std::vector<reco::PFCandidate> > pfCandsToken;
  BPHTokenWrapper<std::vector<BPHTrackReference::candidate> > pcCandsToken;
  BPHTokenWrapper<std::vector<pat::GenericParticle> > gpCandsToken;
  BPHTokenWrapper<std::vector<reco::VertexCompositeCandidate> > k0CandsToken;
  BPHTokenWrapper<std::vector<reco::VertexCompositeCandidate> > l0CandsToken;
  BPHTokenWrapper<std::vector<reco::VertexCompositePtrCandidate> > kSCandsToken;
  BPHTokenWrapper<std::vector<reco::VertexCompositePtrCandidate> > lSCandsToken;

  bool usePV;
  bool usePM;
  bool useCC;
  bool usePF;
  bool usePC;
  bool useGP;
  bool useK0;
  bool useL0;
  bool useKS;
  bool useLS;

  std::string oniaName;
  std::string sdName;
  std::string ssName;
  std::string buName;
  std::string bpName;
  std::string bdName;
  std::string bsName;
  std::string k0Name;
  std::string l0Name;
  std::string b0Name;
  std::string lbName;
  std::string bcName;
  std::string psi2SName;
  std::string x3872Name;

  enum recoType {
    Onia,
    Pmm,
    Psi1,
    Psi2,
    Ups,
    Ups1,
    Ups2,
    Ups3,
    Kx0,
    Pkk,
    Bu,
    Bp,
    Bd,
    Bs,
    K0s,
    Lambda0,
    B0,
    Lambdab,
    Bc,
    Psi2S,
    X3872
  };
  enum parType {
    ptMin,
    etaMax,
    mPsiMin,
    mPsiMax,
    mKx0Min,
    mKx0Max,
    mPhiMin,
    mPhiMax,
    mK0sMin,
    mK0sMax,
    mLambda0Min,
    mLambda0Max,
    massMin,
    massMax,
    probMin,
    mFitMin,
    mFitMax,
    constrMass,
    constrSigma,
    requireJPsi,
    constrMJPsi,
    constrMPsi2,
    writeCandidate
  };
  std::map<std::string, recoType> rMap;
  std::map<std::string, parType> pMap;
  std::map<std::string, parType> fMap;
  std::map<recoType, std::map<parType, double> > parMap;

  bool recoOnia;
  bool recoKx0;
  bool recoPkk;
  bool recoBu;
  bool recoBp;
  bool recoBd;
  bool recoBs;
  bool recoK0s;
  bool recoLambda0;
  bool recoB0;
  bool recoLambdab;
  bool recoBc;
  bool recoPsi2S;
  bool recoX3872;

  bool allKx0;
  bool allPkk;
  bool allK0s;
  bool allLambda0;

  bool writeOnia;
  bool writeKx0;
  bool writePkk;
  bool writeBu;
  bool writeBp;
  bool writeBd;
  bool writeBs;
  bool writeK0s;
  bool writeLambda0;
  bool writeB0;
  bool writeLambdab;
  bool writeBc;
  bool writePsi2S;
  bool writeX3872;

  bool writeVertex;
  bool writeMomentum;

  std::vector<BPHPlusMinusConstCandPtr> lFull;
  std::vector<BPHPlusMinusConstCandPtr> lJPsi;
  std::vector<BPHRecoConstCandPtr> lSd;
  std::vector<BPHRecoConstCandPtr> lSs;
  std::vector<BPHRecoConstCandPtr> lBu;
  std::vector<BPHRecoConstCandPtr> lBp;
  std::vector<BPHRecoConstCandPtr> lBd;
  std::vector<BPHRecoConstCandPtr> lBs;
  std::vector<BPHPlusMinusConstCandPtr> lK0;
  std::vector<BPHPlusMinusConstCandPtr> lL0;
  std::vector<BPHRecoConstCandPtr> lB0;
  std::vector<BPHRecoConstCandPtr> lLb;
  std::vector<BPHRecoConstCandPtr> lBc;
  std::vector<BPHRecoConstCandPtr> lPsi2S;
  std::vector<BPHRecoConstCandPtr> lX3872;

  std::map<const BPHRecoCandidate*, const BPHRecoCandidate*> jPsiOMap;
  std::map<const BPHRecoCandidate*, const BPHRecoCandidate*> daughMap;
  typedef edm::Ref<std::vector<reco::Vertex> > vertex_ref;
  std::map<const BPHRecoCandidate*, vertex_ref> pvRefMap;
  typedef edm::Ref<pat::CompositeCandidateCollection> compcc_ref;
  std::map<const BPHRecoCandidate*, compcc_ref> ccRefMap;

  void setRecoParameters(const edm::ParameterSet& ps);

  static void addTrackModes(const std::string& name, const BPHRecoCandidate& cand, std::string& modes, bool& count);

  static void addTrackModes(const std::string& name, const BPHRecoCandidate& cand, pat::CompositeCandidate& cc);

  template <class T>
  edm::OrphanHandle<pat::CompositeCandidateCollection> write(edm::Event& ev,
                                                             const std::vector<T>& list,
                                                             const std::string& name) {
    pat::CompositeCandidateCollection* ccList = new pat::CompositeCandidateCollection;
    int i;
    int n = list.size();
    std::map<const BPHRecoCandidate*, const BPHRecoCandidate*>::const_iterator dauIter;
    std::map<const BPHRecoCandidate*, const BPHRecoCandidate*>::const_iterator dauIend = daughMap.end();
    std::map<const BPHRecoCandidate*, const BPHRecoCandidate*>::const_iterator jpoIter;
    std::map<const BPHRecoCandidate*, const BPHRecoCandidate*>::const_iterator jpoIend = jPsiOMap.end();
    std::map<const BPHRecoCandidate*, vertex_ref>::const_iterator pvrIter;
    std::map<const BPHRecoCandidate*, vertex_ref>::const_iterator pvrIend = pvRefMap.end();
    std::map<const BPHRecoCandidate*, compcc_ref>::const_iterator ccrIter;
    std::map<const BPHRecoCandidate*, compcc_ref>::const_iterator ccrIend = ccRefMap.end();
    for (i = 0; i < n; ++i) {
      const T& ptr = list[i];
      ccList->push_back(ptr->composite());
      pat::CompositeCandidate& cc = ccList->back();
      std::string modes;
      bool count = false;
      addTrackModes("", *ptr, modes, count);
      cc.addUserData("trackModes", modes, true);
      addTrackModes("trackMode_", *ptr, cc);
      if ((pvrIter = pvRefMap.find(ptr.get())) != pvrIend)
        cc.addUserData("primaryVertex", pvrIter->second);
      const std::vector<std::string>& cNames = ptr->compNames();
      int j = 0;
      int m = cNames.size();
      while (j < m) {
        const std::string& compName = cNames[j++];
        const BPHRecoCandidate* cptr = ptr->getComp(compName).get();
        if ((ccrIter = ccRefMap.find(cptr)) == ccrIend) {
          if ((dauIter = daughMap.find(cptr)) != dauIend)
            cptr = dauIter->second;
          if ((jpoIter = jPsiOMap.find(cptr)) != jpoIend)
            cptr = jpoIter->second;
        }
        if ((ccrIter = ccRefMap.find(cptr)) != ccrIend) {
          compcc_ref cref = ccrIter->second;
          if (cref.isNonnull())
            cc.addUserData("refTo" + compName, cref);
        }
      }
      const BPHPlusMinusCandidate* pmp = dynamic_cast<const BPHPlusMinusCandidate*>(ptr.get());
      if (pmp != nullptr) {
        cc.addUserInt("cowboy", (pmp->isCowboy() ? +1 : -1));
        //        cc.addUserFloat(    "dca", pmp->cAppInRPhi().distance() );
      }
      if (writeVertex)
        cc.addUserData("vertex", ptr->vertex());
      if (ptr->isEmpty())
        continue;
      if (writeVertex)
        cc.addUserData("fitVertex", reco::Vertex(*ptr->topDecayVertex()));
      if (ptr->isValidFit()) {
        const RefCountedKinematicParticle kinPart = ptr->topParticle();
        const KinematicState kinStat = kinPart->currentState();
        cc.addUserFloat("fitMass", kinStat.mass());
        if (writeMomentum)
          cc.addUserData("fitMomentum", kinStat.kinematicParameters().momentum());
      }
    }
    typedef std::unique_ptr<pat::CompositeCandidateCollection> ccc_pointer;
    edm::OrphanHandle<pat::CompositeCandidateCollection> ccHandle = ev.put(ccc_pointer(ccList), name);
    for (i = 0; i < n; ++i) {
      const BPHRecoCandidate* ptr = list[i].get();
      edm::Ref<pat::CompositeCandidateCollection> ccRef(ccHandle, i);
      ccRefMap[ptr] = ccRef;
    }
    return ccHandle;
  }
};

#endif
