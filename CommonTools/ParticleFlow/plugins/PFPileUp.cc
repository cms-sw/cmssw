// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/Association.h"

#include "CommonTools/ParticleFlow/interface/PFPileUpAlgo.h"

#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace std;
using namespace edm;
using namespace reco;

/**\class PFPileUp
\brief Identifies pile-up candidates from a collection of PFCandidates, and
produces the corresponding collection of PileUpCandidates.

\author Colin Bernet
\date   february 2008
\updated Florian Beaudette 30/03/2012

*/

class PFPileUp : public edm::stream::EDProducer<> {
public:
  typedef std::vector<edm::FwdPtr<reco::PFCandidate>> PFCollection;
  typedef edm::View<reco::PFCandidate> PFView;
  typedef std::vector<reco::PFCandidate> PFCollectionByValue;
  typedef edm::Association<reco::VertexCollection> CandToVertex;

  explicit PFPileUp(const edm::ParameterSet&);

  ~PFPileUp() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  PFPileUpAlgo pileUpAlgo_;

  /// PFCandidates to be analyzed
  edm::EDGetTokenT<PFCollection> tokenPFCandidates_;
  /// fall-back token
  edm::EDGetTokenT<PFView> tokenPFCandidatesView_;

  /// vertices
  edm::EDGetTokenT<reco::VertexCollection> tokenVertices_;

  /// enable PFPileUp selection
  bool enable_;

  /// verbose ?
  bool verbose_;

  /// use the closest z vertex if a track is not in a vertex
  bool checkClosestZVertex_;

  edm::EDGetTokenT<CandToVertex> tokenVertexAssociation_;
  edm::EDGetTokenT<edm::ValueMap<int>> tokenVertexAssociationQuality_;
  bool fUseVertexAssociation;
  int vertexAssociationQuality_;
  unsigned int fNumOfPUVtxsForCharged_;
  double fDzCutForChargedFromPUVtxs_;
};

PFPileUp::PFPileUp(const edm::ParameterSet& iConfig) {
  tokenPFCandidates_ = consumes<PFCollection>(iConfig.getParameter<InputTag>("PFCandidates"));
  tokenPFCandidatesView_ = mayConsume<PFView>(iConfig.getParameter<InputTag>("PFCandidates"));

  tokenVertices_ = consumes<VertexCollection>(iConfig.getParameter<InputTag>("Vertices"));

  fUseVertexAssociation = iConfig.getParameter<bool>("useVertexAssociation");
  vertexAssociationQuality_ = iConfig.getParameter<int>("vertexAssociationQuality");
  if (fUseVertexAssociation) {
    tokenVertexAssociation_ = consumes<CandToVertex>(iConfig.getParameter<edm::InputTag>("vertexAssociation"));
    tokenVertexAssociationQuality_ =
        consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("vertexAssociation"));
  }
  fNumOfPUVtxsForCharged_ = iConfig.getParameter<unsigned int>("NumOfPUVtxsForCharged");
  fDzCutForChargedFromPUVtxs_ = iConfig.getParameter<double>("DzCutForChargedFromPUVtxs");

  enable_ = iConfig.getParameter<bool>("enable");

  verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);

  checkClosestZVertex_ = iConfig.getParameter<bool>("checkClosestZVertex");

  // Configure the algo
  pileUpAlgo_.setVerbose(verbose_);
  pileUpAlgo_.setCheckClosestZVertex(checkClosestZVertex_);
  pileUpAlgo_.setNumOfPUVtxsForCharged(fNumOfPUVtxsForCharged_);
  pileUpAlgo_.setDzCutForChargedFromPUVtxs(fDzCutForChargedFromPUVtxs_);

  //produces<reco::PFCandidateCollection>();
  produces<PFCollection>();
  // produces< PFCollectionByValue > ();
}

PFPileUp::~PFPileUp() {}

void PFPileUp::produce(Event& iEvent, const EventSetup& iSetup) {
  //   LogDebug("PFPileUp")<<"START event: "<<iEvent.id().event()
  // 			 <<" in run "<<iEvent.id().run()<<endl;

  // get PFCandidates

  unique_ptr<PFCollection> pOutput(new PFCollection);

  unique_ptr<PFCollectionByValue> pOutputByValue(new PFCollectionByValue);

  if (enable_) {
    // get vertices
    Handle<VertexCollection> vertices;
    iEvent.getByToken(tokenVertices_, vertices);

    // get PF Candidates
    Handle<PFCollection> pfCandidates;
    PFCollection const* pfCandidatesRef = nullptr;
    PFCollection usedIfNoFwdPtrs;
    bool getFromFwdPtr = iEvent.getByToken(tokenPFCandidates_, pfCandidates);
    if (getFromFwdPtr) {
      pfCandidatesRef = pfCandidates.product();
    }
    // Maintain backwards-compatibility.
    // If there is no vector of FwdPtr<PFCandidate> found, then
    // make a dummy vector<FwdPtr<PFCandidate> > for the PFPileupAlgo,
    // set the pointer "pfCandidatesRef" to point to it, and
    // then we can pass it to the PFPileupAlgo.
    else {
      Handle<PFView> pfView;
      bool getFromView = iEvent.getByToken(tokenPFCandidatesView_, pfView);
      if (!getFromView) {
        throw cms::Exception(
            "PFPileUp is misconfigured. This needs to be either vector<FwdPtr<PFCandidate> >, or View<PFCandidate>");
      }
      for (edm::View<reco::PFCandidate>::const_iterator viewBegin = pfView->begin(),
                                                        viewEnd = pfView->end(),
                                                        iview = viewBegin;
           iview != viewEnd;
           ++iview) {
        usedIfNoFwdPtrs.push_back(
            edm::FwdPtr<reco::PFCandidate>(pfView->ptrAt(iview - viewBegin), pfView->ptrAt(iview - viewBegin)));
      }
      pfCandidatesRef = &usedIfNoFwdPtrs;
    }

    if (pfCandidatesRef == nullptr) {
      throw cms::Exception(
          "Something went dreadfully wrong with PFPileUp. pfCandidatesRef should never be zero, so this is a logic "
          "error.");
    }

    if (fUseVertexAssociation) {
      const edm::Association<reco::VertexCollection>& associatedPV = iEvent.get(tokenVertexAssociation_);
      const edm::ValueMap<int>& associationQuality = iEvent.get(tokenVertexAssociationQuality_);
      PFCollection pfCandidatesFromPU;
      for (auto& p : (*pfCandidatesRef)) {
        const reco::VertexRef& PVOrig = associatedPV[p];
        int quality = associationQuality[p];
        if (PVOrig.isNonnull() && (PVOrig.key() > 0) && (quality >= vertexAssociationQuality_))
          pfCandidatesFromPU.push_back(p);
      }
      pOutput->insert(pOutput->end(), pfCandidatesFromPU.begin(), pfCandidatesFromPU.end());
    } else {
      pileUpAlgo_.process(*pfCandidatesRef, *vertices);
      pOutput->insert(
          pOutput->end(), pileUpAlgo_.getPFCandidatesFromPU().begin(), pileUpAlgo_.getPFCandidatesFromPU().end());
    }
    // for ( PFCollection::const_iterator byValueBegin = pileUpAlgo_.getPFCandidatesFromPU().begin(),
    // 	    byValueEnd = pileUpAlgo_.getPFCandidatesFromPU().end(), ibyValue = byValueBegin;
    // 	  ibyValue != byValueEnd; ++ibyValue ) {
    //   pOutputByValue->push_back( **ibyValue );
    // }

  }  // end if enabled
  // outsize of the loop to fill the collection anyway even when disabled
  iEvent.put(std::move(pOutput));
  // iEvent.put(std::move(pOutputByValue));
}

void PFPileUp::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("PFCandidates", edm::InputTag("particleFlowTmpPtrs"));
  desc.add<edm::InputTag>("Vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("enable", true);
  desc.addUntracked<bool>("verbose", false);
  desc.add<bool>("checkClosestZVertex", true);
  desc.add<bool>("useVertexAssociation", false);
  desc.add<int>("vertexAssociationQuality", 0);
  desc.add<edm::InputTag>("vertexAssociation", edm::InputTag(""));
  desc.add<unsigned int>("NumOfPUVtxsForCharged", 0);
  desc.add<double>("DzCutForChargedFromPUVtxs", .2);
  descriptions.add("pfPileUp", desc);
}

DEFINE_FWK_MODULE(PFPileUp);
