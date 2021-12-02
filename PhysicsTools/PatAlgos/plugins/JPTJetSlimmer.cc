
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

class JPTJetSlimmer : public edm::stream::EDProducer<> {
public:
  JPTJetSlimmer(edm::ParameterSet const& params)
      : srcToken_(consumes<edm::View<reco::JPTJet> >(params.getParameter<edm::InputTag>("src"))),
        srcCaloToken_(consumes<edm::View<reco::CaloJet> >(params.getParameter<edm::InputTag>("srcCalo"))),
        cut_(params.getParameter<std::string>("cut")),
        selector_(cut_) {
    produces<reco::JPTJetCollection>();
    produces<reco::CaloJetCollection>();
  }

  ~JPTJetSlimmer() override {}

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    auto jptJets = std::make_unique<reco::JPTJetCollection>();
    auto caloJets = std::make_unique<reco::CaloJetCollection>();

    edm::RefProd<reco::CaloJetCollection> pOut1RefProd = iEvent.getRefBeforePut<reco::CaloJetCollection>();
    edm::Ref<reco::CaloJetCollection>::key_type idxCaloJet = 0;

    edm::Handle<edm::View<reco::CaloJet> > h_calojets;
    iEvent.getByToken(srcCaloToken_, h_calojets);

    edm::Handle<edm::View<reco::JPTJet> > h_jets;
    iEvent.getByToken(srcToken_, h_jets);

    for (auto const& ijet : *h_jets) {
      if (selector_(ijet)) {
        // Add specific : only reference to CaloJet collection. It is necessary for
        // recalibration JPTJet at MiniAod.
        const edm::RefToBase<reco::Jet>& jptjetRef = ijet.getCaloJetRef();
        reco::CaloJet const* rawcalojet = dynamic_cast<reco::CaloJet const*>(&*jptjetRef);
        int icalo = -1;
        int i = 0;
        for (auto const& icjet : *h_calojets) {
          double dr2 = deltaR2(icjet, *rawcalojet);
          if (dr2 <= 0.001) {
            icalo = i;
          }
          i++;
        }
        reco::JPTJet::Specific tmp_specific;
        if (icalo < 0) {
          // Add reference to the created CaloJet collection
          reco::CaloJetRef myjet(pOut1RefProd, idxCaloJet++);
          tmp_specific.theCaloJetRef = edm::RefToBase<reco::Jet>(myjet);
          caloJets->push_back(*rawcalojet);
        } else {
          //  Add reference to existing slimmedCaloJet Collection to JPTJet
          tmp_specific.theCaloJetRef = edm::RefToBase<reco::Jet>(h_calojets->refAt(icalo));
          ;
        }
        const reco::Candidate::Point& orivtx = ijet.vertex();
        reco::JPTJet newJPTJet(ijet.p4(), orivtx, tmp_specific, ijet.getJetConstituents());
        float jetArea = ijet.jetArea();
        newJPTJet.setJetArea(fabs(jetArea));
        jptJets->push_back(newJPTJet);
      }
    }
    iEvent.put(std::move(caloJets));
    iEvent.put(std::move(jptJets));
  }

protected:
  const edm::EDGetTokenT<edm::View<reco::JPTJet> > srcToken_;
  const edm::EDGetTokenT<edm::View<reco::CaloJet> > srcCaloToken_;
  const std::string cut_;
  const StringCutObjectSelector<reco::Jet> selector_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JPTJetSlimmer);
