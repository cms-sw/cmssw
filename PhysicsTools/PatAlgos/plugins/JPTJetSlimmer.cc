
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
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

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    auto jptJets = std::make_unique<reco::JPTJetCollection>();
    auto caloJets = std::make_unique<reco::CaloJetCollection>();

    edm::RefProd<reco::CaloJetCollection> pOut1RefProd = iEvent.getRefBeforePut<reco::CaloJetCollection>();
    edm::Ref<reco::CaloJetCollection>::key_type idxCaloJet = 0;

    auto const& h_calojets = iEvent.get(srcCaloToken_);
    auto const& h_jets = iEvent.get(srcToken_);

    for (auto const& ijet : h_jets) {
      if (selector_(ijet)) {
        // Add specific : only reference to CaloJet collection. It is necessary for
        // recalibration JPTJet at MiniAod.
        const edm::RefToBase<reco::Jet>& rawcalojet = ijet.getCaloJetRef();
        int icalo = -1;
        int i = 0;
        for (auto const& icjet : h_calojets) {
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
          reco::CaloJet const* rawcalojetc = dynamic_cast<reco::CaloJet const*>(&*rawcalojet);
          caloJets->push_back(*rawcalojetc);
          const reco::Candidate::Point& orivtx = ijet.vertex();
          reco::JPTJet newJPTJet(ijet.p4(), orivtx, tmp_specific, ijet.getJetConstituents());
          float jetArea = ijet.jetArea();
          newJPTJet.setJetArea(fabs(jetArea));
          jptJets->push_back(newJPTJet);
        }
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
void JPTJetSlimmer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // slimmedJPTJets
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("JetPlusTrackZSPCorJetAntiKt4"));
  desc.add<edm::InputTag>("srcCalo", edm::InputTag("slimmedCaloJets"));
  desc.add<std::string>("cut", "pt>20");
  descriptions.add("slimmedJPTJets", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JPTJetSlimmer);
