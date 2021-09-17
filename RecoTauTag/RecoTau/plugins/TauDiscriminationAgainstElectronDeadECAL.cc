/** \class TauDiscriminationAgainstElectronDeadECAL
 *
 * Template class for producing PFTau and PATTau discriminators which
 * flag tau candidates reconstructed near dead ECAL channels,
 * in order to reduce e -> tau fakes not rejected by anti-e MVA discriminator
 *
 * The motivation for this flag is this presentation:
 *   https://indico.cern.ch/getFile.py/access?contribId=0&resId=0&materialId=slides&confId=177223
 *
 * \authors Lauri Andreas Wendland,
 *          Christian Veelken
 *
 *
 *
 */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronDeadECAL.h"

template <class TauType, class TauDiscriminator>
class TauDiscriminationAgainstElectronDeadECAL : public TauDiscriminationProducerBase<TauType, TauDiscriminator> {
public:
  typedef std::vector<TauType> TauCollection;
  typedef edm::Ref<TauCollection> TauRef;
  explicit TauDiscriminationAgainstElectronDeadECAL(const edm::ParameterSet& cfg)
      : TauDiscriminationProducerBase<TauType, TauDiscriminator>::TauDiscriminationProducerBase(cfg),
        moduleLabel_(cfg.getParameter<std::string>("@module_label")),
        verbosity_(cfg.getParameter<int>("verbosity")),
        antiElectronDeadECAL_(cfg, edm::EDConsumerBase::consumesCollector()) {}
  ~TauDiscriminationAgainstElectronDeadECAL() override {}

  void beginEvent(const edm::Event& evt, const edm::EventSetup& es) override { antiElectronDeadECAL_.beginEvent(es); }

  double discriminate(const TauRef& tau) const override {
    if (verbosity_) {
      edm::LogPrint(this->getTauTypeString() + "AgainstEleDeadECAL")
          << "<" + this->getTauTypeString() + "AgainstElectronDeadECAL::discriminate>:";
      edm::LogPrint(this->getTauTypeString() + "AgainstEleDeadECAL") << " moduleLabel = " << moduleLabel_;
      edm::LogPrint(this->getTauTypeString() + "AgainstEleDeadECAL")
          << " tau: Pt = " << tau->pt() << ", eta = " << tau->eta() << ", phi = " << tau->phi();
    }
    double discriminator = 1.;
    if (antiElectronDeadECAL_(tau.get())) {
      discriminator = 0.;
    }
    if (verbosity_) {
      edm::LogPrint(this->getTauTypeString() + "AgainstEleDeadECAL") << "--> discriminator = " << discriminator;
    }
    return discriminator;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::string moduleLabel_;
  int verbosity_;

  AntiElectronDeadECAL antiElectronDeadECAL_;
};

template <class TauType, class TauDiscriminator>
void TauDiscriminationAgainstElectronDeadECAL<TauType, TauDiscriminator>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  // {pfReco,pat}TauDiscriminationAgainstElectronDeadECAL
  edm::ParameterSetDescription desc;

  desc.add<double>("dR", 0.08);
  desc.add<unsigned int>("minStatus", 12);
  desc.add<bool>("extrapolateToECalEntrance", true);
  desc.add<int>("verbosity", 0);

  TauDiscriminationProducerBase<TauType, TauDiscriminator>::fillProducerDescriptions(
      desc);  // inherited from the base-class

  descriptions.addWithDefaultLabel(desc);
}

typedef TauDiscriminationAgainstElectronDeadECAL<reco::PFTau, reco::PFTauDiscriminator>
    PFRecoTauDiscriminationAgainstElectronDeadECAL;
typedef TauDiscriminationAgainstElectronDeadECAL<pat::Tau, pat::PATTauDiscriminator>
    PATTauDiscriminationAgainstElectronDeadECAL;

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronDeadECAL);
DEFINE_FWK_MODULE(PATTauDiscriminationAgainstElectronDeadECAL);
