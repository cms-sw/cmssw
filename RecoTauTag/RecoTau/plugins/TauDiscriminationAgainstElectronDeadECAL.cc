/** \class TauDiscriminationAgainstElectronDeadECALBase
 *
 * Base class for producing PFTau and PATTau discriminators
 *
 * Flag tau candidates reconstructed near dead ECAL channels,
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

// helper function retrieve the correct tau type name
template <class TauType>
std::string getTauTypeNameString(bool capitalise) {
  // this generic one shoudl never be called.
  throw cms::Exception("TauDiscriminationAgainstElectronDeadECALBase")
      << "Unsupported TauType used. You must use either reco::PFTau or pat::Tau.";
}
// template specialiazation to get the correct (PF/PAT)Tau type names
template <>
std::string getTauTypeNameString<reco::PFTau>(bool capitalise) {
  return capitalise ? "PFRecoTau" : "pfRecoTau";
}
template <>
std::string getTauTypeNameString<pat::Tau>(bool capitalise) {
  return capitalise ? "PATTau" : "patTau";
}

template <class TauType, class TauDiscriminator>
class TauDiscriminationAgainstElectronDeadECALBase : public TauDiscriminationProducerBase<TauType, TauDiscriminator> {
public:
  typedef std::vector<TauType> TauCollection;
  typedef edm::Ref<TauCollection> TauRef;
  explicit TauDiscriminationAgainstElectronDeadECALBase(const edm::ParameterSet& cfg)
      : TauDiscriminationProducerBase<TauType, TauDiscriminator>::TauDiscriminationProducerBase(cfg),
        moduleLabel_(cfg.getParameter<std::string>("@module_label")),
        verbosity_(cfg.getParameter<int>("verbosity")),
        antiElectronDeadECAL_(cfg) {}
  ~TauDiscriminationAgainstElectronDeadECALBase() override {}

  void beginEvent(const edm::Event& evt, const edm::EventSetup& es) override { antiElectronDeadECAL_.beginEvent(es); }

  double discriminate(const TauRef& tau) const override {
    if (verbosity_) {
      edm::LogPrint(getTauTypeNameString<TauType>(true) + "AgainstEleDeadECAL")
          << "<" + getTauTypeNameString<TauType>(true) + "AgainstElectronDeadECAL::discriminate>:";
      edm::LogPrint(getTauTypeNameString<TauType>(true) + "AgainstEleDeadECAL") << " moduleLabel = " << moduleLabel_;
      edm::LogPrint(getTauTypeNameString<TauType>(true) + "AgainstEleDeadECAL")
          << " tau: Pt = " << tau->pt() << ", eta = " << tau->eta() << ", phi = " << tau->phi();
    }
    double discriminator = 1.;
    if (antiElectronDeadECAL_(tau.get())) {
      discriminator = 0.;
    }
    if (verbosity_) {
      edm::LogPrint(getTauTypeNameString<TauType>(true) + "AgainstEleDeadECAL")
          << "--> discriminator = " << discriminator;
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
void TauDiscriminationAgainstElectronDeadECALBase<TauType, TauDiscriminator>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  // {pfReco,pat}TauDiscriminationAgainstElectronDeadECAL
  edm::ParameterSetDescription desc;

  desc.add<double>("dR", 0.08);
  desc.add<unsigned int>("minStatus", 12);
  desc.add<bool>("extrapolateToECalEntrance", true);
  desc.add<int>("verbosity", 0);

  TauDiscriminationProducerBase<TauType, TauDiscriminator>::fillProducerDescriptions(
      desc);  // inherited from the base-class

  descriptions.add(getTauTypeNameString<TauType>(false) + "DiscriminationAgainstElectronDeadECAL", desc);  //base
}

typedef TauDiscriminationAgainstElectronDeadECALBase<reco::PFTau, reco::PFTauDiscriminator>
    PFRecoTauDiscriminationAgainstElectronDeadECAL;
typedef TauDiscriminationAgainstElectronDeadECALBase<pat::Tau, pat::PATTauDiscriminator>
    PATTauDiscriminationAgainstElectronDeadECAL;

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronDeadECAL);
DEFINE_FWK_MODULE(PATTauDiscriminationAgainstElectronDeadECAL);
