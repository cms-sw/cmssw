
/** \class PFRecoTauDiscriminationAgainstElectronDeadECAL
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
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronDeadECAL.h"

class PFRecoTauDiscriminationAgainstElectronDeadECAL : public PFTauDiscriminationProducerBase 
{
 public:
  explicit PFRecoTauDiscriminationAgainstElectronDeadECAL(const edm::ParameterSet& cfg)
      : PFTauDiscriminationProducerBase(cfg),
        moduleLabel_(cfg.getParameter<std::string>("@module_label")),
        antiElectronDeadECAL_(cfg) 
  {}
  ~PFRecoTauDiscriminationAgainstElectronDeadECAL() override 
  {}

  void beginEvent(const edm::Event& evt, const edm::EventSetup& es) override 
  { 
    antiElectronDeadECAL_.beginEvent(es); 
  }

  double discriminate(const reco::PFTauRef& tau) const override 
  {
    double discriminator = 1.;
    const reco::Candidate* leadPFChargedHadron = ( tau->leadPFChargedHadrCand().isNonnull() ) ? tau->leadPFChargedHadrCand().get() : nullptr;
    if ( antiElectronDeadECAL_(tau->p4(), leadPFChargedHadron) ) {
      discriminator = 0.;
    }
    return discriminator;
  }

 private:
  std::string moduleLabel_;

  AntiElectronDeadECAL antiElectronDeadECAL_;
};

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronDeadECAL);
