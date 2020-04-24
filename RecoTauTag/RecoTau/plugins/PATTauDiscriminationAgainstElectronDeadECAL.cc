
/** \class PATTauDiscriminationAgainstElectronDeadECAL
 *
 * Flag tau candidates reconstructed near dead ECAL channels,
 * in order to reduce e -> tau fakes not rejected by anti-e MVA discriminator
 *
 * Adopted from RecoTauTag/RecoTau/plugins/PFRecoTauDiscriminationAgainstElectronDeadECAL.cc
 * to enable computation of the discriminator on MiniAOD
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

class PATTauDiscriminationAgainstElectronDeadECAL : public PATTauDiscriminationProducerBase
{
 public:
  explicit PATTauDiscriminationAgainstElectronDeadECAL(const edm::ParameterSet& cfg)
      : PATTauDiscriminationProducerBase(cfg),
        moduleLabel_(cfg.getParameter<std::string>("@module_label")),
        antiElectronDeadECAL_(cfg) 
  {}
  ~PATTauDiscriminationAgainstElectronDeadECAL() override 
  {}

  void beginEvent(const edm::Event& evt, const edm::EventSetup& es) override 
  { 
    antiElectronDeadECAL_.beginEvent(es); 
  }

  double discriminate(const TauRef& tau) const override 
  {
    double discriminator = 1.;
    if ( tau->leadChargedHadrCand().isNonnull() ) {
      if ( antiElectronDeadECAL_(tau->leadChargedHadrCand().get()) ) {
        discriminator = 0.;
      }
    }
    return discriminator;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  std::string moduleLabel_;

  AntiElectronDeadECAL antiElectronDeadECAL_;
};

void PATTauDiscriminationAgainstElectronDeadECAL::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  // patTauDiscriminationAgainstElectronDeadECAL
  edm::ParameterSetDescription desc;
  
  desc.add<double>("dR", 0.08);
  desc.add<unsigned int>("minStatus", 12);
  desc.add<int>("verbosity", 0);

  fillProducerDescriptions(desc); // inherited from the base-class

  descriptions.add("patTauDiscriminationAgainstElectronDeadECAL", desc);
}

DEFINE_FWK_MODULE(PATTauDiscriminationAgainstElectronDeadECAL);
