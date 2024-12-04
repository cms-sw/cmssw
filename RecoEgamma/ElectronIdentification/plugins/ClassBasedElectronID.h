#ifndef ClassBasedElectronID_H
#define ClassBasedElectronID_H

#include "ElectronIDAlgo.h"

class ClassBasedElectronID : public ElectronIDAlgo {
public:
  explicit ClassBasedElectronID(const edm::ParameterSet&);

  double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) const override;

private:
  struct Cuts {
    Cuts(edm::ParameterSet const&);
    std::vector<double> deltaEtaIn_;
    std::vector<double> sigmaIetaIetaMax_;
    std::vector<double> sigmaIetaIetaMin_;
    std::vector<double> HoverE_;
    std::vector<double> EoverPOutMax_;
    std::vector<double> EoverPOutMin_;
    std::vector<double> deltaPhiInChargeMax_;
    std::vector<double> deltaPhiInChargeMin_;
  } cuts_;
};

#endif  // ClassBasedElectronID_H
