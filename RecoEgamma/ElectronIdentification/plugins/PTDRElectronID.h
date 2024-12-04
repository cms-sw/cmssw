#ifndef PTDRElectronID_H
#define PTDRElectronID_H

#include "ElectronIDAlgo.h"

class PTDRElectronID : public ElectronIDAlgo {
public:
  PTDRElectronID(const edm::ParameterSet& conf);

  double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) const override;

private:
  struct Cuts {
    Cuts(edm::ParameterSet const&);
    std::vector<double> EoverPInMax_;
    std::vector<double> EoverPInMin_;

    std::vector<double> deltaEtaIn_;

    std::vector<double> deltaPhiIn_;

    std::vector<double> HoverE_;

    std::vector<double> EoverPOutMax_;
    std::vector<double> EoverPOutMin_;

    std::vector<double> deltaPhiOut_;

    std::vector<double> invEMinusInvP_;

    std::vector<double> bremFraction_;

    std::vector<double> E9overE25_;

    std::vector<double> sigmaEtaEtaMax_;
    std::vector<double> sigmaEtaEtaMin_;

    std::vector<double> sigmaPhiPhiMin_;
    std::vector<double> sigmaPhiPhiMax_;

    bool useEoverPIn_;
    bool useDeltaEtaIn_;
    bool useDeltaPhiIn_;
    bool useHoverE_;
    bool useE9overE25_;
    bool useEoverPOut_;
    bool useDeltaPhiOut_;
    bool useInvEMinusInvP_;
    bool useBremFraction_;
    bool useSigmaEtaEta_;
    bool useSigmaPhiPhi_;
    bool acceptCracks_;
  } cuts_;
};

#endif  // PTDRElectronID_H
