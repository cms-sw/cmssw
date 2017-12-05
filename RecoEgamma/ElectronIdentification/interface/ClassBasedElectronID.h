#ifndef ClassBasedElectronID_H
#define ClassBasedElectronID_H

#include "RecoEgamma/ElectronIdentification/interface/ElectronIDAlgo.h"

class ClassBasedElectronID : public ElectronIDAlgo {

public:

  ClassBasedElectronID(){};

  ~ClassBasedElectronID() override{};

  void setup(const edm::ParameterSet& conf) override;
  double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) override;

 private:

  std::string quality_;

  /*  std::vector<int> useEoverPIn_; */
  /*   std::vector<int> useDeltaEtaIn_; */
  /*   std::vector<int> useDeltaPhiIn_; */
  /*   std::vector<int> useHoverE_; */
  /*   std::vector<int> useE9overE25_; */
  /*   std::vector<int> useEoverPOut_; */
  /*   std::vector<int> useDeltaPhiOut_; */
  /*   std::vector<int> useInvEMinusInvP_; */
  /*   std::vector<int> useBremFraction_; */
  /*   std::vector<int> useSigmaEtaEta_; */
  /*   std::vector<int> useSigmaPhiPhi_; */
  /*   std::vector<int> acceptCracks_; */
  
  edm::ParameterSet cuts_;

  //int variables_;
};

#endif // ClassBasedElectronID_H
