#ifndef ClassBasedElectronID_H
#define ClassBasedElectronID_H

#include "ElectronIDAlgo.h"

class ClassBasedElectronID : public ElectronIDAlgo {
public:
  ClassBasedElectronID(){};

  ~ClassBasedElectronID() override{};

  void setup(const edm::ParameterSet& conf) override;
  double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) override;

private:
  std::string quality_;
  edm::ParameterSet cuts_;
};

#endif  // ClassBasedElectronID_H
