#ifndef CutBasedElectronID_H
#define CutBasedElectronID_H

#include "RecoEgamma/ElectronIdentification/interface/ElectronIDAlgo.h"

class CutBasedElectronID : public ElectronIDAlgo {

public:

  CutBasedElectronID(){};

  virtual ~CutBasedElectronID(){};

  void setup(const edm::ParameterSet& conf);
  double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&);
  int classify(const reco::GsfElectron*);
  
 private:
  
  std::string type_;
  std::string quality_;
  std::string version_;
  edm::ParameterSet cuts_;
  
};

#endif // CutBasedElectronID_H
