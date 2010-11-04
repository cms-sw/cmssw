#ifndef CutBasedElectronID_H
#define CutBasedElectronID_H

#include "RecoEgamma/ElectronIdentification/interface/ElectronIDAlgo.h"

class CutBasedElectronID : public ElectronIDAlgo {

public:

  CutBasedElectronID(){};

  virtual ~CutBasedElectronID() {};

  void setup(const edm::ParameterSet& conf);
  double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&);
  double cicSelection(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&);
  double robustSelection(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&);
  int classify(const reco::GsfElectron*);
  bool compute_cut(double x, double et, double cut_min, double cut_max, bool gtn=false);

 private:
  bool wantBinning_;
  bool newCategories_;
  std::string type_;
  std::string quality_;
  std::string version_;
  edm::InputTag verticesCollection_;
  edm::ParameterSet cuts_;

};

#endif // CutBasedElectronID_H
