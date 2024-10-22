#ifndef PhysicsTools_NanoAODTools_mhtjuProducerCppWorker_h
#define PhysicsTools_NanoAODTools_mhtjuProducerCppWorker_h
// Example of a C++ helper function that can be called from a python Module.
// see python/postprocessing/examples/mhtjuProducerCpp.py for more details.

#include <utility>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TLorentzVector.h>

class MhtjuProducerCppWorker {
public:
  MhtjuProducerCppWorker() {}

  void setJets(TTreeReaderValue<unsigned> *nJet_, TTreeReaderArray<float> *Jet_pt_, TTreeReaderArray<float> *Jet_phi_) {
    nJet = nJet_;
    Jet_pt = Jet_pt_;
    Jet_phi = Jet_phi_;
  }

  std::pair<float, float> getHT();

private:
  TTreeReaderValue<unsigned> *nJet = nullptr;
  TTreeReaderArray<float> *Jet_pt = nullptr;
  TTreeReaderArray<float> *Jet_phi = nullptr;
};

#endif
