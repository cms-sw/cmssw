#ifndef CutBasedElectronID_H
#define CutBasedElectronID_H

#include "ElectronIDAlgo.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class CutBasedElectronID : public ElectronIDAlgo {
public:
  CutBasedElectronID(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);

  ~CutBasedElectronID() override{};

  void setup(const edm::ParameterSet& conf) override;
  double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) override;
  double cicSelection(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&);
  double robustSelection(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&);
  int classify(const reco::GsfElectron*);
  bool compute_cut(double x, double et, double cut_min, double cut_max, bool gtn = false);

private:
  bool wantBinning_;
  bool newCategories_;
  std::string type_;
  std::string quality_;
  std::string version_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > verticesCollection_;
  edm::ParameterSet cuts_;
};

#endif  // CutBasedElectronID_H
