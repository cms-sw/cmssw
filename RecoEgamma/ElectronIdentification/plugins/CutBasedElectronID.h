#ifndef CutBasedElectronID_H
#define CutBasedElectronID_H

#include "ElectronIDAlgo.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class CutBasedElectronID : public ElectronIDAlgo {
public:
  CutBasedElectronID(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);

  double result(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) const override;

private:
  double cicSelection(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) const;
  double robustSelection(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) const;
  int classify(const reco::GsfElectron*) const;
  bool compute_cut(double x, double et, double cut_min, double cut_max, bool gtn = false) const;

  bool wantBinning_;
  bool newCategories_;
  std::string type_;
  std::string quality_;
  std::string version_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > verticesCollection_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpot_;
  edm::ParameterSet cuts_;
};

#endif  // CutBasedElectronID_H
