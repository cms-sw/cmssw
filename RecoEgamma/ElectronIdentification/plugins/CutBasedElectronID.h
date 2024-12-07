#ifndef CutBasedElectronID_H
#define CutBasedElectronID_H

#include "ElectronIDAlgo.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <memory>

class CutBasedElectronIDClassbasedVersionStrategyBase {
public:
  virtual ~CutBasedElectronIDClassbasedVersionStrategyBase() = default;

  virtual double selection(const reco::GsfElectron& electron, const reco::VertexCollection* e) const = 0;
  virtual bool needsVertices() const = 0;
};

class CutBasedElectronIDRobustVersionStrategyBase {
public:
  virtual ~CutBasedElectronIDRobustVersionStrategyBase() = default;

  virtual double sigmaee(const reco::GsfElectron& electron) const;
  virtual double ip(const reco::GsfElectron& electron,
                    edm::Handle<reco::BeamSpot>,
                    const reco::VertexCollection*) const;
  virtual bool needsBeamSpot() const;
  virtual bool needsVertices() const;
  struct Iso {
    double ecal;
    double hcal;
    double hcal1;
    double hcal2;
  };
  virtual Iso iso(const reco::GsfElectron& electron) const;
};

class CutBasedElectronID : public ElectronIDAlgo {
public:
  CutBasedElectronID(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);

  double result(const reco::GsfElectron* electron, const edm::Event& e, const edm::EventSetup& es) const override;

private:
  double cicSelection(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) const;
  double robustSelection(const reco::GsfElectron*, const edm::Event&, const edm::EventSetup&) const;

  std::string type_;
  std::string quality_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > verticesCollection_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpot_;
  std::vector<double> barrelCuts_;
  std::vector<double> endcapCuts_;

  std::unique_ptr<CutBasedElectronIDClassbasedVersionStrategyBase const> classbasedVersionStrategy_;
  std::unique_ptr<CutBasedElectronIDRobustVersionStrategyBase const> robustVersionStrategy_;
};

#endif  // CutBasedElectronID_H
