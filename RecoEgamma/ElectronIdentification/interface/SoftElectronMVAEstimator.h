#ifndef __ElectronIdentification_SoftElectronMVAEstimator_H__
#define __ElectronIdentification_SoftElectronMVAEstimator_H__

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include <memory>
#include <string>

class SoftElectronMVAEstimator {
public:
  constexpr static unsigned int ExpectedNBins = 1;

  struct Configuration {
    std::vector<std::string> vweightsfiles;
  };
  SoftElectronMVAEstimator(const Configuration&);
  ~SoftElectronMVAEstimator();
  double mva(const reco::GsfElectron& myElectron, const reco::VertexCollection&) const;

private:
  void bindVariables(float vars[25]) const;
  void init();

private:
  const Configuration cfg_;
  std::vector<std::unique_ptr<const GBRForest> > gbr_;
};

#endif
