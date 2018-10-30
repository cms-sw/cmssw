#ifndef __RecoEgamma_ElectronIdentification_ElectronMVAEstimator_H__
#define __RecoEgamma_ElectronIdentification_ElectronMVAEstimator_H__

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/GBRForestTools.h"

#include <memory>
#include <string>

class ElectronMVAEstimator {
 public:
  struct Configuration{
         std::vector<std::string> vweightsfiles;
   };
  ElectronMVAEstimator();
  ElectronMVAEstimator(std::string fileName);
  ElectronMVAEstimator(const Configuration & );
  ~ElectronMVAEstimator() {;}
  double mva(const reco::GsfElectron& myElectron, int nvertices=0) const;

 private:
  const Configuration cfg_;
  void bindVariables(float vars[18]) const;
  
  std::vector<std::unique_ptr<const GBRForest> > gbr_;
  
};

#endif
