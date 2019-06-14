#ifndef PhysicsTools_HepMCCandAlgos_PDFWeightsHelper_h
#define PhysicsTools_HepMCCandAlgos_PDFWeightsHelper_h

#include <Eigen/Dense>

#include <iostream>

#include "FWCore/ParameterSet/interface/FileInPath.h"

class PDFWeightsHelper {
public:
  PDFWeightsHelper();

  void Init(unsigned int nreplicas, unsigned int neigenvectors, const edm::FileInPath &incsv);
  void DoMC2Hessian(double nomweight, const double *inweights, double *outweights) const;

  unsigned int neigenvectors() const { return transformation_.cols(); }

protected:
  Eigen::MatrixXd transformation_;
};
#endif
