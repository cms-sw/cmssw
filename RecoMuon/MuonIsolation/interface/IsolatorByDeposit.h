#ifndef MuonIsolation_IsolatorByDeposit_H
#define MuonIsolation_IsolatorByDeposit_H

/** \class IsolatorByDeposit
 *  Define the isolation variable simply as the deposit within the cone.
 *  This is the simplest definition possible, for isolation algorithms
 *  where the cut is directly on, e.g., the deposited energy.
 *
 *  $Date: 2004/10/26 14:23:21 $
 *  $Revision: 1.4 $
 *  \author M. Konecki, N. Amapane
 */

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"
#include <vector>


namespace muonisolation {
class IsolatorByDeposit : public MuIsoBaseIsolator {
public:
  typedef MuIsoBaseIsolator::DepositContainer DepositContainer;

  struct ConeSizeFunction {
   virtual float  coneSize( float eta, float pt) const = 0;
  };

  IsolatorByDeposit(float conesize, std::vector<double> & weights);
  IsolatorByDeposit(const ConeSizeFunction * conesize, std::vector<double> & weights);

  virtual ~IsolatorByDeposit() {}

  /// Set the weights for summing deposits of different types
  virtual void setWeights(const std::vector<double>& weights) {theWeights=weights;}

  /// Compute the deposit within the cone and return the isolation result
  virtual float result(DepositContainer deposits) const;

  void setConeSize(float conesize) { theConeSize = conesize; theConeSizeFunction = 0;} 

  void setConeSize(ConeSizeFunction * conesize) { theConeSizeFunction = conesize; }


  /// Get the cone size
  virtual float coneSize(float eta, float pT) const {
    return theConeSizeFunction ? theConeSizeFunction->coneSize(eta,pT) : theConeSize;
  }

private:
  // Compute the weighted sum of deposits of different type within dRcone
  double weightedSum(const DepositContainer& deposits, float dRcone) const;

private:
  const ConeSizeFunction * theConeSizeFunction;
  float theConeSize;
  std::vector<double> theWeights;
};
}

#endif
