#ifndef MuonIsolation_IsolatorByDepositCount_H
#define MuonIsolation_IsolatorByDepositCount_H

/** \class IsolatorByDepositCount
 *  Define the isolation variable simply as the deposit within the cone.
 *  This is the simplest definition possible, for isolation algorithms
 *  where the cut is directly on, e.g., the deposited energy.
 *
 *  \author M. Konecki, N. Amapane
 */

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"
#include <vector>


namespace muonisolation {
class IsolatorByDepositCount : public MuIsoBaseIsolator {
public:
  typedef MuIsoBaseIsolator::DepositContainer DepositContainer;

  struct ConeSizeFunction {
   virtual ~ConeSizeFunction() = default;
   virtual float  coneSize( float eta, float pt) const = 0;
  };

  //! construct with non-default thresholds per deposit
  IsolatorByDepositCount(float conesize, const std::vector<double>& thresh);
  IsolatorByDepositCount(const ConeSizeFunction * conesize, const std::vector<double>& thresh);

  ~IsolatorByDepositCount() override = default;

  //! Compute the deposit within the cone and return the isolation result
  Result result(const DepositContainer& deposits, const edm::Event* = nullptr) const override;


  void setConeSize(float conesize) { theConeSize = conesize; theConeSizeFunction = nullptr;} 

  void setConeSize(ConeSizeFunction * conesize) { theConeSizeFunction = conesize; }


  //! Get the cone size
  virtual float coneSize(float eta, float pT) const {
    return theConeSizeFunction ? theConeSizeFunction->coneSize(eta,pT) : theConeSize;
  }

  ResultType resultType() const override { return ISOL_INT_TYPE;}


private:
  const ConeSizeFunction * theConeSizeFunction;
  float theConeSize;
  std::vector<double> theDepThresholds;
};
}

#endif
