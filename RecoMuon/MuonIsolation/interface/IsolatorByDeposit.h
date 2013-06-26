#ifndef MuonIsolation_IsolatorByDeposit_H
#define MuonIsolation_IsolatorByDeposit_H

/** \class IsolatorByDeposit
 *  Define the isolation variable simply as the deposit within the cone.
 *  This is the simplest definition possible, for isolation algorithms
 *  where the cut is directly on, e.g., the deposited energy.
 *
 *  $Date: 2012/01/27 06:02:19 $
 *  $Revision: 1.3 $
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

  //! construct with no addtnl thresholds on deposits
  IsolatorByDeposit(float conesize, const  std::vector<double> & weights);
  IsolatorByDeposit(const ConeSizeFunction * conesize, const std::vector<double> & weights);

  //! construct with non-default thresholds per deposit
  IsolatorByDeposit(float conesize, 
		    const std::vector<double> & weights, const std::vector<double>& thresh);
  IsolatorByDeposit(const ConeSizeFunction * conesize, 
		    const std::vector<double> & weights, const std::vector<double>& thresh);

  virtual ~IsolatorByDeposit() {}

  //! Set the weights for summing deposits of different types
  virtual void setWeights(const std::vector<double>& weights) {theWeights=weights;}

  //! Compute the deposit within the cone and return the isolation result
  virtual Result result(const DepositContainer& deposits, const edm::Event* = 0) const;

  //! Compute the count of deposit within the cone and return the isolation result
/*   virtual int resultInt(DepositContainer deposits) const; */



  void setConeSize(float conesize) { theConeSize = conesize; theConeSizeFunction = 0;} 

  void setConeSize(ConeSizeFunction * conesize) { theConeSizeFunction = conesize; }


  //! Get the cone size
  virtual float coneSize(float eta, float pT) const {
    return theConeSizeFunction ? theConeSizeFunction->coneSize(eta,pT) : theConeSize;
  }

  virtual ResultType resultType() const { return ISOL_FLOAT_TYPE;}


private:
  // Compute the weighted sum of deposits of different type within dRcone
  double weightedSum(const DepositContainer& deposits, float dRcone) const;

private:
  const ConeSizeFunction * theConeSizeFunction;
  float theConeSize;
  std::vector<double> theWeights;
  std::vector<double> theDepThresholds;
};
}

#endif
