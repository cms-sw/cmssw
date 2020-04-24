#include "RecoMuon/MuonIsolation/interface/IsolatorByDepositCount.h"

using std::vector;
using reco::IsoDeposit;
using namespace muonisolation;

IsolatorByDepositCount::IsolatorByDepositCount(float conesize, const vector<double>& dThresh) 
  : theConeSizeFunction(nullptr), theConeSize(conesize), theDepThresholds(dThresh)
{ }

IsolatorByDepositCount::IsolatorByDepositCount(const ConeSizeFunction * conesize, 
					       const vector<double>& dThresh) 
  : theConeSizeFunction(conesize), theConeSize(0.), theDepThresholds(dThresh)
{ }

MuIsoBaseIsolator::Result IsolatorByDepositCount::result(const DepositContainer& deposits, const edm::Event*) const{
  if (deposits.empty()) return Result(resultType());
  if (deposits.size()>1){    return Result(ISOL_INVALID_TYPE);
  }

  // To determine the threshold, the direction of the cone of the first
  // set of deposits is used.
  // For algorithms where different cone axis definitions are used
  // for different types deposits (eg. HCAL and ECAL deposits for
  // calorimeter isolation), the first one is used to determine the threshold
  // value!
  float eta = deposits.front().dep->eta();
  float pt = deposits.front().dep->candEnergy();
  float dr= coneSize(eta,pt);
  DepositAndVetos depVet = deposits.front();
  std::pair<double, int> sumAndCount = depVet.dep->depositAndCountWithin(dr, *depVet.vetos, theDepThresholds.front());


  Result res(resultType()); 
  res.valInt = sumAndCount.second;
  return res;
}
