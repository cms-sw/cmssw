#include "RecoMuon/MuonIsolation/interface/IsolatorByDeposit.h"

using std::vector;
using reco::IsoDeposit;
using namespace muonisolation;

IsolatorByDeposit::IsolatorByDeposit(float conesize, const vector<double>& weights) 
  : theConeSizeFunction(nullptr), theConeSize(conesize), theWeights(weights)
{ 
  theDepThresholds = std::vector<double>(weights.size(), -1e12);
}

IsolatorByDeposit::IsolatorByDeposit(const ConeSizeFunction * conesize, const vector<double>& weights) 
  : theConeSizeFunction(conesize), theConeSize(0.), theWeights(weights)
{ 
  theDepThresholds = std::vector<double>(weights.size(), -1e12);
}

IsolatorByDeposit::IsolatorByDeposit(float conesize, const vector<double>& weights, const vector<double>& dThresh) 
  : theConeSizeFunction(nullptr), theConeSize(conesize), theWeights(weights),
    theDepThresholds(dThresh)
{ }

IsolatorByDeposit::IsolatorByDeposit(const ConeSizeFunction * conesize, 
				     const vector<double>& weights, const vector<double>& dThresh) 
  : theConeSizeFunction(conesize), theConeSize(0.), theWeights(weights),
    theDepThresholds(dThresh)
{ }

MuIsoBaseIsolator::Result IsolatorByDeposit::result(const DepositContainer& deposits, const edm::Event*) const{
  if (deposits.empty()) return Result(resultType());

  // To determine the threshold, the direction of the cone of the first
  // set of deposits is used.
  // For algorithms where different cone axis definitions are used
  // for different types deposits (eg. HCAL and ECAL deposits for
  // calorimeter isolation), the first one is used to determine the threshold
  // value!
  float eta = deposits.front().dep->eta();
  float pt = deposits.front().dep->candEnergy();
  float dr= coneSize(eta,pt);
  float sumDep = weightedSum(deposits,dr);


  Result res(resultType()); 
  res.valFloat = sumDep;
  return res;
}

double
IsolatorByDeposit::weightedSum(const DepositContainer& deposits,
			       float dRcone) const {
  double sumDep=0;

  assert(deposits.size()==theWeights.size());

  vector<double>::const_iterator w = theWeights.begin();
  vector<double>::const_iterator dThresh = theDepThresholds.begin();

  typedef DepositContainer::const_iterator DI;
  for (DI dep = deposits.begin(), depEnd = deposits.end(); dep != depEnd; ++dep) {
    if (dep->vetos != nullptr){
      sumDep += dep->dep->depositAndCountWithin(dRcone, *dep->vetos, (*dThresh)).first * (*w);
    } else {
      sumDep += dep->dep->depositAndCountWithin(dRcone, Vetos(), (*dThresh)).first * (*w);
    }
//  cout << "IsolatorByDeposit: type = " << (*dep)->type() << " weight = " << (*w) << endl;
    w++;
    dThresh++;
  }
  return sumDep;
}
