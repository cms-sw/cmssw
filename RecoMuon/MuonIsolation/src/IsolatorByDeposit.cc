#include "RecoMuon/MuonIsolation/interface/IsolatorByDeposit.h"

using std::vector;
using reco::MuIsoDeposit;
using namespace muonisolation;

IsolatorByDeposit::IsolatorByDeposit(float conesize, vector<double>& weights) 
: theConeSizeFunction(0), theConeSize(conesize), theWeights(weights)
{ }

IsolatorByDeposit::IsolatorByDeposit(const ConeSizeFunction * conesize, vector<double>& weights) 
: theConeSizeFunction(conesize), theConeSize(0.), theWeights(weights)
{ }

float IsolatorByDeposit::result(DepositContainer deposits) const{
  if (deposits.empty()) return -999.;

  // To determine the threshold, the direction of the cone of the first
  // set of deposits is used.
  // For algorithms where different cone axis definitions are used
  // for different types deposits (eg. HCAL and ECAL deposits for
  // calorimeter isolation), the first one is used to determine the threshold
  // value!
  float eta = deposits.front()->eta();
  float pt = deposits.front()->muonEnergy();
  float dr= coneSize(eta,pt);
  float sumDep = weightedSum(deposits,dr);

  return sumDep;
}

double
IsolatorByDeposit::weightedSum(const DepositContainer& deposits,
                            float dRcone) const {
  double sumDep=0;

  assert(deposits.size()==theWeights.size());

  vector<double>::const_iterator w = theWeights.begin();
  typedef DepositContainer::const_iterator DI;
  for (DI dep = deposits.begin(), depEnd = deposits.end(); dep != depEnd; ++dep) {
    sumDep += (*dep)->depositWithin(dRcone) * (*w);
//  cout << "IsolatorByDeposit: type = " << (*dep)->type() << " weight = " << (*w) << endl;
    w++;
  }
  return sumDep;
}
