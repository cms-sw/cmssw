#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

using namespace reco;

MuIsoDeposit::MuIsoDeposit() :
eta_(0), phi_(0), depositFromMuon(0)
{};

MuIsoDeposit::MuIsoDeposit(const std::string type, double eta, double phi) : 
type_(type), eta_(eta), phi_(phi), depositFromMuon(0)
{};

void MuIsoDeposit::addDeposit(double dr, double value){
  deposits.insert(std::pair<double,double>(dr,value));
}

double MuIsoDeposit::depositWithin(double coneSize) const {
  double result = 0;
  if (deposits.size()!=0) {
    // Note that upper_bound corresponds to dr<=coneSize 
    // while lower_bound would correspond to dr<coneSize
    std::multimap<double,double>::const_iterator loc=deposits.upper_bound(coneSize);
    while (loc!=deposits.begin()) { // No deposit inside
      result += (--loc)->second;
    }
  }
  return result;
}
