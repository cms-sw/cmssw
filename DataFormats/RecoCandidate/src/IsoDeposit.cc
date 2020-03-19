#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

using namespace reco;

IsoDeposit::IsoDeposit(const Direction& candDirection) : theDirection(candDirection), theCandTag(0.) {
  theVeto.vetoDir = theDirection;
  theVeto.dR = 0.;
}

IsoDeposit::IsoDeposit(double eta, double phi) : theDirection(Direction(eta, phi)), theCandTag(0.) {
  theVeto.vetoDir = theDirection;
  theVeto.dR = 0.;
}

void IsoDeposit::addDeposit(double dr, double value) {
  Distance relDir = {float(dr), 0.f};
  theDeposits.insert(std::make_pair(relDir, value));
}

void IsoDeposit::addDeposit(const Direction& depDir, double deposit) {
  Distance relDir = depDir - theDirection;
  theDeposits.insert(std::make_pair(relDir, deposit));
}

double IsoDeposit::depositWithin(double coneSize, const Vetos& vetos, bool skipDepositVeto) const {
  return depositAndCountWithin(coneSize, vetos, -1e+36, skipDepositVeto).first;
}

double IsoDeposit::depositWithin(Direction dir, double coneSize, const Vetos& vetos, bool skipDepositVeto) const {
  return depositAndCountWithin(dir, coneSize, vetos, -1e+36, skipDepositVeto).first;
}

std::pair<double, int> IsoDeposit::depositAndCountWithin(double coneSize,
                                                         const Vetos& vetos,
                                                         double threshold,
                                                         bool skipDepositVeto) const {
  double result = 0;
  int count = 0;

  Vetos allVetos = vetos;
  typedef Vetos::const_iterator IV;
  if (!skipDepositVeto)
    allVetos.push_back(theVeto);
  IV ivEnd = allVetos.end();

  Distance maxDistance = {float(coneSize), 999.f};
  typedef DepositsMultimap::const_iterator IM;
  IM imLoc = theDeposits.upper_bound(maxDistance);
  for (IM im = theDeposits.begin(); im != imLoc; ++im) {
    bool vetoed = false;
    for (IV iv = allVetos.begin(); iv < ivEnd; ++iv) {
      Direction dirDep = theDirection + im->first;
      if (dirDep.deltaR(iv->vetoDir) < iv->dR)
        vetoed = true;
    }
    if (!vetoed && im->second > threshold) {
      result += im->second;
      count++;
    }
  }
  return std::pair<double, int>(result, count);
}

std::pair<double, int> IsoDeposit::depositAndCountWithin(
    Direction dir, double coneSize, const Vetos& vetos, double threshold, bool skipDepositVeto) const {
  double result = 0;
  int count = 0;

  Vetos allVetos = vetos;
  typedef Vetos::const_iterator IV;
  if (!skipDepositVeto)
    allVetos.push_back(theVeto);
  IV ivEnd = allVetos.end();

  typedef DepositsMultimap::const_iterator IM;
  for (IM im = theDeposits.begin(); im != theDeposits.end(); ++im) {
    bool vetoed = false;
    Direction dirDep = theDirection + im->first;
    Distance newDist = dirDep - dir;
    if (newDist.deltaR > coneSize)
      continue;
    for (IV iv = allVetos.begin(); iv < ivEnd; ++iv) {
      if (dirDep.deltaR(iv->vetoDir) < iv->dR)
        vetoed = true;
    }
    if (!vetoed && im->second > threshold) {
      result += im->second;
      count++;
    }
  }
  return std::pair<double, int>(result, count);
}

std::pair<double, int> IsoDeposit::depositAndCountWithin(double coneSize,
                                                         const AbsVetos& vetos,
                                                         bool skipDepositVeto) const {
  using namespace reco::isodeposit;
  double result = 0;
  int count = 0;
  typedef AbsVetos::const_iterator IV;

  IV ivEnd = vetos.end();

  Distance maxDistance = {float(coneSize), 999.f};
  typedef DepositsMultimap::const_iterator IM;
  IM imLoc = theDeposits.upper_bound(maxDistance);
  for (IM im = theDeposits.begin(); im != imLoc; ++im) {
    bool vetoed = false;
    Direction dirDep = theDirection + im->first;
    for (IV iv = vetos.begin(); iv < ivEnd; ++iv) {
      if ((*iv)->veto(dirDep.eta(), dirDep.phi(), im->second)) {
        vetoed = true;
        break;
      }
    }
    if (!vetoed) {
      if (skipDepositVeto || (dirDep.deltaR(theVeto.vetoDir) > theVeto.dR)) {
        result += im->second;
        count++;
      }
    }
  }
  return std::pair<double, int>(result, count);
}

double IsoDeposit::depositWithin(double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const {
  return depositAndCountWithin(coneSize, vetos, skipDepositVeto).first;
}

double IsoDeposit::countWithin(double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const {
  return algoWithin<CountAlgo>(coneSize, vetos, skipDepositVeto);
}
double IsoDeposit::sumWithin(double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const {
  return algoWithin<SumAlgo>(coneSize, vetos, skipDepositVeto);
}
double IsoDeposit::sumWithin(const Direction& dir, double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const {
  return algoWithin<SumAlgo>(dir, coneSize, vetos, skipDepositVeto);
}
double IsoDeposit::sum2Within(double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const {
  return algoWithin<Sum2Algo>(coneSize, vetos, skipDepositVeto);
}
double IsoDeposit::maxWithin(double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const {
  return algoWithin<MaxAlgo>(coneSize, vetos, skipDepositVeto);
}

double IsoDeposit::nearestDR(double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const {
  using namespace reco::isodeposit;
  double result = coneSize;
  typedef AbsVetos::const_iterator IV;

  IV ivEnd = vetos.end();

  Distance maxDistance = {float(coneSize), 999.f};
  typedef DepositsMultimap::const_iterator IM;
  IM imLoc = theDeposits.upper_bound(maxDistance);
  for (IM im = theDeposits.begin(); im != imLoc; ++im) {
    bool vetoed = false;
    Direction dirDep = theDirection + im->first;
    for (IV iv = vetos.begin(); iv < ivEnd; ++iv) {
      if ((*iv)->veto(dirDep.eta(), dirDep.phi(), im->second)) {
        vetoed = true;
        break;
      }
    }
    if (!vetoed) {
      if (skipDepositVeto || (dirDep.deltaR(theVeto.vetoDir) > theVeto.dR)) {
        result = (dirDep.deltaR(theVeto.vetoDir) < result) ? dirDep.deltaR(theVeto.vetoDir) : result;
      }
    }
  }
  return result;
}

std::string IsoDeposit::print() const {
  std::ostringstream str;
  str << "Direction : " << theDirection.print() << std::endl;
  str << "Veto:       (" << theVeto.vetoDir.eta() << ", " << theVeto.vetoDir.phi() << " dR=" << theVeto.dR << ")"
      << std::endl;
  typedef DepositsMultimap::const_iterator IM;
  IM imEnd = theDeposits.end();
  for (IM im = theDeposits.begin(); im != imEnd; ++im) {
    str << "(dR=" << im->first.deltaR << ", alpha=" << im->first.relativeAngle << ", Pt=" << im->second << "),";
  }
  str << std::endl;

  return str.str();
}
