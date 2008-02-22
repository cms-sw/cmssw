#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositVetos.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

using namespace reco;

//bool MuIsoDeposit::Closer::operator()( const Distance & d1, const Distance & d2) const
//{
//  if ( d1.deltaR < d2.deltaR) return true;
//  else if (d1.deltaR == d2.deltaR && d1.relativeAngle < d2.relativeAngle) return true;
//  else return false;
//}

MuIsoDeposit::MuIsoDeposit(const std::string type, const Direction & muonDirection)
  : theType(type), theDirection(muonDirection),theMuonTag(0.)
{ 
  theVeto.vetoDir = theDirection;
  theVeto.dR = 0.;
}

MuIsoDeposit::MuIsoDeposit(const std::string type, double eta, double phi)
  : theType(type), theDirection(Direction(eta,phi)), theMuonTag(0.)
{ 
  theVeto.vetoDir = theDirection;
  theVeto.dR = 0.;
}

void MuIsoDeposit::addDeposit(double dr, double value){
  Distance relDir = {dr,0.};
  theDeposits.insert( std::make_pair( relDir, value));
}

void MuIsoDeposit::addDeposit(const Direction & depDir, double deposit)
{
  Distance relDir = depDir - theDirection;
  theDeposits.insert( std::make_pair( relDir,deposit));
/*
  {
    Direction dtmp = theDirection+relDir;
    if (dtmp.deltaR(depDir) > 1.e-4) {
         std::cout <<"ERROR, dR= "<<dtmp.deltaR(depDir)<<std::endl;
         std::cout<<" theDirection"<<theDirection.print()<<std::endl;
         std::cout<<" depDir:     "<<depDir.print()<<std::endl;
         std::cout<<" recovered:  "<<dtmp.print()<<std::endl;
         std::cout<<" relative dr: " << relDir.deltaR<<" relative angle: " << relDir.relativeAngle<<std::endl;
    }
  }
*/
}

double MuIsoDeposit::depositWithin(double coneSize, const Vetos& vetos, bool skipDepositVeto) const 
{
  return depositAndCountWithin(coneSize, vetos, -1e+36, skipDepositVeto).first;
}

double MuIsoDeposit::depositWithin(Direction dir, double coneSize, const Vetos& vetos, bool skipDepositVeto) const 
{
  return depositAndCountWithin(dir, coneSize, vetos, -1e+36, skipDepositVeto).first;
}

std::pair<double,int> MuIsoDeposit::depositAndCountWithin(double coneSize, const Vetos& vetos, 
							  double threshold, bool skipDepositVeto) const 
{
  double result = 0;
  int count = 0;

  Vetos allVetos = vetos;
  typedef Vetos::const_iterator IV;
  if (!skipDepositVeto) allVetos.push_back(theVeto);
  IV ivEnd = allVetos.end();

  Distance maxDistance = {coneSize,999.};
  typedef DepositsMultimap::const_iterator IM;
  IM imLoc = theDeposits.upper_bound( maxDistance ); 
  for (IM im = theDeposits.begin(); im != imLoc; ++im) {
    bool vetoed = false;
    for ( IV iv = allVetos.begin(); iv < ivEnd; ++iv) {
      Direction dirDep = theDirection+im->first;
     if (dirDep.deltaR(iv->vetoDir) < iv->dR) vetoed = true; 
    }  
    if (!vetoed && im->second > threshold){
      result += im->second;
      count++;
    }
  }
  return std::pair<double,int>(result,count);
}

std::pair<double,int> MuIsoDeposit::depositAndCountWithin(Direction dir, double coneSize, const Vetos& vetos, 
							  double threshold, bool skipDepositVeto) const 
{
  double result = 0;
  int count = 0;

  Vetos allVetos = vetos;
  typedef Vetos::const_iterator IV;
  if (!skipDepositVeto) allVetos.push_back(theVeto);
  IV ivEnd = allVetos.end();

  typedef DepositsMultimap::const_iterator IM;
  for (IM im = theDeposits.begin(); im != theDeposits.end(); ++im) {
    bool vetoed = false;
    Direction dirDep = theDirection+im->first;
    Distance newDist = dirDep - dir;
    if (newDist.deltaR > coneSize) continue;
    for ( IV iv = allVetos.begin(); iv < ivEnd; ++iv) {
      if (dirDep.deltaR(iv->vetoDir) < iv->dR) vetoed = true; 
    }  
    if (!vetoed && im->second > threshold){
      result += im->second;
      count++;
    }
  }
  return std::pair<double,int>(result,count);
}

std::pair<double,int>  MuIsoDeposit::depositAndCountWithin(double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const 
{
  using namespace reco::isodeposit;
  double result = 0;
  int count =  0;
  typedef AbsVetos::const_iterator IV;

  IV ivEnd = vetos.end();

  Distance maxDistance = {coneSize,999.};
  typedef DepositsMultimap::const_iterator IM;
  IM imLoc = theDeposits.upper_bound( maxDistance ); 
  for (IM im = theDeposits.begin(); im != imLoc; ++im) {
    bool vetoed = false;
    Direction dirDep = theDirection+im->first;
    for ( IV iv = vetos.begin(); iv < ivEnd; ++iv) {
      if ((*iv)->veto(dirDep.eta(), dirDep.phi(), im->second)) { vetoed = true;  break; }
    }
    if (!vetoed) {
       if (skipDepositVeto || (dirDep.deltaR(theVeto.vetoDir) > theVeto.dR)) {
          result += im->second;
		  count++;
        }
    }
  }
  return std::pair<double,int>(result,count);
}


double MuIsoDeposit::depositWithin(double coneSize, const AbsVetos& vetos, bool skipDepositVeto) const 
{
	return depositAndCountWithin(coneSize, vetos, skipDepositVeto).first;
}



std::string MuIsoDeposit::print() const
{
  std::ostringstream str;
  str<<"Direction : "<<theDirection.print()<<std::endl;
  str<<"Veto:       ("<<theVeto.vetoDir.eta()<<", "<<theVeto.vetoDir.phi()<<" dR="<<theVeto.dR<<")"<<std::endl;
  typedef DepositsMultimap::const_iterator IM;
  IM imEnd = theDeposits.end();
  for (IM im = theDeposits.begin(); im != imEnd; ++im) {
    str<<"(dR="<< im->first.deltaR<<", alpha="<<im->first.relativeAngle<<", Pt="<<im->second<<"),";
  }
  str<<std::endl;

  
  
  return str.str();
}
