#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace JetMCTagUtils;
using namespace CandMCTagUtils;

///////////////////////////////////////////////////////////////////////

double JetMCTagUtils::EnergyRatioFromBHadrons(const Candidate &c) {
  double ratioForBjet = 0;
  double ratio = 0;
  for (Candidate::const_iterator itC = c.begin(); itC != c.end(); itC++) {
    bool isFromB = decayFromBHadron(*itC);
    ratio = itC->energy() / c.energy();
    if (isFromB)
      ratioForBjet += ratio;
  }
  return ratioForBjet;
}

double JetMCTagUtils::EnergyRatioFromCHadrons(const Candidate &c) {
  double ratioForCjet = 0;
  double ratio = 0;
  for (Candidate::const_iterator itC = c.begin(); itC != c.end(); itC++) {
    bool isFromC = decayFromCHadron(*itC);
    ratio = itC->energy() / c.energy();
    if (isFromC)
      ratioForCjet += ratio;
  }
  return ratioForCjet;
}

bool JetMCTagUtils::decayFromBHadron(const Candidate &c) {
  bool isFromB = false;
  vector<const Candidate *> allParents = getAncestors(c);
  for (vector<const Candidate *>::const_iterator aParent = allParents.begin(); aParent != allParents.end(); aParent++) {
    if (hasBottom(**aParent))
      isFromB = true;
    /*
         cout << "     particle Parent is " << (*aParent)->status()
              << " type " << (*aParent)->pdgId()
              << " pt=" << (*aParent)->pt()
              << " isB = " << isFromB
              << endl;
*/
  }
  return isFromB;
}

bool JetMCTagUtils::decayFromCHadron(const Candidate &c) {
  bool isFromC = false;
  vector<const Candidate *> allParents = getAncestors(c);
  for (vector<const Candidate *>::const_iterator aParent = allParents.begin(); aParent != allParents.end(); aParent++) {
    if (hasCharm(**aParent))
      isFromC = true;
    /*
         cout << "     particle Parent is " << (*aParent)->status()
              << " type " << (*aParent)->pdgId()
              << " pt=" << (*aParent)->pt()
              << " isC = " << isFromC
              << endl;
*/
  }
  return isFromC;
}

std::string JetMCTagUtils::genTauDecayMode(const CompositePtrCandidate &c) {
  int numElectrons = 0;
  int numMuons = 0;
  int numTaus = 0;
  int numChargedHadrons = 0;
  int numNeutralHadrons = 0;
  int numPhotons = 0;

  const CompositePtrCandidate::daughters &daughters = c.daughterPtrVector();
  for (CompositePtrCandidate::daughters::const_iterator daughter = daughters.begin(); daughter != daughters.end();
       ++daughter) {
    int pdg_id = abs((*daughter)->pdgId());

    switch (pdg_id) {
      case 22:
        numPhotons++;
        break;
      case 11:
        numElectrons++;
        break;
      case 13:
        numMuons++;
        break;
      case 15:
        numTaus++;
        break;
      default: {
        if ((*daughter)->charge() != 0)
          numChargedHadrons++;
        else
          numNeutralHadrons++;
      }
    }
  }

  if (numElectrons == 1)
    return std::string("electron");
  else if (numMuons == 1)
    return std::string("muon");
  else if (numTaus == 1)  //MB: a tau undecayed by generator or an intermediate state used to generate radiations
    return std::string("tau");

  switch (numChargedHadrons) {
    case 1:
      if (numNeutralHadrons != 0)
        return std::string("oneProngOther");
      switch (numPhotons) {
        case 0:
          return std::string("oneProng0Pi0");
        case 2:
          return std::string("oneProng1Pi0");
        case 4:
          return std::string("oneProng2Pi0");
        default:
          return std::string("oneProngOther");
      }
    case 3:
      if (numNeutralHadrons != 0)
        return std::string("threeProngOther");
      switch (numPhotons) {
        case 0:
          return std::string("threeProng0Pi0");
        case 2:
          return std::string("threeProng1Pi0");
        default:
          return std::string("threeProngOther");
      }
    default:
      return std::string("rare");
  }
}
