#include "GeneratorInterface/GenFilters/interface/MCFilterZboostHelper.h"

HepMC::FourVector MCFilterZboostHelper::zboost(const HepMC::FourVector& mom, double betaBoost) {
   //Boost this Lorentz vector (from TLorentzVector::Boost)
   double b2 = betaBoost*betaBoost;
   double gamma = 1.0 / sqrt(1.0 - b2);
   double bp = betaBoost*mom.pz();
   double gamma2 = b2 > 0 ? (gamma - 1.0)/b2 : 0.0;

   return HepMC::FourVector(mom.px(), mom.py(), mom.pz() + gamma2*bp*betaBoost + gamma*betaBoost*mom.e(), gamma*(mom.e()+bp));
}
