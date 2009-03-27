#include "DataFormats/PatCandidates/interface/MHT.h"

void
pat::MHT::setNumberOfMuons(const double & numberOfMuons)
{
  number_of_muons_ = numberOfMuons;
}

double
pat::MHT::getNumberOfMuons() const
{
  return number_of_muons_;
}

void 
pat::MHT::setNumberOfJets(const double & numberOfJets)
{
  number_of_jets_ = numberOfJets;
}

double
pat::MHT::getNumberOfJets() const{
  return number_of_jets_;
}

void 
pat::MHT::setNumberOfElectrons(const double & numberOfElectrons){
  number_of_electrons_ = numberOfElectrons;
}

double 
pat::MHT::getNumberOfElectrons() const{
  return number_of_electrons_;
}

void 
pat::MHT::setMETsignificance(const double & met_significance){
  met_significance_  = met_significance;
}


double 
pat::MHT::getMETsignificance() const{
  return met_significance_;
}

