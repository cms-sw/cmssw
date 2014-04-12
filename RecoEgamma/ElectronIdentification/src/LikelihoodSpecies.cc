#include "RecoEgamma/ElectronIdentification/interface/LikelihoodSpecies.h"

LikelihoodSpecies::LikelihoodSpecies(const char* name, float prior) {
  _name=std::string(name);
  _prior=prior;
}

LikelihoodSpecies::~LikelihoodSpecies() {
  std::vector<const LikelihoodPdf*>::iterator pdfItr;
  for(pdfItr=_pdfList.begin(); pdfItr!=_pdfList.end(); pdfItr++) {
    delete *pdfItr;
  }
}

void LikelihoodSpecies::setName(const char* name) {
  _name = std::string(name);
}

void LikelihoodSpecies::addPdf(const LikelihoodPdf* pdf) {
  _pdfList.push_back(pdf);
}

void LikelihoodSpecies::setPrior(float prior) {
  _prior=prior;
}

void LikelihoodSpecies::setSplitFraction(std::pair<std::string,float> splitfrac) {
  _splitFractions.insert(splitfrac);
}

std::vector<const LikelihoodPdf*> const& LikelihoodSpecies::getListOfPdfs() const {
  return _pdfList;
}

const char* LikelihoodSpecies::getName() const {
  return _name.c_str();
}

float LikelihoodSpecies::getPrior() const {
  return _prior;
}

std::map<std::string,float> const& LikelihoodSpecies::getSplitFractions() const {
  return _splitFractions;
}

