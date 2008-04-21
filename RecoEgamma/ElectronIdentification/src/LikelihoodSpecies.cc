#include "RecoEgamma/ElectronIdentification/interface/LikelihoodSpecies.h"

LikelihoodSpecies::LikelihoodSpecies(const char* name, float prior) {
  _name=std::string(name);
  _prior=prior;
}

LikelihoodSpecies::~LikelihoodSpecies() {
  std::vector<LikelihoodPdf*>::iterator pdfItr;
  for(pdfItr=_pdfList.begin(); pdfItr!=_pdfList.end(); pdfItr++) {
    delete *pdfItr;
  }
}

void LikelihoodSpecies::setName(const char* name) {
  _name = std::string(name);
}

void LikelihoodSpecies::addPdf(LikelihoodPdf* pdf) {
  _pdfList.push_back(pdf);
}

void LikelihoodSpecies::setPrior(float prior) {
  _prior=prior;
}

void LikelihoodSpecies::setSplitFraction(std::pair<std::string,float> splitfrac) {
  _splitFractions.insert(splitfrac);
}

std::vector<LikelihoodPdf*> LikelihoodSpecies::getListOfPdfs() {
  return _pdfList;
}

const char* LikelihoodSpecies::getName() {
  return _name.c_str();
}

float LikelihoodSpecies::getPrior() {
  return _prior;
}

std::map<std::string,float> LikelihoodSpecies::getSplitFractions() {
  return _splitFractions;
}

