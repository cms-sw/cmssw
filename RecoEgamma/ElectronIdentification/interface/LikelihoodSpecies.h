#ifndef LikelihoodSpecies_h
#define LikelihoodSpecies_h

#include "RecoEgamma/ElectronIdentification/interface/LikelihoodPdf.h"
#include <vector>
#include <string>
#include <map>

class LikelihoodSpecies {
 public:
  LikelihoodSpecies() {};
  LikelihoodSpecies(const char* name, float prior);

  virtual ~LikelihoodSpecies();

  // modifiers
  void setName(const char* name);
  void addPdf(LikelihoodPdf* pdf);
  void setPrior(float prior);
  void setSplitFraction(std::pair<std::string,float> splitfrac);

  // methods
  std::vector<LikelihoodPdf*> getListOfPdfs();
  const char* getName();
  float getPrior();
  std::map<std::string,float> getSplitFractions();

 private:
  std::vector<LikelihoodPdf*> _pdfList;
  std::string _name;
  float _prior;
  std::map<std::string,float> _splitFractions;

};
#endif
