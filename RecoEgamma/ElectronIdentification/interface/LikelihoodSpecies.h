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
  void addPdf(const LikelihoodPdf* pdf);
  void setPrior(float prior);
  void setSplitFraction(std::pair<std::string,float> splitfrac);

  // methods
  std::vector<const LikelihoodPdf*> const& getListOfPdfs() const;
  const char* getName() const;
  float getPrior() const;
  std::map<std::string,float> const& getSplitFractions() const;

 private:
  std::vector<const LikelihoodPdf*> _pdfList;
  std::string _name;
  float _prior;
  std::map<std::string,float> _splitFractions;

};
#endif
