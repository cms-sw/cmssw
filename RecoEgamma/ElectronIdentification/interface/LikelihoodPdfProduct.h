#ifndef LikelihoodPdfProduct_h
#define LikelihoodPdfProduct_h

#include "RecoEgamma/ElectronIdentification/interface/LikelihoodSpecies.h"
#include "RecoEgamma/ElectronIdentification/interface/LikelihoodPdf.h"
#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCalibration.h"
#include <TDirectory.h>
#include <string>
#include <vector>
#include <map>

class LikelihoodPdfProduct {
 public:
  LikelihoodPdfProduct(const char* name, int ecalsubdet, int ptbin);
  ~LikelihoodPdfProduct();
  
  //! initialize the PDFs from CondDB  
  void initFromDB(const ElectronLikelihoodCalibration *calibration);

  //! add a species (hypothesis) to the likelihood, with a priori probability 
  void addSpecies(const char* name, float priorWeight=1.);

  //! add a pdf for a species, splitted or not
  void addPdf(const char* specname, const char* name, bool splitPdf=false);

  //! set the fraction of one category for a given species
  void setSplitFrac(const char* specname, const char* catName, float frac=1.0);

  //! get the likelihood ratio p(a priori) * L(specName) / L_tot
  float getRatio(const char* specName, const std::vector<float>& measurements, const std::string&) const;

 private:

  float getSpeciesProb(const char* specName, const std::vector<float>& measurements, const std::string& gsfClass) const;
  std::string _name;
  const ElectronLikelihoodCalibration *_calibration;
  std::vector<LikelihoodSpecies*> _specList;
  std::vector<float> _priorList;
  int _ecalsubdet;
  int _ptbin;

};
#endif
    
