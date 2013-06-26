#ifndef LikelihoodPdf_H
#define LikelihoodPdf_H

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCalibration.h"
#include <TH1F.h>
#include <TDirectory.h>
#include <string>
#include <vector>
#include <map>

class LikelihoodPdf {

 public:
  LikelihoodPdf() {};
  LikelihoodPdf(const LikelihoodPdf& pdf) {}; 
  LikelihoodPdf(const char* name, const char* species, int ecalsubdet, int ptbin);
  virtual ~LikelihoodPdf();
  
  //! initialize PDFs from CondDB
  void initFromDB(const ElectronLikelihoodCalibration *calibration);

  //! split the pdf by category if splitPdf is true. split map is: <"class",classFraction>
  //! if splitPdf is false, pdf is splitted, but they are all equal (but allowing different priors)
  void split(const std::map<std::string,float>& splitFractions, bool splitPdf = false);

  //! get Value of pdf at point x for class catName
  float getVal(float x, std::string catName="NOSPLIT", bool normalized = true);

  //! get PDF name
  std::string getName() { return _name; }

  //! get PDF species
  std::string getSpecies() { return _species; }



 private:

  float normalization(const PhysicsTools::Calibration::HistogramF *thePdf);
  
  std::string _name;
  std::string _species;
  int _ecalsubdet;
  int _ptbin;

  std::map<std::string,const PhysicsTools::Calibration::HistogramF*> _splitPdf;
  std::map<std::string,std::string> _splitRule;

};

#endif
