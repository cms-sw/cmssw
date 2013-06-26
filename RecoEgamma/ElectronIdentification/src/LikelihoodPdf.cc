#include "RecoEgamma/ElectronIdentification/interface/LikelihoodPdf.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>




LikelihoodPdf::LikelihoodPdf(const char* name, const char* species, int ecalsubdet, int ptbin) {
  _name = std::string(name);
  _species = std::string(species);
  _ecalsubdet = ecalsubdet;
  _ptbin = ptbin;
}



LikelihoodPdf::~LikelihoodPdf() {
}



void 
LikelihoodPdf::split(const std::map<std::string,float>& splitFractions, 
		     bool splitPdf) {
//  std::map<std::string,float>splitFractions = _splitFractions;
  char buffer[100];
  //! use different a-priori probabilities and different PDFs 
  //! depending by category
  if(splitFractions.size()>0 && splitPdf) {
    std::map<std::string,float>::const_iterator splitCatItr;
    for(splitCatItr=splitFractions.begin();splitCatItr!=splitFractions.end();splitCatItr++) {
      snprintf(buffer, 100, "%s_%s_subdet%d_ptbin%d_%s",_name.c_str(),_species.c_str(),_ecalsubdet,_ptbin,splitCatItr->first.c_str());
      std::string totPdfName = std::string(buffer);
      _splitRule.insert( std::make_pair(splitCatItr->first,totPdfName) );
    }
  }

  //! use different a-priori, but same PDFs for all categories
  else if(splitFractions.size()>0) {
    std::map<std::string,float>::const_iterator splitCatItr;
    for(splitCatItr=splitFractions.begin();splitCatItr!=splitFractions.end();splitCatItr++) {
      snprintf(buffer, 100, "%s_%s_subdet%d_ptbin%d",_name.c_str(),_species.c_str(),_ecalsubdet,_ptbin);
      std::string totPdfName = std::string(buffer);
      _splitRule.insert( std::make_pair(splitCatItr->first,totPdfName) );
    }
  }
  
  //! do not split at all (same PDF's, same a-priori for all categories)
  else {
    snprintf(buffer, 100, "%s_%s_subdet%d_ptbin%d",_name.c_str(),_species.c_str(),_ecalsubdet,_ptbin);
    std::string totPdfName = std::string(buffer);
    _splitRule.insert( std::make_pair("NOSPLIT",totPdfName) );
  }
}



void 
LikelihoodPdf::initFromDB(const ElectronLikelihoodCalibration *calibration) {

  std::map<std::string,std::string>::const_iterator ruleItr;
  for(ruleItr=_splitRule.begin();ruleItr!=_splitRule.end();ruleItr++) {
    // look for the requested PDF in the CondDB
    std::vector<ElectronLikelihoodCalibration::Entry>::const_iterator entryItr;
    bool foundPdf=false;
    for(entryItr=calibration->data.begin(); entryItr!=calibration->data.end(); entryItr++) {
      if(entryItr->category.label.compare(ruleItr->second)==0) { 
	const PhysicsTools::Calibration::HistogramF *histo = &(entryItr->histogram);
	_splitPdf.insert( std::make_pair(ruleItr->first,histo) );
	foundPdf=true;
      }
    }
    if(!foundPdf) {
      throw cms::Exception("LikelihoodPdf") << "The pdf requested: " << _name
					    << " for species: " << _species
					    << " is not present in the Conditions DB!";
    }
  }
}



float 
LikelihoodPdf::getVal(float x, std::string gsfClass, 
		      bool normalized) {
  const PhysicsTools::Calibration::HistogramF *thePdf=0;
  if(_splitPdf.size()>1) {
    edm::LogInfo("LikelihoodPdf") << "The PDF " << _name
				  << " is SPLITTED by category " << gsfClass;
    thePdf=_splitPdf.find(gsfClass)->second;
  }
  else {
    edm::LogInfo("LikelihoodPdf") << "The PDF " << _name
				  << " is UNSPLITTED";
    thePdf=_splitPdf.find("NOSPLIT")->second;
  }
  
  float prob=-1;

  if(normalized)
    // using thePdf->normalization() neglects the overflows... calculating it manually.
    // prob=thePdf->value(x)/thePdf->normalization();
    prob=thePdf->value(x)/normalization(thePdf);
  else
    prob=thePdf->value(x);

  edm::LogInfo("LikelihoodPdf") << "sanity check: PDF name = " << _name
                                << " for species = " << _species
                                << " for class = " << gsfClass 
                                << " bin content = " << thePdf->binContent(thePdf->findBin(x))
                                << " normalization (std) = " << thePdf->normalization()
                                << " normalization (manual) = " << normalization(thePdf)
                                << " prob = " << prob;
  edm::LogInfo("LikelihoodPdf") << "From likelihood with ecalsubdet = " << _ecalsubdet
                                << " ptbin = " << _ptbin;
 

  return prob;
}

// Histogram::normalization() gives the integral excluding the over-underflow...
float
LikelihoodPdf::normalization(const PhysicsTools::Calibration::HistogramF *thePdf) {
  int nBins = thePdf->numberOfBins();
  float sum=0.;
  for(int i=0; i<=nBins+1; i++) {
    sum += thePdf->binContent(i);
  }
  return sum;
}
