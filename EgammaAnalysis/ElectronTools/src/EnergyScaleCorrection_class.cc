#include "../interface/EnergyScaleCorrection_class.hh"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <RooDataSet.h>
#include <RooArgSet.h>

#define NGENMAX 10000

// for exit(0)
#include <stdlib.h>
#include <float.h> 
// for setw
#include <iomanip>
//for istreamstring
#include <sstream>

//#define DEBUG
//#define PEDANTIC_OUTPUT

EnergyScaleCorrection_class::EnergyScaleCorrection_class(std::string correctionFileName, unsigned int genSeed):
  doScale(false), doSmearings(false),
  smearingType_(ECALELF)
{
  
  if(correctionFileName.size() > 0) { 
    std::string filename = correctionFileName+"_scales.dat";
    ReadFromFile(filename);
    if(scales.empty()) {
      std::cerr << "[ERROR] scale correction map empty" << std::endl;
      exit(1);
    }
  }
  
  if(correctionFileName.size() > 0) { 
    std::string filename = correctionFileName+"_smearings.dat";
    ReadSmearingFromFile(filename);
    if(smearings.empty()) {
      std::cerr << "[ERROR] smearing correction map empty" << std::endl;
      exit(1);
    }
  }
  
  return;
}

EnergyScaleCorrection_class::~EnergyScaleCorrection_class(void)
{
  return;
}



float EnergyScaleCorrection_class::ScaleCorrection(unsigned int runNumber, bool isEBEle,
						   double R9Ele, double etaSCEle, double EtEle) const
{
  float correction = 1;
  if(doScale) correction *= getScaleOffset(runNumber, isEBEle, R9Ele, etaSCEle, EtEle);
  
  return correction; 
}



float EnergyScaleCorrection_class::ScaleCorrectionUncertainty(unsigned int runNumber, bool isEBEle,
							      double R9Ele, double etaSCEle, double EtEle) const
{
  float statUncert = getScaleStatUncertainty(runNumber, isEBEle, R9Ele, etaSCEle, EtEle);
  float systUncert = getScaleSystUncertainty(runNumber, isEBEle, R9Ele, etaSCEle, EtEle);
  
  return sqrt(statUncert * statUncert + systUncert * systUncert);
}


float EnergyScaleCorrection_class::getScaleStatUncertainty(unsigned int runNumber, bool isEBEle,
							   double R9Ele, double etaSCEle, double EtEle) const
{
  return getScaleCorrection(runNumber, isEBEle, R9Ele, etaSCEle, EtEle).scale_err;
}	

float EnergyScaleCorrection_class::getScaleSystUncertainty(unsigned int runNumber, bool isEBEle,
							   double R9Ele, double etaSCEle, double EtEle) const
{
  return getScaleCorrection(runNumber, isEBEle, R9Ele, etaSCEle, EtEle).scale_err_syst;
}	


correctionValue_class EnergyScaleCorrection_class::getScaleCorrection(unsigned int runNumber, bool isEBEle, double R9Ele, double etaSCEle, double EtEle) const
{
  // buld the category based on the values of the object
  correctionCategory_class category(runNumber, etaSCEle, R9Ele, EtEle);
  correction_map_t::const_iterator corr_itr = scales.find(category); // find the correction value in the map that associates the category to the correction
  
  if(corr_itr == scales.end()) { // if not in the standard classes, add it in the list of not defined classes
    
    // this part is commented because it makes the method not constant
    // if(scales_not_defined.count(category) == 0) {
    // 	correctionValue_class corr;
    // 	scales_not_defined[category] = corr;
    // }
    // corr_itr = scales_not_defined.find(category);
    /// \todo this can be switched to an exeption 
    std::cout << "[ERROR] Scale category not found: " << std::endl;
    std::cout << category << std::endl;
    std::cout << "Returning uncorrected value." << std::endl;
    //     exit(1);
    correctionValue_class nocorr;
    std::cout << nocorr << std::endl;
    return nocorr;
  }
  
#ifdef DEBUG
  std::cout << "[DEBUG] Checking scale correction for category: " << category << std::endl;
  std::cout << "[DEBUG] Correction is: " << corr_itr->second
	    << std::endl
	    << "        given for category " <<  corr_itr->first << std::endl;;
#endif
  return corr_itr->second;
}

float EnergyScaleCorrection_class::getScaleOffset(unsigned int runNumber, bool isEBEle, double R9Ele, double etaSCEle, double EtEle) const
{
  // buld the category based on the values of the object
  correctionCategory_class category(runNumber, etaSCEle, R9Ele, EtEle);
  correction_map_t::const_iterator corr_itr = scales.find(category); // find the correction value in the map that associates the category to the correction
  
  if(corr_itr == scales.end()) { // if not in the standard classes, add it in the list of not defined classes
    
    // this part is commented because it makes the method not constant
    // if(scales_not_defined.count(category) == 0) {
    // 	correctionValue_class corr;
    // 	scales_not_defined[category] = corr;
    // }
    // corr_itr = scales_not_defined.find(category);
    /// \todo this can be switched to an exeption 
    std::cout << "[ERROR] Scale offset category not found: " << std::endl;
    std::cout << category << std::endl;
    std::cout << "Returning uncorrected value." << std::endl;
    //     exit(1);
    correctionValue_class nocorr;
    return nocorr.scale;
  }
  
#ifdef DEBUG
  std::cout << "[DEBUG] Checking scale offset correction for category: " << category << std::endl;
  std::cout << "[DEBUG] Correction is: " << corr_itr->second
	    << std::endl
	    << "        given for category " <<  corr_itr->first << std::endl;;
#endif
  
  return corr_itr->second.scale;
}


/**
 *  Input file structure:
 *  category  "runNumber"   runMin  runMax   deltaP  err_deltaP(stat on single bins)  err_deltaP_stat(to be used) err_deltaP_syst(to be used)
 *
 */
void EnergyScaleCorrection_class::ReadFromFile(TString filename)
{
#ifdef PEDANTIC_OUTPUT
  std::cout << "[STATUS] Reading energy scale correction  values from file: " << filename << std::endl;
#endif
  std::cout << "[STATUS] Reading energy scale correction  values from file: " << filename << std::endl;

  //std::ifstream Ccufile(edm::FileInPath(Ccufilename).fullPath().c_str(),std::ios::in);
  std::ifstream f_in(edm::FileInPath(filename).fullPath().c_str());

  if(!f_in.good()) {
    std::cerr << "[ERROR] file " << filename << " not readable" << std::endl;
    exit(1);
    return;
  }
  
  int runMin, runMax;
  TString category, region2;
  double deltaP, err_deltaP, err_deltaP_stat, err_deltaP_syst;
  
  
  for(f_in >> category; f_in.good(); f_in >> category) {
    f_in >> region2
	 >> runMin >> runMax
	 >> deltaP >> err_deltaP >> err_deltaP_stat >> err_deltaP_syst;
    
    AddScale(category, runMin, runMax, deltaP, err_deltaP_stat, err_deltaP_syst);
  }
  
  f_in.close();
  
  return;
}

// this method adds the correction values read from the txt file to the map
void EnergyScaleCorrection_class::AddScale(TString category_, int runMin_, int runMax_, double deltaP_, double err_deltaP_, double err_syst_deltaP)
{
  
  correctionCategory_class cat(category_); // build the category from the string
  cat.runmin = runMin_;
  cat.runmax = runMax_;
  
  // the following check is not needed, can be removed
  if(scales.count(cat) != 0) { 
    std::cerr << "[ERROR] Category already defined!" << std::endl;
    std::cerr << "        Adding category:  " << cat << std::endl;
    std::cerr << "        Defined category: " << scales[cat] << std::endl;
    exit(1);
  }
  
  correctionValue_class corr; // define the correction values
  corr.scale = deltaP_;
  corr.scale_err = err_deltaP_;
  corr.scale_err_syst = err_syst_deltaP;
  scales[cat] = corr;
  
#ifdef PEDANTIC_OUTPUT
  std::cout << "[INFO:scale correction] " << cat << corr << std::endl;
#endif
  return;
}

//============================== SMEARING
void EnergyScaleCorrection_class::AddSmearing(TString category_, int runMin_, int runMax_,
					      double rho, double err_rho, double phi, double err_phi,
					      double Emean, double err_Emean)
{
  
  correctionCategory_class cat(category_);
  cat.runmin = (runMin_ < 0) ? 0 : runMin_;
  cat.runmax = runMax_;
  
  if(smearings.count(cat) != 0) {
    std::cerr << "[ERROR] Smearing category already defined!" << std::endl;
    std::cerr << "        Adding category:  " << cat << std::endl;
    std::cerr << "        Defined category: " << smearings[cat] << std::endl;
    exit(1);
  }
  
  correctionValue_class corr;
  corr.rho    = rho;
  corr.rho_err = err_rho;
  corr.phi        = phi;
  corr.phi_err    = err_phi;
  corr.Emean        = Emean;
  corr.Emean_err    = err_Emean;
  smearings[cat] = corr;
  
#ifdef PEDANTIC_OUTPUT
#ifndef CMSSW
  std::cout << "[INFO:smearings] " << cat << corr << std::endl;
#else
  edm::LogInfo("[INFO:smearings] ") << cat << corr;
#endif
#endif
  
  return;
}

/**
 *  File structure:
 EBlowEtaBad8TeV    0 0.0 1.0 -999. 0.94 -999999 999999 6.73 0. 7.7e-3  6.32e-4 0.00 0.16
 EBlowEtaGold8TeV   0 0.0 1.0 0.94  999. -999999 999999 6.60 0. 7.4e-3  6.50e-4 0.00 0.16
 EBhighEtaBad8TeV   0 1.0 1.5 -999. 0.94 -999999 999999 6.73 0. 1.26e-2 1.03e-3 0.00 0.07
 EBhighEtaGold8TeV  0 1.0 1.5 0.94  999. -999999 999999 6.52 0. 1.12e-2 1.32e-3 0.00 0.22
 ##################################################################################################
 EElowEtaBad8TeV    0 1.5 2.0 -999. 0.94 -999999 999999 0.   0. 1.98e-2 3.03e-3 0.  0.
 EElowEtaGold8TeV   0 1.5 2.0 0.94  999. -999999 999999 0.   0. 1.63e-2 1.22e-3 0.  0.
 EEhighEtaBad8TeV   0 2.0 3.0 -999. 0.94 -999999 999999 0.   0. 1.92e-2 9.22e-4 0.  0.
 EEhighEtaGold8TeV  0 2.0 3.0 0.94  999. -999999 999999 0.   0. 1.86e-2 7.81e-4 0.  0.
 ##################################################################################################
 *
 */

void EnergyScaleCorrection_class::ReadSmearingFromFile(TString filename)
{
#ifdef PEDANTIC_OUTPUT
  std::cout << "[STATUS] Reading smearing values from file: " << filename << std::endl;
#endif
  //edm::FileInPath(Ccufilename).fullPath().c_str(),std::ios::in); .fullPath().c_str()
  std::ifstream f_in(edm::FileInPath(filename).fullPath().c_str());
  if(!f_in.good()) {
    std::cerr << "[ERROR] file " << filename << " not readable" << std::endl;
    exit(1);
  }
  
  int runMin = 0, runMax = 900000;
  int unused = 0;
  TString category, region2;
  //double smearing, err_smearing;
  double rho, phi, Emean, err_rho, err_phi, err_Emean;
  double etaMin, etaMax, r9Min, r9Max;
  std::string phi_string, err_phi_string;
  
  
  while(f_in.peek() != EOF && f_in.good()) {
    if(f_in.peek() == 10) { // 10 = \n
      f_in.get();
      continue;
    }
    
    if(f_in.peek() == 35) { // 35 = #
      f_in.ignore(1000, 10); // ignore the rest of the line until \n
      continue;
    }
    
    if(smearingType_ == UNKNOWN) { // trying to guess: not recommended
		std::cerr << "[ERROR] Not implemented" << std::endl;
		assert(false);
		
    } else if(smearingType_ == GLOBE) {
		f_in >> category >> unused >> etaMin >> etaMax >> r9Min >> r9Max >> runMin >> runMax >>
			Emean >> err_Emean >>
			rho >> err_rho >> phi >> err_phi;

		AddSmearing(category, runMin, runMax, rho,  err_rho, phi, err_phi, Emean, err_Emean);
      
    } else if(smearingType_ == ECALELF) {
		f_in >> category >> 
			Emean >> err_Emean >>
			rho >> err_rho >> phi_string >> err_phi_string;
#ifdef DEBUG
		std::cout << category 
			//<< "\t" << etaMin << "\t" << etaMax << "\t" << r9Min << "\t" << r9Max << "\t" << runMin << "\t" << runMax 
				  << "\tEmean=" << Emean << "\t" 
				  << rho << "\t" << err_rho << "\tphi_string=" 
				  << phi_string << "#\terr_phi_string=" << err_phi_string << std::endl;
#endif
		
      if(phi_string=="M_PI_2") phi=M_PI_2;
      else phi = std::stod(phi_string);
      
      if(err_phi_string=="M_PI_2") err_phi=M_PI_2;
      else err_phi = std::stod(err_phi_string);
      
      
      AddSmearing(category, runMin, runMax, rho,  err_rho, phi, err_phi, Emean, err_Emean);
      
    } else {
      f_in >> category >> rho >> phi;
      AddSmearing(category, runMin, runMax, rho,  err_rho, phi, err_phi, Emean, err_Emean);
    }
#ifdef DEBUG
    std::cout << category << "\t" << etaMin << "\t" << etaMax << "\t" << r9Min << "\t" << r9Max << "\t" << runMin << "\t" << runMax << "\tEmean=" << Emean << "\t" << rho << "\t" << phi << std::endl;
#endif
  }
  
  f_in.close();
  //  runCorrection_itr=runMin_map.begin();
  
  
  return;
}



float EnergyScaleCorrection_class::getSmearingSigma(int runNumber, bool isEBEle, float R9Ele, float etaSCEle, float EtEle, paramSmear_t par, float nSigma) const
{
  if (par == kRho) return getSmearingSigma(runNumber, isEBEle, R9Ele, etaSCEle, EtEle, nSigma, 0.);
  if (par == kPhi) return getSmearingSigma(runNumber, isEBEle, R9Ele, etaSCEle, EtEle, 0., nSigma);
  return getSmearingSigma(runNumber, isEBEle, R9Ele, etaSCEle, EtEle, 0., 0.);
}

float EnergyScaleCorrection_class::getSmearingSigma(int runNumber, bool isEBEle, float R9Ele, float etaSCEle, float EtEle, float nSigma_rho, float nSigma_phi) const
{
  
  correctionCategory_class category(runNumber, etaSCEle, R9Ele, EtEle);
  correction_map_t::const_iterator corr_itr = smearings.find(category);
  if(corr_itr == smearings.end()) { // if not in the standard classes, add it in the list of not defined classes
    // the following commented part makes the method non const
    // if(smearings_not_defined.count(category) == 0) {
    // 	correctionValue_class corr;
    // 	smearings_not_defined[category] = corr;
    // }
    corr_itr = smearings_not_defined.find(category);
    std::cerr << "[WARNING] Smearing category not found: " << std::endl;
    std::cerr << category << std::endl;
    //     exit(1);
  }
  
#ifdef DEBUG
  std::cout << "[DEBUG] Checking smearing correction for category: " << category << std::endl;
  std::cout << "[DEBUG] Correction is: " << corr_itr->second
	    << std::endl
	    << "        given for category " <<  corr_itr->first;
#endif

  double rho = corr_itr->second.rho + corr_itr->second.rho_err * nSigma_rho;
  double phi = corr_itr->second.phi + corr_itr->second.phi_err * nSigma_phi;

  double constTerm =  rho * sin(phi);
  double alpha =  rho *  corr_itr->second.Emean * cos( phi);

  return sqrt(constTerm * constTerm + alpha * alpha / EtEle);
  
}

float EnergyScaleCorrection_class::getSmearingRho(int runNumber, bool isEBEle, float R9Ele, float etaSCEle, float EtEle) const
{
  
  correctionCategory_class category(runNumber, etaSCEle, R9Ele, EtEle);
  correction_map_t::const_iterator corr_itr = smearings.find(category);
  if(corr_itr == smearings.end()) { // if not in the standard classes, add it in the list of not defined classes
    // if(smearings_not_defined.count(category) == 0) {
    // 	correctionValue_class corr;
    // 	smearings_not_defined[category] = corr;
    // }
    corr_itr = smearings_not_defined.find(category);
  }
  
  return corr_itr->second.rho;
}

bool correctionCategory_class::operator<(const correctionCategory_class& b) const
{
  if(runmin < b.runmin && runmax < b.runmax) return true;
  if(runmax > b.runmax && runmin > b.runmin) return false;
  
  if(etamin < b.etamin && etamax < b.etamax) return true;
  if(etamax > b.etamax && etamin > b.etamin) return false;
  
  if(r9min  < b.r9min && r9max < b.r9max) return true;
  if(r9max  > b.r9max && r9min > b.r9min) return  false;
  
  if(etmin  < b.etmin && etmax < b.etmax) return true;
  if(etmax  > b.etmax && etmin > b.etmin) return  false;
  return false;
  
}

correctionCategory_class::correctionCategory_class(TString category_)
{
  std::string category(category_.Data());
#ifdef DEBUG
  std::cout << "[DEBUG] correctionClass defined for category: " << category << std::endl;
#endif
  // default values (corresponding to an existing category -- the worst one)
  runmin = 0;
  runmax = 999999;
  etamin = 2;
  etamax = 7;
  r9min = -1;
  r9max = 0.94;
  etmin = -1;
  etmax = 99999.;
  
  size_t p1, p2; // boundary
  // eta region
  p1 = category.find("absEta_");
  p2 = p1 + 1;
  if(category.find("absEta_0_1") != std::string::npos) {
    etamin = 0;
    etamax = 1;
  } else if(category.find("absEta_1_1.4442") != std::string::npos) {
    etamin = 1;
    etamax = 1.479;
  } //etamax=1.4442; }
  else if(category.find("absEta_1.566_2") != std::string::npos) {
    etamin = 1.479;
    etamax = 2;
  } //etamin=1.566; etamax=2;}
  else if(category.find("absEta_2_2.5") != std::string::npos) {
    etamin = 2;
    etamax = 3;
  } else {
    if(p1 != std::string::npos) {
      p1 = category.find("_", p1);
      p2 = category.find("_", p1 + 1);
      etamin = TString(category.substr(p1 + 1, p2 - p1 - 1)).Atof();
      p1 = p2;
      p2 = category.find("-", p1);
      etamax = TString(category.substr(p1 + 1, p2 - p1 - 1)).Atof();
    }
  }
  
  if(category.find("EBlowEta") != std::string::npos) {
    etamin = 0;
    etamax = 1;
  };
  if(category.find("EBhighEta") != std::string::npos) {
    etamin = 1;
    etamax = 1.479;
  };
  if(category.find("EElowEta") != std::string::npos) {
    etamin = 1.479;
    etamax = 2;
  };
  if(category.find("EEhighEta") != std::string::npos) {
    etamin = 2;
    etamax = 7;
  };
  
  // Et region
  p1 = category.find("-Et_");
  p2 = p1 + 1;
  //    std::cout << "p1 = " << p1 << "\t" << std::string::npos << "\t" << category.size() << std::endl;
  //std::cout << etmin << "\t" << etmax << "\t" << category.substr(p1+1, p2-p1-1) << "\t" << p1 << "\t" << p2 << std::endl;
  if(p1 != std::string::npos) {
    p1 = category.find("_", p1);
    p2 = category.find("_", p1 + 1);
    etmin = TString(category.substr(p1 + 1, p2 - p1 - 1)).Atof();
    p1 = p2;
    p2 = category.find("-", p1);
    etmax = TString(category.substr(p1 + 1, p2 - p1 - 1)).Atof();
    //  std::cout << etmin << "\t" << etmax << "\t" << category.substr(p1 + 1, p2 - p1 - 1) << std::endl;
  }
  
  if(category.find("gold")   != std::string::npos || 
     category.find("Gold")   != std::string::npos || 
     category.find("highR9") != std::string::npos) {
    r9min = 0.94;
    r9max = FLT_MAX;
  } else if(category.find("bad") != std::string::npos || 
	    category.find("Bad") != std::string::npos ||
	    category.find("lowR9") != std::string::npos
	    ) {
    r9min = -1;
    r9max = 0.94;
  };	
}
