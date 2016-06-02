#ifndef EgammaAnalysis_ElectronTools_EnergyScaleCorrection_class_h
#define EgammaAnalysis_ElectronTools_EnergyScaleCorrection_class_h
/// Read and get energy scale and smearings from .dat files
/**\class EnergyScaleCorrection_class EnergyScaleCorrection_class.cc Calibration/ZFitter/src/EnergyScaleCorrection_class.cc
 *  \author Shervin Nourbakhsh
 *
 */

/** Description
	This module is taken from the ECALELF package, used to derive the energy scales and smearings.

	There are two sub-classes: 
	 - correctionValue_class that defines the corrections
     - correctionCategory_class that defines the categories
	 There is one map that associates the correction to the category (values read from text file)

	 There is one class that reads the text files with the corrections and returns the scale/smearings given the electron/photon properties
 */

#include <TString.h>
#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <TChain.h>
#include <TRandom3.h>
#include <string>

//============================== First auxiliary class
class correctionValue_class
{
public:
  // values
  float scale, scale_err, scale_err_syst;
  float rho, rho_err;
  float phi, phi_err;
  float Emean, Emean_err;
  
  correctionValue_class(void)
  {
    scale = 1;
    scale_err = 0;
    scale_err_syst=0;
    rho = 0;
    rho_err = 0;
    phi = 0;
    phi_err = 0;
    Emean = 0;
    Emean_err = 0;
  };
  
  friend std::ostream& operator << (std::ostream& os, const correctionValue_class a)
  {
    os <<  "( "
       << a.scale << " +/- " << a.scale_err << " +/- " << a.scale_err_syst << ")" 
       <<  "\t"
       << a.rho << " +/- " << a.rho_err 
       <<  "\t"
       << a.phi << " +/- " << a.phi_err
       <<  "\t"
       << a.Emean << " +/- " << a.Emean_err;
    return os;
  };
};

//============================== Second auxiliary class
class correctionCategory_class
{
  // for class definition and ordering
public:
  unsigned int runmin;
  unsigned int runmax;
  
private:
  // definition of the variables used for binning and the min-max ranges
  float r9min;  ///< min R9 vaule for the bin
  float r9max;  ///< max R9 value for the bin
  float etmin;  ///< min Et value for the bin
  float etmax;  ///< max Et value for the bin
  float etamin; ///< min eta value for the bin
  float etamax; ///< max eta value for the bin
  
  
public:
  /** there are two constructors:
      - the first using the values taken from the e/gamma object
      - the second used to define the categories and the correction values
  */
  
  /** This constructor uses a string to define the category
      The string is used in the .dat file where the corrections are defined
      The syntax of the strings follows the definitions in the ECALELF ElectronCategory_class: http://ecalelfs.github.io/ECALELF/d5/d11/classElectronCategory__class.html
  */
  correctionCategory_class(TString category_); ///< constructor with name of the category according to ElectronCategory_class
  
  /// this constructor is used to assign a category to the electron/photon given values in input
  inline  correctionCategory_class(const unsigned int runNumber, const float etaEle, const float R9Ele, const float EtEle)
  {
    runmin = runNumber;
    runmax = runNumber;
    etamin = fabs(etaEle);
    etamax = fabs(etaEle);
    r9min = R9Ele;
    r9max = R9Ele;
    etmin = EtEle;
    etmax = EtEle;
  }
  
  /// for ordering of the categories
  bool operator<(const correctionCategory_class& b) const;
  
  /// for DEBUG
  friend std::ostream& operator << (std::ostream& os, const correctionCategory_class a)
  {
    os <<  a.runmin << " " << a.runmax
       << "\t" << a.etamin << " " << a.etamax
       << "\t" << a.r9min << " " << a.r9max
       << "\t" << a.etmin << " " << a.etmax;
    return os;
  };
};


//============================== 
/// map associating the category and the correction
typedef std::map < correctionCategory_class, correctionValue_class > correction_map_t;



//============================== Main class
class EnergyScaleCorrection_class
{
  
  enum fileFormat_t {
    UNKNOWN=0,
    GLOBE,
    ECALELF_TOY,
    ECALELF
  };

  enum paramSmear_t {
        kNone = 0,
        kRho,
        kPhi,
        kNParamSmear
  };
  
public:
  bool doScale, doSmearings;
  
public:
  EnergyScaleCorrection_class(std::string correctionFileName, unsigned int genSeed=0);
  EnergyScaleCorrection_class(){}; ///< dummy constructor needed in ElectronEnergyCalibratorRun2
  ~EnergyScaleCorrection_class(void);
  

//------------------------------ scales
	float ScaleCorrection(unsigned int runNumber, bool isEBEle, double R9Ele, double etaSCEle,
	                      double EtEle ) const; ///< method to get energy scale corrections

	float ScaleCorrectionUncertainty(unsigned int runNumber, bool isEBEle,
									 double R9Ele, double etaSCEle, double EtEle) const; ///< method to get scale correction uncertainties: it's stat+syst in eta x R9 categories

private:
	correctionValue_class getScaleCorrection(unsigned int runNumber, bool isEBEle, double R9Ele, double etaSCEle, double EtEle) const; ///< returns the correction value class
	float getScaleOffset(unsigned int runNumber, bool isEBEle, double R9Ele, double etaSCEle, double EtEle) const; // returns the correction value
	float getScaleStatUncertainty(unsigned int runNumber, bool isEBEle, double R9Ele, double etaSCEle, double EtEle) const; // returns the stat uncertainty
	float getScaleSystUncertainty(unsigned int runNumber, bool isEBEle, double R9Ele, double etaSCEle, double EtEle) const; // technical implementation


	void ReadFromFile(TString filename); ///<   category  "runNumber"   runMin  runMax   deltaP  err_deltaP_per_bin err_deltaP_stat err_deltaP_syst

	// this method adds the correction values read from the txt file to the map
	void AddScale(TString category_, int runMin_, int runMax_, double deltaP_, double err_deltaP_, double err_syst_deltaP);

	//============================== smearings
public:
	float getSmearingSigma(int runNumber, bool isEBEle, float R9Ele, float etaSCEle, float EtEle, paramSmear_t par, float nSigma = 0.) const;
	float getSmearingSigma(int runNumber, bool isEBEle, float R9Ele, float etaSCEle, float EtEle, float nSigma_rho, float nSigma_phi) const;


private:
	fileFormat_t smearingType_;

	correction_map_t scales, scales_not_defined;
	correction_map_t smearings, smearings_not_defined;

	void AddSmearing(TString category_, int runMin_, int runMax_, //double smearing_, double err_smearing_);
	                 double rho, double err_rho, double phi, double err_phi, double Emean, double err_Emean);
	void ReadSmearingFromFile(TString filename); ///< File structure: category constTerm alpha;
public:
	inline void SetSmearingType(fileFormat_t value)
	{
		if(value >= 0 && value <= 1) {
			smearingType_ = value;
		} else {
			smearingType_ = UNKNOWN;
		}
	};

	float getSmearingRho(int runNumber, bool isEBEle, float R9Ele, float etaSCEle, float EtEle) const; ///< public for sigmaE estimate



};


#endif
