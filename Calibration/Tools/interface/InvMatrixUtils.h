/** \class InvMatrixUtils

    \brief various utilities

    $Date: 2008/02/25 17:39:59 $
    $Revision: 1.2 $
    $Id: InvMatrixUtils.h,v 1.2 2008/02/25 17:39:59 malberti Exp $ 
    \author $Author: malberti $
*/

#ifndef InvMatrixUtils_h
#define InvMatrixUtils_h

#include <string>
#include <map>
#include "TObject.h"
#include "TF1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#include "TFile.h"
 
//#include "ConfigParser.h"
#include "Calibration/Tools/interface/InvMatrixCommonDefs.h"

/** set the style for the printout*/
void setStyle () ;

/** search for an existing canvas with the name 
and returns the poiter to it */
TCanvas * getGlobalCanvas (std::string name = "Inv MatrixCanvas") ;

/** search for an existing TFile with the name 
and returns the poiter to it */
TFile * getGlobalTFile (std::string name = "Inv MatrixTFile.root") ;
//TFile * getGlobalTFile (const char* name) ;

/** search for an existing TFile with the name 
and saves it to disk with his name */
int saveGlobalTFile (std::string name = "Inv MatrixFile.root") ;

/** search for an existing calib matrix saved with the name 
and returns the poiter to it,
the deletion is responsiblity of the user */
CLHEP::HepMatrix * getSavedMatrix (const std::string & name) ;

/** return the impact position of the electron over ECAL */
HepGeom::Point3D<Float_t>  TBposition (const Float_t amplit[7][7], 
                                       const Float_t beamEne,
                                       const Float_t w0 = 4.0,
                                       const Float_t x0 = 8.9, //mm
                                       const Float_t a0 = 6.2,
                                       const Float_t sideX = 24.06, //mm
                                       const Float_t sideY = 22.02) ; //mm

/** get the energy in the 5x5 
from the 7x7 array around the most energetic crystal*/
double get5x5 (const Float_t energy[7][7]) ;

/** get the energy in the 3x3 
from the 7x7 array around the most energetic crystal*/
double get3x3 (const Float_t energy[7][7]) ;

/**to get the parameters from a congiguration file*/
int parseConfigFile (const TString& config) ;

/**to get the crystal number from eta and phi*/
int xtalFromEtaPhi (const int & myEta, const int & myPhi) ;

/**to get the crystal number from iEta and iPhi
iEta runs from 1 to 85
iPhi runs from 1 to 20
*/
int xtalFromiEtaiPhi (const int & iEta, const int & iPhi) ;

/** get the eta coord [0,84] */
int etaFromXtal (const int & xtal) ;

/** get the phi coord [0,19] */
int phiFromXtal (const int & xtal) ;

/** get the eta coord [1,85] */
int ietaFromXtal (const int & xtal) ;

/** get the phi coord [1,20] */
int iphiFromXtal (const int & xtal) ;

/** to read a file containing unserted integers
while avoiding comment lines */
int extract (std::vector<int> * output , const std::string & dati) ;

/** to write the calibration constants file */
int writeCalibTxt (const CLHEP::HepMatrix & AmplitudeMatrix,
                   const CLHEP::HepMatrix & SigmaMatrix,
                   const CLHEP::HepMatrix & StatisticMatrix,
                   std::string fileName = "calibOutput.txt") ;

/** to write the file fpr the CMSSW in the DB compliant format (using Energy as reference)*/
int writeCMSSWCoeff (const CLHEP::HepMatrix & amplMatrix,
                     double calibThres,
                     float ERef,
                     const CLHEP::HepMatrix & sigmaMatrix,
                     const CLHEP::HepMatrix & statisticMatrix,
                     std::string fileName = "calibOutput.txt",
                     std::string genTag = "CAL_GENTAG",
                     std::string method = "CAL_METHOD",
                     std::string version = "CAL_VERSION",
                     std::string type = "CAL_TYPE") ;

/** to write the file fpr the CMSSW in the DB compliant format 
    (using Crystal as reference) */
int writeCMSSWCoeff (const CLHEP::HepMatrix & amplMatrix,
                     double calibThres,
                     int etaRef, int phiRef,
                     const CLHEP::HepMatrix & sigmaMatrix,
                     const CLHEP::HepMatrix & statisticMatrix,
                     std::string fileName = "calibOutput.txt",
                     std::string genTag = "CAL_GENTAG",
                     std::string method = "CAL_METHOD",
                     std::string version = "CAL_VERSION",
                     std::string type = "CAL_TYPE") ;

/** translates the calib coefficients format,
    from the TB06Studies one to the CMSSSW one */
int translateCoeff (const CLHEP::HepMatrix & calibcoeff,
                    const CLHEP::HepMatrix & sigmaMatrix,
                    const CLHEP::HepMatrix & statisticMatrix,
                    std::string SMnumber = "1",
                    double calibThres = 0.01,
                    std::string fileName = "calibOutput.txt",
                    std::string genTag = "CAL_GENTAG",
                    std::string method = "CAL_METHOD",
                    std::string version = "CAL_VERSION",
                    std::string type = "CAL_TYPE") ;

/** translates the calib coefficients format,
    from the CMSSW one to the TB06Studies one */
int readCMSSWcoeff (CLHEP::HepMatrix & calibcoeff,
                    const std::string & inputFileName,
                    double defaultVal = 1.) ;

/** translates the calib coefficients format,
    from the CMSSW one to the TB06Studies one */
int readCMSSWcoeffForComparison (CLHEP::HepMatrix & calibcoeff,
                               const std::string & inputFileName) ;

/** smart profiling by double averaging */
TH1D * smartProfile (TH2F * strip, double width) ; 

/** smart profiling by fixing gaussian parameters and
    range from a first averaging */
TH1D * smartGausProfile (TH2F * strip, double width) ; 

/**
   */
TH1D * smartError (TH1D * strip) ;

/**
find the effective sigma as the half width of the sub-distribution
containing 68.3% of the total distribution
*/
double effectiveSigma (TH1F & histogram, int vSteps = 100) ;

/**
find the support of the histogram above a threshold
return the min and max bins
*/
std::pair<int,int> findSupport (TH1F & histogram, double thres = 0.) ;

/** 
transfers a CLHEP matrix into a double array
with the size of a supermodule
*/
void mtrTransfer (double output[SCMaxEta][SCMaxPhi], 
                  CLHEP::HepMatrix * input, 
                  double Default) ;

/**
reset the matrices f the size of a supermodule
*/
template <class Type>
void
mtrReset (Type superModules[SCMaxEta][SCMaxPhi], const Type val)
  {
    for (int e = 0 ; e < SCMaxEta ; ++e)
      for (int p = 0 ; p < SCMaxPhi ; ++p)
        {         
          superModules[e][p] = val ;
        }
  }

/**
correction for eta containment for 3*3 cluster */
double etaCorrE1E9 (int eta) ;
/**
correction for eta containment for 7*7 cluster */
double etaCorrE1E49 (int eta) ;
/**
correction for eta containment for 5*5 cluster */
double etaCorrE1E25 (int eta) ;

#endif

