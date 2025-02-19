/** 
    $Date: 2009/02/26 13:48:08 $
    $Revision: 1.4 $
    $Id: InvMatrixUtils.cc,v 1.4 2009/02/26 13:48:08 argiro Exp $ 
    \author $Author: argiro $
*/

#include "Calibration/Tools/interface/InvMatrixUtils.h"
#include "Calibration/Tools/interface/InvMatrixCommonDefs.h"
#include "TStyle.h"
#include "TROOT.h"
#include "CLHEP/Geometry/Point3D.h"
//#include "ConfigParser.h"
#include "Calibration/Tools/interface/matrixSaver.h"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>

/** set the style for the printout*/
void setStyle ()
{
  gROOT->SetStyle ("Plain") ;
  gStyle->SetTextSize(0.5);  
  //gStyle->SetOptStat (1111111) ;
  gStyle->SetOptStat (0) ;
  //gStyle->SetOptFit (1111) ;
  gStyle->SetOptFit (0) ;
  gStyle->SetTitleBorderSize (0) ;
  gStyle->SetTitleX (0.08) ;
  gStyle->SetTitleY (0.97) ;
  gStyle->SetPalette (1,0) ;
  gStyle->SetStatColor (0) ;
  gStyle->SetFrameFillStyle (0) ;
  gStyle->SetFrameFillColor (0) ;
  return ;
}


// -----------------------------------------------------------------


TCanvas * getGlobalCanvas (std::string name) 
{
  setStyle () ;
  TCanvas * globalCanvas = static_cast<TCanvas*> 
                           (gROOT->FindObject (name.c_str ())) ;
  if (globalCanvas)
    {
      globalCanvas->Clear () ;
      globalCanvas->UseCurrentStyle () ;
      globalCanvas->SetWindowSize (700, 600) ;
    
    }
  else
    {
      globalCanvas = new TCanvas (name.c_str (),name.c_str (), 700, 600) ;
    }
  return globalCanvas ;
}

// -----------------------------------------------------------------


TFile * getGlobalTFile (std::string name) 
{
//    std::cout << "writing " << name << std::endl ;
//    setStyle () ;
    TFile * globalTFile = (TFile*) gROOT->FindObject (name.c_str()) ;
    if (!globalTFile)
        {
//        std::cout << "does not exist. creating it " << std::endl;
        globalTFile = new TFile (name.c_str(),"RECREATE") ;
        }
    
    return globalTFile ;
}


// -----------------------------------------------------------------


int saveGlobalTFile (std::string name) 
{
  TFile * globalTFile = static_cast<TFile*> 
                         (gROOT->FindObject (name.c_str ())) ;
  if (!globalTFile) return 1 ;
  globalTFile->Write () ;
  globalTFile->Close () ;
  delete globalTFile ;
  return 0 ;
}


// -----------------------------------------------------------------


CLHEP::HepMatrix * getSavedMatrix (const std::string & name) 
{
  matrixSaver reader ;
  CLHEP::HepMatrix * savedMatrix ;
  if (reader.touch (name)) 
    {
       savedMatrix = static_cast<CLHEP::HepMatrix *> (
                       reader.getMatrix (name)
                     );
    }
  else
    {
       savedMatrix = new CLHEP::HepMatrix (SCMaxEta,SCMaxPhi,0) ;
    }

  return savedMatrix ;
}



//========================================================================
//da usare 
//HepGeom::Point3D<double> TBimpactPoint = TBposition (amplitude,m_beamEnergy) ;

/* 
dove trovare il codice di Chiara in CMSSW, per la ricostruzione
della posizione:

http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/RecoTBCalo/EcalTBAnalysisCoreTools/src/TBPositionCalc.cc?rev=1.1&cvsroot=CMSSW&content-type=text/vnd.viewcvs-markup

*/

/** return the impact position of the electron over ECAL*/
HepGeom::Point3D<Float_t>
TBposition (const Float_t amplit[7][7], 
            const Float_t beamEne,
            const Float_t w0,
            const Float_t x0,
            const Float_t a0,
            const Float_t sideX, // crystal geometry, in mm
            const Float_t sideY)
{
  // variables
    Float_t caloX = 0. ;
    Float_t caloY = 0. ;
    Float_t sumWeight = 0. ;
    Float_t depth = x0 * (log (beamEne)+ a0) ;  // shower depthh, in mm
    Float_t sin3 = 0.052335956 ; // sin (3 degrees) , sin3 = sin(3.*3.141592654/180.)
    
    Float_t invE3x3 = 1. / get3x3 (amplit) ;
    
  // loop over 3x3 crystals
  for (int eta = 2; eta <= 4; eta++)
      {
           for (int phi = 2; phi <= 4; phi++)
           {
            Float_t weight = log( amplit[eta][phi] * invE3x3) + w0 ;
            if ( weight>0 )
               {
                 caloX +=  (eta-3) * sideX * weight;
                 caloY -=  (phi-3) * sideY * weight;  
                 sumWeight += weight;
               }
            }
      }
      
  caloX /=sumWeight;
  caloY /=sumWeight;
  
  // correction for shower depthh
  caloX -= depth*sin3;
  caloY -= depth*sin3;
  
  // FIXME the z set to zero
  HepGeom::Point3D<Float_t> TBposition (caloX, caloY, 0) ;
  
  return TBposition ;  
  
}   

// -----------------------------------------------------------------


/** get the energy in the 5x5 
from the 7x7 array around the most energetic crystal*/
double get5x5 (const Float_t energy[7][7]) 
{
  double total = 0. ;

  for (int eta=1 ; eta<6 ; ++eta)
    for (int phi=1 ; phi<6 ; ++phi)
      total += energy[eta][phi] ;

  return total ;
}


// -----------------------------------------------------------------


/** get the energy in the 3x3 
from the 7x7 array around the most energetic crystal*/
double get3x3 (const Float_t energy[7][7]) 
{
  double total = 0. ;

  for (int eta=2 ; eta<5 ; ++eta)
    for (int phi=2 ; phi<5 ; ++phi)
      total += energy[eta][phi] ;

  return total ;
}


// -----------------------------------------------------------------


/**to get the parameters from a congiguration file*/
/* int parseConfigFile (const TString& config)
{
    if (gConfigParser) return 1 ;
    
    std::cout << "parsing "
        << config << " file"
        << std::endl ;
    
    
    gConfigParser = new ConfigParser () ;
    if ( !(gConfigParser->init(config)) )
        {
        std::cout << "Analysis::parseConfigFile: Could not open configuration file "
        << config << std::endl;
        perror ("Analysis::parseConfigFile: ") ;
        exit (-1) ;
        }
    gConfigParser->print () ;
    return 0 ;
}*/


// -----------------------------------------------------------------


// per un certo eta, il cristallo puo' essere qualsiasi intero
// Calibrationtra xmin e xmax
// lo posso fissare solo sfruttando anche phi 
int xtalFromEtaPhi (const int & myEta, const int & myPhi)
{
    int xMin = 20 * myEta + 1 ;  
    int xMax = 20 * (myEta + 1) + 1 ;
    
    int myCryst = 999999 ;
    
    for (int x = xMin ; x < xMax ; x++)
        {
          if (phiFromXtal (x) == myPhi)
            myCryst = x ;
        }
    return myCryst ;
}


// -----------------------------------------------------------------


int xtalFromiEtaiPhi (const int & iEta, const int & iPhi) 
{
  assert (iEta >= 1) ;
  assert (iEta <= 85) ;
  assert (iPhi >= 1) ;
  assert (iPhi <= 20) ;
  return 20 * (iEta-1) + 21 - iPhi ;
}


// -----------------------------------------------------------------


int etaFromXtal (const int & xtal) 
{
//  return floor (static_cast<double> ((xtal-1) / 20)) ;
  return int (floor ((xtal-1) / 20) );
}


// -----------------------------------------------------------------


int phiFromXtal (const int & xtal) 
{
  int phi = (xtal-1) - 20 * etaFromXtal (xtal) ;
  return (20 - phi - 1) ;
}


// -----------------------------------------------------------------


int ietaFromXtal (const int & xtal) 
{
  return etaFromXtal (xtal) + 1 ;
}


// -----------------------------------------------------------------


int iphiFromXtal (const int & xtal) 
{
  return phiFromXtal (xtal) + 1 ;
}


// -----------------------------------------------------------------


int extract (std::vector<int> * output , const std::string & dati) 
  {
    std::ifstream _dati (dati.c_str ()) ;
    // loop over the file
    while (!_dati.eof())
      {
        // get the line
        std::string dataline ;
        do { getline (_dati, dataline,'\n') ; } 
        while (*dataline.begin () == '#') ;
        std::stringstream linea (dataline) ;
        // loop over the line
        while (!linea.eof ())
          {
            int buffer = -1 ;
            linea >> buffer ;
            if (buffer != -1) output->push_back (buffer) ;
          } // loop over the line
      } // loop over the file     
    return output->size () ;
  }


// -----------------------------------------------------------------


// FIXME questi eta, phi sono quelli della matrice CLHEP, 
// FIXME non quelli del super-modulo, giusto?
int writeCalibTxt (const CLHEP::HepMatrix & AmplitudeMatrix,
                   const CLHEP::HepMatrix & SigmaMatrix,
                   const CLHEP::HepMatrix & StatisticMatrix,
                   std::string fileName)
{
  // look for the reference crystal
  double reference = 0. ;
  for (int eta = 0 ; eta<SCMaxEta ; ++eta)
    for (int phi = 0 ; phi<SCMaxPhi ; ++phi)
      {
        if (AmplitudeMatrix[eta][phi] && 
            SigmaMatrix[eta][phi] < 100 /*FIXME sigmaCut*/) 
          {
            reference = AmplitudeMatrix[eta][phi] ;
            std::cout << "[InvMatrixUtils][writeCalibTxt] reference crystal: "
                      << "(" << eta << "," << phi << ") -> "
                      << reference << "\n" ;
            break ;
          }
      }
  if (!reference)
    {
      std::cerr << "ERROR: no calibration coefficients found" << std::endl ;
      return 1 ;
    }

  // open the file for output
  std::ofstream txt_outfile ;
  txt_outfile.open (fileName.c_str ()) ;  
  txt_outfile << "# xtal\tcoeff\tsigma\tevt\tisGood\n" ; 

  // loop over the crystals
  for (int eta = 0 ; eta<SCMaxEta ; ++eta)
    for (int phi = 0 ; phi<SCMaxPhi ; ++phi)
      {
        int isGood = 1 ;
        if (AmplitudeMatrix[eta][phi] == 0) isGood = 0 ;
        if (SigmaMatrix[eta][phi] > 100 /*FIXME sigmaCut*/) isGood = 0 ;
        txt_outfile << xtalFromEtaPhi (eta,phi) 
                    << "\t" << AmplitudeMatrix[eta][phi]/reference 
                    << "\t" << SigmaMatrix[eta][phi]
                    << "\t" << StatisticMatrix[eta][phi]
                    << "\t" << isGood <<"\n" ;
      }

  // save and close the file 
  txt_outfile.close () ;
  return 0 ;
}


// -----------------------------------------------------------------


int writeCMSSWCoeff (const CLHEP::HepMatrix & amplMatrix,
                     double calibThres,
                     float ERef,
                     const CLHEP::HepMatrix & sigmaMatrix,
                     const CLHEP::HepMatrix & statisticMatrix,
                     std::string fileName,
                     std::string genTag,
                     std::string method,
                     std::string version,
                     std::string type) 
{
  // open the file for output
  std::ofstream txt_outfile ;
  txt_outfile.open (fileName.c_str ()) ;  
  txt_outfile << "1\n" ; // super-module number 
  txt_outfile << "-1\n" ; // number of events
  txt_outfile << genTag << "\n" ;
  txt_outfile << method << "\n" ;
  txt_outfile << version << "\n" ;
  txt_outfile << type << "\n" ;

  double reference = ERef ;

  // loop over crystals
  for (int eta = 0 ; eta < SCMaxEta ; ++eta)
    for (int phi = 0 ; phi < SCMaxPhi ; ++phi)
      {
        if (amplMatrix[eta][phi] <= calibThres) 
          txt_outfile << xtalFromiEtaiPhi (eta+1,phi+1) 
                      << "\t" << 1 
                      << "\t" << -1
                      << "\t" << -1
                      << "\t" << 0 <<"\n" ;
        else  
          txt_outfile << xtalFromiEtaiPhi (eta+1,phi+1) 
                      << "\t" << reference / amplMatrix[eta][phi] 
                      << "\t" << sigmaMatrix[eta][phi]
                      << "\t" << statisticMatrix[eta][phi]
                      << "\t" << 1 <<"\n" ;
      } // loop over crystals

  // save and close the file 
  txt_outfile.close () ;
  return 0 ;
}                     


// -----------------------------------------------------------------


int writeCMSSWCoeff (const CLHEP::HepMatrix & amplMatrix,
                     double calibThres,
                     int etaRef, int phiRef,
                     const CLHEP::HepMatrix & sigmaMatrix,
                     const CLHEP::HepMatrix & statisticMatrix,
                     std::string fileName,
                     std::string genTag,
                     std::string method,
                     std::string version,
                     std::string type) 
{
  // open the file for output
    std::ofstream txt_outfile ;
    txt_outfile.open (fileName.c_str ()) ;  
    txt_outfile << "1\n" ; // super-module number 
    txt_outfile << "-1\n" ; // number of events
    txt_outfile << genTag << "\n" ;
    txt_outfile << method << "\n" ;
    txt_outfile << version << "\n" ;
    txt_outfile << type << "\n" ;
    
    if (amplMatrix[etaRef-1][phiRef-1] == 0)
        {
        std::cerr << "The reference crystal: ("
        << etaRef << "," << phiRef
        << ") is out of range\n" ;
        return 1 ;
        }
    double reference = amplMatrix[etaRef-1][phiRef-1] ;
    
  // loop over crystals
    for (int eta = 0 ; eta < SCMaxEta ; ++eta)
        for (int phi = 0 ; phi < SCMaxPhi ; ++phi)
            {
            if (amplMatrix[eta][phi] <= calibThres) 
                txt_outfile << xtalFromiEtaiPhi (eta+1,phi+1) 
                    << "\t" << 1 
                    << "\t" << -1
                    << "\t" << -1
                    << "\t" << 0 <<"\n" ;
            else  
                txt_outfile << xtalFromiEtaiPhi (eta+1,phi+1) 
                    << "\t" << reference / amplMatrix[eta][phi] 
                    << "\t" << sigmaMatrix[eta][phi]
                    << "\t" << statisticMatrix[eta][phi]
                    << "\t" << 1 <<"\n" ;
            } // loop over crystals
            
  // save and close the file 
    txt_outfile.close () ;
    return 0 ;
}                    


// -----------------------------------------------------------------


int translateCoeff (const CLHEP::HepMatrix & calibcoeff,
                    const CLHEP::HepMatrix & sigmaMatrix,
                    const CLHEP::HepMatrix & statisticMatrix,
                    std::string SMnumber,
                    double calibThres,
                    std::string fileName,
                    std::string genTag,
                    std::string method,
                    std::string version,
                    std::string type) 
{
    // open the file for output
    std::ofstream txt_outfile ;
    txt_outfile.open (fileName.c_str ()) ;  
    txt_outfile << SMnumber << "\n" ; // super-module number 
    txt_outfile << "-1\n" ; // number of events
    txt_outfile << genTag << "\n" ;
    txt_outfile << method << "\n" ;
    txt_outfile << version << "\n" ;
    txt_outfile << type << "\n" ;
    
    // loop over crystals
    for (int eta = 0 ; eta < SCMaxEta ; ++eta)
        for (int phi = 0 ; phi < SCMaxPhi ; ++phi)
            {
            if (calibcoeff[eta][phi] < calibThres) 
              {
                txt_outfile << xtalFromiEtaiPhi (eta+1,phi+1) 
                    << "\t" << 1 
                    << "\t" << -1
                    << "\t" << -1
                    << "\t" << 0 <<"\n" ;
                std::cout << "[translateCoefff][" << SMnumber 
                    << "]\t WARNING crystal " << xtalFromiEtaiPhi (eta+1,phi+1)
                    << " calib coeff below threshold: " 
                    << "\t" << 1 
                    << "\t" << -1
                    << "\t" << -1
                    << "\t" << 0 <<"\n" ;
              }
            else  
                txt_outfile << xtalFromiEtaiPhi (eta+1,phi+1) 
                    << "\t" << calibcoeff[eta][phi] 
                    << "\t" << sigmaMatrix[eta][phi]
                    << "\t" << statisticMatrix[eta][phi]
                    << "\t" << 1 <<"\n" ;
            } // loop over crystals
            
    // save and close the file 
    txt_outfile.close () ;
    return 0 ;
}                    


// -----------------------------------------------------------------


int readCMSSWcoeff (CLHEP::HepMatrix & calibcoeff,
                    const std::string & inputFileName,
                    double defaultVal) 
{
    std::ifstream CMSSWfile ;
    CMSSWfile.open (inputFileName.c_str ()) ; 
    std::string buffer ; 
    CMSSWfile >> buffer ;  
    CMSSWfile >> buffer ; 
    CMSSWfile >> buffer ;
    CMSSWfile >> buffer ;
    CMSSWfile >> buffer ;
    CMSSWfile >> buffer ;
    while (!CMSSWfile.eof ())
      {
        int xtalnum ;
        CMSSWfile >> xtalnum ;
        double coeff ;
        CMSSWfile >> coeff ;
        double buffer ;
        CMSSWfile >> buffer ;
        int good ;
        CMSSWfile >> good ;
        CMSSWfile >> good ;
        if (!good) coeff = defaultVal ; //FIXME 0 o 1?
        calibcoeff[etaFromXtal (xtalnum)][phiFromXtal (xtalnum)] = coeff ;          
      }       
    return 0 ;

}                    


// -----------------------------------------------------------------


int readCMSSWcoeffForComparison (CLHEP::HepMatrix & calibcoeff,
                    const std::string & inputFileName) 
{
    std::ifstream CMSSWfile ;
    CMSSWfile.open (inputFileName.c_str ()) ; 
    std::string buffer ; 
    CMSSWfile >> buffer ;  
    CMSSWfile >> buffer ; 
    CMSSWfile >> buffer ;
    CMSSWfile >> buffer ;
    CMSSWfile >> buffer ;
    CMSSWfile >> buffer ;
    while (!CMSSWfile.eof ())
      {
        int xtalnum ;
        CMSSWfile >> xtalnum ;
        double coeff ;
        CMSSWfile >> coeff ;
        double buffer ;
        CMSSWfile >> buffer ;
        int good ;
        CMSSWfile >> good ;
        CMSSWfile >> good ;
        if (!good) coeff = 0. ; //FIXME 0 o 1?
        calibcoeff[etaFromXtal (xtalnum)][phiFromXtal (xtalnum)] = coeff ;          
      }       
    return 0 ;

}                    


// -----------------------------------------------------------------


TH1D * smartProfile (TH2F * strip, double width) 
{
  TProfile * stripProfile = strip->ProfileX () ;

  // (from FitSlices of TH2.h)

  double xmin = stripProfile->GetXaxis ()->GetXmin () ;
  double xmax = stripProfile->GetXaxis ()->GetXmax () ;
  int profileBins = stripProfile->GetNbinsX () ;

  std::string name = strip->GetName () ;
  name += "_smart" ; 
  TH1D * prof = new TH1D
      (name.c_str (),strip->GetTitle (),profileBins,xmin,xmax) ;
   
  int cut = 0 ; // minimum number of entries per fitted bin
  int nbins = strip->GetXaxis ()->GetNbins () ;
  int binmin = 1 ;
  int ngroup = 1 ; // bins per step
  int binmax = nbins ;

  // loop over the strip bins
  for (int bin=binmin ; bin<=binmax ; bin += ngroup) 
    {
      TH1D *hpy = strip->ProjectionY ("_temp",bin,bin+ngroup-1,"e") ;
      if (hpy == 0) continue ;
      int nentries = Int_t (hpy->GetEntries ()) ;
      if (nentries == 0 || nentries < cut) {delete hpy ; continue ;} 
 
      Int_t biny = bin + ngroup/2 ;
      
      hpy->GetXaxis ()->SetRangeUser ( hpy->GetMean () - width * hpy->GetRMS (), 
                                       hpy->GetMean () + width * hpy->GetRMS ()) ;         
      prof->Fill (strip->GetXaxis ()->GetBinCenter (biny),
                  hpy->GetMean ()) ;       
      prof->SetBinError (biny,hpy->GetRMS()) ;
      
      delete hpy ;
    } // loop over the bins

  delete stripProfile ;
  return prof ;
}


// -----------------------------------------------------------------


TH1D * smartGausProfile (TH2F * strip, double width) 
{
  TProfile * stripProfile = strip->ProfileX () ;

  // (from FitSlices of TH2.h)

  double xmin = stripProfile->GetXaxis ()->GetXmin () ;
  double xmax = stripProfile->GetXaxis ()->GetXmax () ;
  int profileBins = stripProfile->GetNbinsX () ;

  std::string name = strip->GetName () ;
  name += "_smartGaus" ; 
  TH1D * prof = new TH1D
      (name.c_str (),strip->GetTitle (),profileBins,xmin,xmax) ;
   
  int cut = 0 ; // minimum number of entries per fitted bin
  int nbins = strip->GetXaxis ()->GetNbins () ;
  int binmin = 1 ;
  int ngroup = 1 ; // bins per step
  int binmax = nbins ;

  // loop over the strip bins
  for (int bin=binmin ; bin<=binmax ; bin += ngroup) 
    {
      TH1D *hpy = strip->ProjectionY ("_temp",bin,bin+ngroup-1,"e") ;
      if (hpy == 0) continue ;
      int nentries = Int_t (hpy->GetEntries ()) ;
      if (nentries == 0 || nentries < cut) {delete hpy ; continue ;} 
 
      Int_t biny = bin + ngroup/2 ;

      TF1 * gaussian = new TF1 ("gaussian","gaus", hpy->GetMean () - width * hpy->GetRMS (),
                                                   hpy->GetMean () + width * hpy->GetRMS ()) ; 
      gaussian->SetParameter (1,hpy->GetMean ()) ;
      gaussian->SetParameter (2,hpy->GetRMS ()) ;
      hpy->Fit ("gaussian","RQL") ;           

      hpy->GetXaxis ()->SetRangeUser ( hpy->GetMean () - width * hpy->GetRMS (), 
                                       hpy->GetMean () + width * hpy->GetRMS ()) ;         
      prof->Fill (strip->GetXaxis ()->GetBinCenter (biny),
                  gaussian->GetParameter (1)) ;       
      prof->SetBinError (biny,gaussian->GetParameter (2)) ;
      
      delete gaussian ;
      delete hpy ;
    } // loop over the bins

  delete stripProfile ;
  return prof ;
}


// -----------------------------------------------------------------


TH1D * smartError (TH1D * strip)
{

  double xmin = strip->GetXaxis ()->GetXmin () ;
  double xmax = strip->GetXaxis ()->GetXmax () ;
  int stripsBins = strip->GetNbinsX () ;

  std::string name = strip->GetName () ;
  name += "_error" ; 
  TH1D * error = new TH1D
      (name.c_str (),strip->GetTitle (),stripsBins,xmin,xmax) ;

  int binmin = 1 ;
  int ngroup = 1 ; // bins per step
  int binmax = stripsBins ;
  for (int bin=binmin ; bin<=binmax ; bin += ngroup) 
    {
      double dummyError = strip->GetBinError (bin) ; 
      error->SetBinContent (bin,dummyError) ;
    }
  return error;  
}


// -----------------------------------------------------------------


double effectiveSigma (TH1F & histogram, int vSteps) 
{
  double totInt = histogram.Integral () ;
  int maxBin = histogram.GetMaximumBin () ;
  int maxBinVal = int(histogram.GetBinContent (maxBin)) ;
  int totBins = histogram.GetNbinsX () ;
  double area = totInt ;
  double threshold = 0 ;
  double vStep = maxBinVal / vSteps ;
  int leftBin = 1 ;
  int rightBin = totBins - 1 ;
  //loop over the vertical range
  while (area/totInt > 0.683)
    {
      threshold += vStep ;
      // loop toward the left
      for (int back = maxBin ; back > 0 ; --back)
         {
           if (histogram.GetBinContent (back) < threshold)
             {
               leftBin = back ;
               break ;
             }
         } // loop toward the left

      // loop toward the right   
      for (int fwd = maxBin ; fwd < totBins ; ++fwd)
         {
           if (histogram.GetBinContent (fwd) < threshold)
             {
               rightBin = fwd ;
               break ;
             }
         } // loop toward the right
       area = histogram.Integral (leftBin,rightBin) ;
    } //loop over the vertical range

  histogram.GetXaxis ()->SetRange (leftBin,rightBin) ;
  // double sigmaEff = histogram.GetRMS () ;
  double halfWidthRange = 0.5 * (histogram.GetBinCenter (rightBin) - histogram.GetBinCenter (leftBin)) ;
  return halfWidthRange ;
}


// -----------------------------------------------------------------


std::pair<int,int> findSupport (TH1F & histogram, double thres) 
{
  int totBins = histogram.GetNbinsX () ;
  if (thres >= histogram.GetMaximum ()) 
    return std::pair<int,int> (0, totBins) ;

  int leftBin = totBins - 1 ;
  // search from left for the minimum
  for (int bin=1 ; bin<totBins ; ++bin)
    {
      if (histogram.GetBinContent (bin) > thres)
        {
          leftBin = bin ;
          break ; 
        }    
    } // search from left for the minimum
  int rightBin = 1 ;
  // search from right for the maximum
  for (int bin=totBins - 1 ; bin> 0 ; --bin)
    {
      if (histogram.GetBinContent (bin) > thres)
        {
          rightBin = bin ;
          break ; 
        }    
    } // search from right for the maximum
  return std::pair<int,int> (leftBin,rightBin) ;  
}


// -----------------------------------------------------------------


void
mtrTransfer (double output[SCMaxEta][SCMaxPhi], 
             CLHEP::HepMatrix * input, 
             double Default)
{
  for (int eta = 0 ; eta < SCMaxEta ; ++eta)                              
    for (int phi = 0 ; phi < SCMaxPhi ; ++phi)                              
      {
        if ((*input)[eta][phi]) 
        output[eta][phi] = (*input)[eta][phi] ;
        else output[eta][phi] = Default ;
      }
  return ;
}

// -----------------------------------------------------------------

double etaCorrE1E25 (int eta)
{
    double p0 = 0.807883 ;
    double p1 = 0.000182551 ;
    double p2 = -5.76961e-06 ;
    double p3 = 7.41903e-08 ;
    double p4 = -2.25384e-10 ;
    
    double corr ;
    if (eta < 6) corr = p0 ;
    else corr = p0 + p1*eta + p2*eta*eta + p3*eta*eta*eta + p4*eta*eta*eta*eta;
    return corr/p0 ;
}
// -----------------------------------------------------------------

double etaCorrE1E49 (int eta)
{
    double p0 = 0.799895 ;
    double p1 = 0.000235487 ;
    double p2 = -8.26496e-06 ;
    double p3 = 1.21564e-07 ;
    double p4 = -4.83286e-10 ;
    
    double corr ;
    if (eta < 8) corr = p0 ;
    else corr = p0 + p1*eta + p2*eta*eta + p3*eta*eta*eta + p4*eta*eta*eta*eta;
    return corr/p0 ;
}
// -----------------------------------------------------------------

double etaCorrE1E9 (int eta)
{
    if (eta < 4) return 1.0 ;
    // grazie Paolo
    double p0 = 0.834629 ;
    double p1 = 0.00015254 ;
    double p2 = -4.91784e-06 ;
    double p3 = 6.54652e-08 ;
    double p4 = -2.4894e-10 ;
    
    double corr ;
    if (eta < 6) corr = p0 ;
    else corr = p0 + p1*eta + p2*eta*eta + p3*eta*eta*eta + p4*eta*eta*eta*eta;
    return corr/p0 ;
}

