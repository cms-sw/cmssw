#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"

#include "Calibration/Tools/bin/trivialParser.h"

#include "TH2.h"
#include "TH1.h"
#include "TFile.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TCanvas.h"

#define PI_GRECO 3.14159265

int main (int argc, char* argv[]) 
{
  std::string fileName = argv[1] ;
  std::cerr << "parsing coeff file: " << fileName << std::endl ;

  int EEradStart = 15 ;
  int EEradEnd = 50 ;
  int EEphiStart = 20 ;
  int EEphiEnd = 45 ;

  //PG open the XML file
  CaloMiscalibMapEcal map ;
  map.prefillMap () ;
  MiscalibReaderFromXMLEcalEndcap endcapreader (map) ;
  if (!fileName.empty ()) endcapreader.parseXMLMiscalibFile (fileName) ;
  EcalIntercalibConstants* constants = 
         new EcalIntercalibConstants (map.get ()) ;
  EcalIntercalibConstantMap imap = 
      constants->getMap () ;  

  TH1F coeffDistr ("coeffDistrEE","coeffDistrEE",500,0,2) ;
  TH2F coeffMap ("coeffMapEE","coeffMapEE",101,0,101,101,0,101) ;
  coeffMap.SetStats (0) ;

  // ECAL barrel
  for (int ix = 1 ; ix <= 100 ; ++ix)
   for (int iy = 1 ; iy <= 100 ; ++iy)
    {      
      int rad = static_cast<int> (sqrt ((ix - 50) * (ix - 50) +
                                        (iy - 50) * (iy - 50))) ;
      if (rad < EEradStart || rad > EEradEnd) continue ;
      double phiTemp = atan2 (iy - 50, ix - 50) ;
      if (phiTemp < 0) phiTemp += 2 * PI_GRECO ;
      int phi = static_cast<int> ( phiTemp * 180 / PI_GRECO) ;
      if (phi < EEphiStart || phi > EEphiEnd) continue ;

      if (!EEDetId::validDetId (ix,iy,1)) continue ;
      EEDetId det = EEDetId (ix,iy,1,EEDetId::XYMODE) ;
      double coeff = (*(imap.find (det.rawId ())));
      std::cerr << "found coeff " << ix << " " << iy 
                << " " << coeff << std::endl ;
      coeffDistr.Fill (coeff) ;
      coeffMap.Fill (ix,iy,coeff) ;
    } // ECAL barrel

  gROOT->SetStyle ("Plain") ;
  gStyle->SetPalette (1) ;
  TCanvas c1 ;
  c1.SetGrid () ;
  
  coeffMap.GetZaxis ()->SetRangeUser (0,2) ;
  coeffMap.GetXaxis ()->SetTitle ("ix") ;
  coeffMap.GetYaxis ()->SetTitle ("iy") ;
  coeffMap.Draw ("COLZ") ;
  c1.Print ("coeffMapEE.gif","gif") ;
  c1.SetLogy () ;
  coeffDistr.GetXaxis ()->SetTitle ("calib coeff EE") ;
  coeffDistr.SetFillColor (8) ;
  coeffDistr.Draw () ;
  c1.Print ("coeffDistrEE.gif","gif") ;
     
  TFile out ("coeffEE.root","recreate") ;
  coeffDistr.Write () ;
  coeffMap.Write () ;
  out.Close () ;
  
}
