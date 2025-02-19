#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TFile.h"

int main () 
{
  calibXMLwriter barrelWriter(EcalBarrel);
  CaloMiscalibMapEcal map ;
  std::string barrelfile ="/afs/cern.ch/user/p/presotto/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/ecal_barrel_startup.xml" ; 
  map.prefillMap () ;
  MiscalibReaderFromXMLEcalBarrel barrelreader (map) ;
  if (!barrelfile.empty ()) barrelreader.parseXMLMiscalibFile (barrelfile) ;
  
  EcalIntercalibConstants* constants = 
         new EcalIntercalibConstants (map.get ()) ;
  EcalIntercalibConstantMap imap = constants->getMap () ;
  
  std::string barrelfile2 ="EBcalib.xml" ; 
  CaloMiscalibMapEcal map2;
  map2.prefillMap ();
  MiscalibReaderFromXMLEcalBarrel barrelreader2 (map2) ;
  if (!barrelfile2.empty ()) barrelreader2.parseXMLMiscalibFile (barrelfile2) ;
  EcalIntercalibConstants* constants2 = 
         new EcalIntercalibConstants (map2.get ()) ;
  EcalIntercalibConstantMap imap2 = constants2->getMap () ;
  for (int  eta =-85;eta<=85;++eta)
   for (int phi = 1; phi<=360;++phi)
    {
     if (!EBDetId::validDetId(eta,phi)) continue;
     EBDetId eb (eta,phi,EBDetId::ETAPHIMODE);
     barrelWriter.writeLine (eb, *(imap.find(eb.rawId())) * *(imap2.find(eb.rawId()))); 
    }
}
