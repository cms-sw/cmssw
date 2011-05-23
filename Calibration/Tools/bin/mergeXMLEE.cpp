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

//#include "Calibration/EcalAlCaRecoProducers/interface/trivialParser.h"
#include "Calibration/Tools/bin/trivialParser.h"

#include "TH2.h"
#include "TH1.h"
#include "TFile.h"

#include <sstream>
#include <string>



inline double degrees (double radiants)
 {
  return radiants * 180 * (1/M_PI) ;
 }


// ------------------------------------------------------------------------


int EEregionCheck (int ics, int ips,
                   int radStart, int radEnd,
                   int phiStart, int phiEnd)  
{
  int x = ics-50;
  int y = ips-50;
  double radius2 = x*x + y*y ;
  if (radius2 < 10*10) return 1;  //center of the donut
  if (radius2 > 50*50) return 1;  //outer part of the donut
  if (radius2 < radStart * radStart) return 2 ;
  if (radius2 >= radEnd * radEnd) return 2 ;
  double phi = atan2 (static_cast<double> (y),static_cast<double> (x));
  phi = degrees (phi);
  if (phi < 0) phi += 360; 
  if (phiStart < phiEnd 
     && phi > phiStart && phi < phiEnd ) return 0; 
  if (phiStart > phiEnd 
      && (phi > phiStart|| phi < phiEnd )) return 0; 
   return 3;
}


// ------------------------------------------------------------------------


inline int radShifter (const int radOld) 
   {
     if (radOld < 0) return radOld + 85 ;
     else if (radOld > 0) return radOld + 84 ;
     assert(false);
   }


// ------------------------------------------------------------------------


int main (int argc, char* argv[]) 
{
  std::cout << "parsing cfg file: " << argv[1] << std::endl ;
  trivialParser configParams (static_cast<std::string> (argv[1])) ;
  
  int IMAEEradStart = static_cast<int> (configParams.getVal ("IMAEEradStart")) ;
  int IMAEEradEnd = static_cast<int> (configParams.getVal ("IMAEEradEnd")) ;
  int IMAEEradWidth = static_cast<int> (configParams.getVal ("IMAEEradWidth")) ;
  int IMAEEphiStart = static_cast<int> (configParams.getVal ("IMAEEphiStart")) ;
  int IMAEEphiEnd = static_cast<int> (configParams.getVal ("IMAEEphiEnd")) ;

//  std::cerr << "[PG] IMAEEradStart = " << IMAEEradStart << std::endl ;
//  std::cerr << "[PG] IMAEEradEnd = " << IMAEEradEnd << std::endl ;
//  std::cerr << "[PG] IMAEEradWidth = " << IMAEEradWidth << std::endl ;
//  std::cerr << "[PG] IMAEEphiStart = " << IMAEEphiStart << std::endl ;
//  std::cerr << "[PG] IMAEEphiEnd = " << IMAEEphiEnd << std::endl ;
//  std::cerr << "[PG] IMAEEphiWidth = " << IMAEEphiWidth << std::endl ;

  std::string coeffFolder = argv[2] ;

  std::map<int, EcalIntercalibConstantMap> recalibrators ; 
  //PG FIXME  c'e' il copy ctor della CaloMiscalibMapEcal?
 
  //PG loop on EB rad indexes
  for (int radIndex = IMAEEradStart ; 
       radIndex < IMAEEradEnd ; 
       radIndex += IMAEEradWidth)
    {
      int currentIndex = radIndex ;
      int nextIndex = radIndex + IMAEEradWidth ;

      //PG compute the values of the limits
      //PG FIXME questo forse non e' sufficiente, bisogna anche considerare il caso
      //PG FIXME in cui currentIndex e' positivo e effradIndex start no 
      //PG FIXME lo stesso vale per l'end e il nexindex
      //PG FIXME lo stesso vale per gli script perl
//      int effradIndexStart = currentIndex - IMAEEradWidth ;
//      if (effradIndexStart < 0) { effradIndexStart = 0 ; }
//      int effradIndexEnd = nextIndex + IMAEEradWidth ;
//      if (effradIndexEnd > 50) { effradIndexEnd = 50 ; }
    
      //PG build the filename
      std::stringstream nomeFile ;
      nomeFile << coeffFolder << "/EEcalibCoeff_" << currentIndex 
               << "-" << nextIndex << ".xml" ;
      std::string fileName = nomeFile.str () ;      
//      std::cerr << "PG nomefile: " << fileName << std::endl ;

      //PG open the XML file
      CaloMiscalibMapEcal map ;
      map.prefillMap () ;
      MiscalibReaderFromXMLEcalEndcap endcapreader (map) ;
      if (!fileName.empty ()) endcapreader.parseXMLMiscalibFile (fileName) ;
      EcalIntercalibConstants* constants = 
             new EcalIntercalibConstants (map.get ()) ;
      recalibrators[currentIndex] = constants->getMap () ;  
    } //PG loop on EB rad indexes

  //PG prepare the XML to be saved
  
  //PG this command outputs an XML file with a fixed name
  calibXMLwriter endcapWriter (EcalEndcap) ;
  //PG loop on EB rad slices
  for (std::map<int, EcalIntercalibConstantMap>::const_iterator itMap = 
          recalibrators.begin () ;
       itMap != recalibrators.end () ;
       ++itMap)
    {
      //PG compute the values of the limits
      //PG FIXME questo forse non e' sufficiente, bisogna anche considerare il caso
      //PG FIXME in cui currentIndex e' positivo e effradIndex start no 
      //PG FIXME lo stesso vale per l'end e il nexindex
      //PG FIXME lo stesso vale per gli script perl

      //PG non so scegliere da dove partire, questo e' un disastro
      //PG il problema e' dei bordi
      //PG forse la cosa migliore e' salvare i xml file con un nome che dica dei coeff buoni,
      //PG non di quelli calcolati - direi che e' la cosa giusta

      //PG loop over x
      for (int ix = 0 ; ix < 100 ; ++ix)
      //PG loop over y
      for (int iy = 0 ; iy < 100 ; ++iy)
        {
          //PG select the subregion of interest
          if (EEregionCheck (ix,iy,
                             itMap->first,itMap->first+IMAEEradWidth,
                             IMAEEphiStart,IMAEEphiEnd) ) continue ;
          //PG check whether the detid is buildable
          if (!EEDetId::validDetId (ix,iy,1))
            {
              std::cerr << "[WARN] elemento " << ix << " " << iy
                        << " 1" 
                        << " scartato" << std::endl ;
              continue ;
            }
          EEDetId det = EEDetId (ix,iy,1,EEDetId::XYMODE) ;
          std::cerr << "[INFO] writing " << ix << " " << iy << " 1" 
                    << " " << *(itMap->second.find (det.rawId ()))
                    << std::endl ;
          endcapWriter.writeLine (det,*(itMap->second.find (det.rawId ())));

        } //PG loop over x, loop over y
    }

  /* TODOS
   - inizio con EB
  
   - leggi l'intervallo di validita'
   - cerca i XML file in funzione di quello che sta scritto nel parser
     (qui bisogna conoscere gia' la cartella e costruire i nomi allo stesso
      modo, questo e' un punto debole)
   - apri un XML interface per ciascun file XML e leggine sono quello
     che serve
   - riversalo nel XML finale  
   - check the includes
  */
  
  

  
}
