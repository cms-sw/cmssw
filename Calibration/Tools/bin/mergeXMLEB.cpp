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

inline int etaShifter(const int etaOld) {
  if (etaOld < 0)
    return etaOld + 85;
  else if (etaOld > 0)
    return etaOld + 84;
  assert(false);
}

// ------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  std::cout << "parsing cfg file: " << argv[1] << std::endl;
  trivialParser configParams(static_cast<std::string>(argv[1]));

  int IMAEBetaStart = static_cast<int>(configParams.getVal("IMAEBetaStart"));
  int IMAEBetaEnd = static_cast<int>(configParams.getVal("IMAEBetaEnd"));
  int IMAEBetaWidth = static_cast<int>(configParams.getVal("IMAEBetaWidth"));
  int IMAEBphiStart = static_cast<int>(configParams.getVal("IMAEBphiStart"));
  int IMAEBphiEnd = static_cast<int>(configParams.getVal("IMAEBphiEnd"));

  //  std::cerr << "[PG] IMAEBetaStart = " << IMAEBetaStart << std::endl ;
  //  std::cerr << "[PG] IMAEBetaEnd = " << IMAEBetaEnd << std::endl ;
  //  std::cerr << "[PG] IMAEBetaWidth = " << IMAEBetaWidth << std::endl ;
  //  std::cerr << "[PG] IMAEBphiStart = " << IMAEBphiStart << std::endl ;
  //  std::cerr << "[PG] IMAEBphiEnd = " << IMAEBphiEnd << std::endl ;
  //  std::cerr << "[PG] IMAEBphiWidth = " << IMAEBphiWidth << std::endl ;

  std::string coeffFolder = argv[2];

  std::map<int, EcalIntercalibConstantMap> recalibrators;
  //PG FIXME  c'e' il copy ctor della CaloMiscalibMapEcal?

  int alreadySwitched = 0;
  //PG loop on EB eta indexes
  for (int etaIndex = IMAEBetaStart; etaIndex < IMAEBetaEnd; etaIndex += IMAEBetaWidth) {
    int currentIndex = etaIndex;
    int nextIndex = etaIndex + IMAEBetaWidth;

    //PG to cope with the missing 0
    if ((alreadySwitched == 0) && (nextIndex * IMAEBetaStart <= 0)) {
      nextIndex += 1;
      etaIndex += 1;  //PG this is for the next loop
      alreadySwitched = 1;
    }

    //PG compute the values of the limits
    //PG FIXME questo forse non e' sufficiente, bisogna anche considerare il caso
    //PG FIXME in cui currentIndex e' positivo e effEtaIndex start no
    //PG FIXME lo stesso vale per l'end e il nexindex
    //PG FIXME lo stesso vale per gli script perl
    //      int effEtaIndexStart = currentIndex - IMAEBetaWidth ;
    //      if (effEtaIndexStart < -85) { effEtaIndexStart = -85 ; }
    //      int effEtaIndexEnd = nextIndex + IMAEBetaWidth ;
    //      if (effEtaIndexEnd > 85) { effEtaIndexEnd = 85 ; }

    //PG build the filename
    std::stringstream nomeFile;
    nomeFile << coeffFolder << "/EBcalibCoeff_" << currentIndex << "-" << nextIndex << ".xml";
    std::string fileName = nomeFile.str();
    //      std::cerr << "PG nomefile: " << fileName << std::endl ;

    //PG open the XML file
    CaloMiscalibMapEcal map;
    map.prefillMap();
    MiscalibReaderFromXMLEcalBarrel barrelreader(map);
    if (!fileName.empty())
      barrelreader.parseXMLMiscalibFile(fileName);
    EcalIntercalibConstants* constants = new EcalIntercalibConstants(map.get());
    recalibrators[currentIndex] = constants->getMap();
  }  //PG loop on EB eta indexes

  //PG prepare the XML to be saved

  //PG this command outputs an XML file with a fixed name
  calibXMLwriter barrelWriter(EcalBarrel);
  //PG loop on EB eta slices
  for (std::map<int, EcalIntercalibConstantMap>::const_iterator itMap = recalibrators.begin();
       itMap != recalibrators.end();
       ++itMap) {
    //PG compute the values of the limits
    //PG FIXME questo forse non e' sufficiente, bisogna anche considerare il caso
    //PG FIXME in cui currentIndex e' positivo e effEtaIndex start no
    //PG FIXME lo stesso vale per l'end e il nexindex
    //PG FIXME lo stesso vale per gli script perl

    //PG non so scegliere da dove partire, questo e' un disastro
    //PG il problema e' dei bordi
    //PG forse la cosa migliore e' salvare i xml file con un nome che dica dei coeff buoni,
    //PG non di quelli calcolati - direi che e' la cosa giusta

    //PG loop over eta
    int etaStop = itMap->first + IMAEBetaWidth;
    if (etaStop * itMap->first <= 0)
      ++etaStop;
    for (int ieta = itMap->first; ieta < etaStop; ++ieta)
      //PG loop over phi
      for (int iphi = IMAEBphiStart; iphi < IMAEBphiEnd; ++iphi) {
        if (!EBDetId::validDetId(ieta, iphi)) {
          std::cerr << "[WARN] elemento " << ieta << " " << iphi << " scartato" << std::endl;
          continue;
        }
        EBDetId det = EBDetId(ieta, iphi, EBDetId::ETAPHIMODE);
        std::cerr << "[INFO] writing " << ieta << " " << iphi << " " << (*itMap->second.find(det.rawId())) << std::endl;
        barrelWriter.writeLine(det, *(itMap->second.find(det.rawId())));
      }  //PG loop over phi, loop over eta
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
