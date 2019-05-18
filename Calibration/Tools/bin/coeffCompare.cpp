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
//#include "Calibration/Tools/bin/trivialParser.h"

#include "TH2.h"
#include "TH1.h"
#include "TFile.h"

#define PI_GRECO 3.14159265

inline int etaShifter(const int etaOld) {
  if (etaOld < 0)
    return etaOld + 85;
  else if (etaOld > 0)
    return etaOld + 84;
  assert(false && "eta cannot be 0");
}

// ------------------------------------------------------------------------

// MF questo file prende due set di coefficienti e li confronta
//

//-------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  //  std::cout << "parsing cfg file: " << argv[1] << std::endl ;
  //  trivialParser configParams (static_cast<std::string> (argv[1])) ;
  //
  //  int EBetaStart = static_cast<int> (configParams.getVal ("EBetaStart")) ;
  //  int EBetaEnd = static_cast<int> (configParams.getVal ("EBetaEnd")) ;
  //  int EBphiStart = static_cast<int> (configParams.getVal ("EBphiStart")) ;
  //  int EBphiEnd = static_cast<int> (configParams.getVal ("EBphiEnd")) ;
  //  int EEradStart = static_cast<int> (configParams.getVal ("EEradStart")) ;
  //  int EEradEnd = static_cast<int> (configParams.getVal ("EEradEnd")) ;
  //  int EEphiStart = static_cast<int> (configParams.getVal ("EEphiStart")) ;
  //  int EEphiEnd = static_cast<int> (configParams.getVal ("EEphiEnd")) ;

  //  int EBetaStart = -85 ;
  //  int EBetaEnd = 86 ;
  //  int EBphiStart = 1 ;
  //  int EBphiEnd = 361 ;
  //  int EEradStart = 0 ;
  //  int EEradEnd = 50 ;
  //  int EEphiStart = 1 ;
  //  int EEphiEnd = 361 ;

  int EBetaStart = 0;
  int EBetaEnd = 86;
  int EBphiStart = 20;
  int EBphiEnd = 60;
  int EEradStart = 15;
  int EEradEnd = 50;
  int EEphiStart = 20;
  int EEphiEnd = 60;

  //  std::string barrelfile ="miscalib_barrel_0.05.xml" ;
  //  std::string endcapfile ="miscalib_endcap_0.05.xml" ;
  //  std::string calibBarrelfile ="recalib_calibBarrel_0.05.xml" ;
  //  std::string calibEndcapfile ="recalib_calibEndcap_0.05.xml" ;

  std::string barrelfile =
      "/afs/cern.ch/user/g/govoni/scratch1/CMSSW/CALIB/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/"
      "ecal_barrel_startup.xml";
  std::string endcapfile =
      "/afs/cern.ch/user/g/govoni/scratch1/CMSSW/CALIB/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/"
      "ecal_endcap_startup.xml";
  std::string calibBarrelfile =
      "/afs/cern.ch/user/g/govoni/scratch1/CMSSW/CALIB/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/"
      "inv_ecal_barrel_startup.xml";
  std::string calibEndcapfile =
      "/afs/cern.ch/user/g/govoni/scratch1/CMSSW/CALIB/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/"
      "inv_ecal_endcap_startup.xml";

  //PG get the miscalibration files for EB and EE
  //PG ------------------------------------------

  CaloMiscalibMapEcal EBscalibMap;
  EBscalibMap.prefillMap();
  MiscalibReaderFromXMLEcalBarrel barrelreader(EBscalibMap);
  if (!barrelfile.empty())
    barrelreader.parseXMLMiscalibFile(barrelfile);
  EcalIntercalibConstants* EBconstants = new EcalIntercalibConstants(EBscalibMap.get());
  const EcalIntercalibConstantMap& iEBscalibMap = EBconstants->getMap();  //MF prende i vecchi coeff

  CaloMiscalibMapEcal EEscalibMap;
  EEscalibMap.prefillMap();
  MiscalibReaderFromXMLEcalEndcap endcapreader(EEscalibMap);
  if (!endcapfile.empty())
    endcapreader.parseXMLMiscalibFile(endcapfile);
  EcalIntercalibConstants* EEconstants = new EcalIntercalibConstants(EEscalibMap.get());
  const EcalIntercalibConstantMap& iEEscalibMap = EEconstants->getMap();  //MF prende i vecchi coeff

  //PG get the recalibration files for EB and EE
  //PG -----------------------------------------

  CaloMiscalibMapEcal EBcalibMap;
  EBcalibMap.prefillMap();
  MiscalibReaderFromXMLEcalBarrel calibBarrelreader(EBcalibMap);
  if (!calibBarrelfile.empty())
    calibBarrelreader.parseXMLMiscalibFile(calibBarrelfile);
  EcalIntercalibConstants* EBCconstants = new EcalIntercalibConstants(EBcalibMap.get());
  const EcalIntercalibConstantMap& iEBcalibMap = EBCconstants->getMap();  //MF prende i vecchi coeff

  CaloMiscalibMapEcal EEcalibMap;
  EEcalibMap.prefillMap();
  MiscalibReaderFromXMLEcalEndcap calibEndcapreader(EEcalibMap);
  if (!calibEndcapfile.empty())
    calibEndcapreader.parseXMLMiscalibFile(calibEndcapfile);
  EcalIntercalibConstants* EECconstants = new EcalIntercalibConstants(EEcalibMap.get());
  const EcalIntercalibConstantMap& iEEcalibMap = EECconstants->getMap();  //MF prende i vecchi coeff

  //PG fill the histograms
  //PG -------------------

  // ECAL barrel
  TH1F EBCompareCoeffDistr("EBCompareCoeff", "EBCompareCoeff", 300, 0, 2);
  TH2F EBCompareCoeffMap("EBCompareCoeffMap", "EBCompareCoeffMap", 171, -85, 86, 360, 1, 361);

  // ECAL barrel
  for (int ieta = EBetaStart; ieta < EBetaEnd; ++ieta)
    for (int iphi = EBphiStart; iphi <= EBphiEnd; ++iphi) {
      if (!EBDetId::validDetId(ieta, iphi))
        continue;
      EBDetId det = EBDetId(ieta, iphi, EBDetId::ETAPHIMODE);
      double factor = *(iEBcalibMap.find(det.rawId())) * *(iEBscalibMap.find(det.rawId()));
      EBCompareCoeffDistr.Fill(factor);
      EBCompareCoeffMap.Fill(ieta, iphi, factor);
    }  // ECAL barrel

  TH1F EEPCompareCoeffDistr("EEPCompareCoeffDistr", "EEPCompareCoeffDistr", 200, 0, 2);
  TH2F EEPCompareCoeffMap("EEPCompareCoeffMap", "EEPCompareCoeffMap", 101, 0, 101, 101, 0, 101);

  // ECAL endcap +
  for (int ix = 1; ix <= 100; ++ix)
    for (int iy = 1; iy <= 100; ++iy) {
      int rad = static_cast<int>(sqrt((ix - 50) * (ix - 50) + (iy - 50) * (iy - 50)));
      if (rad < EEradStart || rad > EEradEnd)
        continue;
      double phiTemp = atan2(iy - 50, ix - 50);
      if (phiTemp < 0)
        phiTemp += 2 * PI_GRECO;
      int phi = static_cast<int>(phiTemp * 180 / PI_GRECO);
      if (phi < EEphiStart || phi > EEphiEnd)
        continue;
      if (!EEDetId::validDetId(ix, iy, 1))
        continue;
      EEDetId det = EEDetId(ix, iy, 1, EEDetId::XYMODE);
      double factor = *(iEEcalibMap.find(det.rawId())) * *(iEEscalibMap.find(det.rawId()));
      EEPCompareCoeffDistr.Fill(factor);
      EEPCompareCoeffMap.Fill(ix, iy, factor);
    }  // ECAL endcap +

  // ECAL endcap-
  TH1F EEMCompareCoeffDistr("EEMCompareCoeffDistr", "EEMCompareCoeffDistr", 200, 0, 2);
  TH2F EEMCompareCoeffMap("EEMCompareCoeffMap", "EEMCompareCoeffMap", 100, 0, 100, 100, 0, 100);

  // ECAL endcap -
  for (int ix = 1; ix <= 100; ++ix)
    for (int iy = 1; iy <= 100; ++iy) {
      int rad = static_cast<int>(sqrt((ix - 50) * (ix - 50) + (iy - 50) * (iy - 50)));
      if (rad < EEradStart || rad > EEradEnd)
        continue;
      double phiTemp = atan2(iy - 50, ix - 50);
      if (phiTemp < 0)
        phiTemp += 2 * PI_GRECO;
      if (!EEDetId::validDetId(ix, iy, -1))
        continue;
      EEDetId det = EEDetId(ix, iy, -1, EEDetId::XYMODE);
      double factor = *(iEEcalibMap.find(det.rawId())) * *(iEEscalibMap.find(det.rawId()));
      EEMCompareCoeffDistr.Fill(factor);
      EEMCompareCoeffMap.Fill(ix, iy, factor);
    }  // ECAL endcap -

  std::string filename = "coeffcompare.root";
  TFile out(filename.c_str(), "recreate");
  EEMCompareCoeffMap.Write();
  EEMCompareCoeffDistr.Write();
  EEPCompareCoeffMap.Write();
  EEPCompareCoeffDistr.Write();
  EBCompareCoeffMap.Write();
  EBCompareCoeffDistr.Write();
  out.Close();
}
