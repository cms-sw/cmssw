#include "../interface/DBReaderWorkers.h"
#include "../interface/LogicIDTranslation.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/MonCrystalConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTTConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonMemChConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonMemTTConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserBlueCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserGreenCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserIRedCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserRedCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNPedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsOnlineDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTestPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPulseShapeDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNMGPADat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLed1Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonLed2Dat.h"
// #include "OnlineDB/EcalCondDB/interface/MonPNLed1Dat.h"
// #include "OnlineDB/EcalCondDB/interface/MonPNLed2Dat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLed1CrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLed2CrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonOccupancyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFormula.h"
#include "TString.h"

namespace ecaldqm
{
  typedef std::map<DetId, double> ReturnType;

  template<typename DataType>
  ReturnType
  fetchAndFill(std::map<std::string, double(*)(DataType const&)> const& _extractors, EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    typedef std::map<EcalLogicID, DataType> DataSet;
    typedef double (*DataExtractor)(DataType const&);
    typedef std::map<std::string, DataExtractor> ExtractorList;

    ExtractorList varMap;

    for(typename ExtractorList::const_iterator eItr(_extractors.begin()); eItr != _extractors.end(); ++eItr)
      if(_formula.find(eItr->first) != std::string::npos) varMap[eItr->first] = eItr->second;

    if(varMap.size() > 4) throw cms::Exception("EcalDQM") << _formula << " has too many variables";

    TString formula(_formula);
    
    unsigned iV(0);
    char varChars[4][2] = {"x", "y", "z", "t"};
    for(typename ExtractorList::iterator vItr(varMap.begin()); vItr != varMap.end(); ++vItr)
      formula.ReplaceAll(vItr->first, varChars[iV++]);

    TFormula tformula("formula", formula);
    if(tformula.Compile() != 0) throw cms::Exception("EcalDQM") << _formula << " could not be compiled";

    DataSet dataSet;
    _db->fetchDataSet(&dataSet, &_iov);

    ReturnType result;

    for(typename DataSet::const_iterator dItr(dataSet.begin()); dItr != dataSet.end(); ++dItr){
      double vars[4];
      iV = 0; 
      for(typename ExtractorList::iterator vItr(varMap.begin()); vItr != varMap.end(); ++vItr)
        vars[iV++] = vItr->second(dItr->second);

      result[toDetId(dItr->first)] = tformula.EvalPar(vars);
    }

    return result;
  }
 
  ReturnType
  CrystalConsistencyReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonCrystalConsistencyDat const&)> extList;

    extList["processed_events"] = [](MonCrystalConsistencyDat const& dat){ return double(dat.getProcessedEvents()); };
    extList["problematic_events"] = [](MonCrystalConsistencyDat const& dat){ return double(dat.getProblematicEvents()); };
    extList["problems_id"] = [](MonCrystalConsistencyDat const& dat){ return double(dat.getProblemsID()); };
    extList["problems_gain_zero"] = [](MonCrystalConsistencyDat const& dat){ return double(dat.getProblemsGainZero()); };
    extList["problems_gain_switch"] = [](MonCrystalConsistencyDat const& dat){ return double(dat.getProblemsGainSwitch()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TTConsistencyReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonTTConsistencyDat const&)> extList;

    extList["processed_events"] = [](MonTTConsistencyDat const& dat){ return double(dat.getProcessedEvents()); };
    extList["problematic_events"] = [](MonTTConsistencyDat const& dat){ return double(dat.getProblematicEvents()); };
    extList["problems_id"] = [](MonTTConsistencyDat const& dat){ return double(dat.getProblemsID()); };
    extList["problems_size"] = [](MonTTConsistencyDat const& dat){ return double(dat.getProblemsSize()); };
    extList["problems_LV1"] = [](MonTTConsistencyDat const& dat){ return double(dat.getProblemsLV1()); };
    extList["problems_bunch_X"] = [](MonTTConsistencyDat const& dat){ return double(dat.getProblemsBunchX()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  MemChConsistencyReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonMemChConsistencyDat const&)> extList;

    extList["processed_events"] = [](MonMemChConsistencyDat const& dat){ return double(dat.getProcessedEvents()); };
    extList["problematic_events"] = [](MonMemChConsistencyDat const& dat){ return double(dat.getProblematicEvents()); };
    extList["problems_id"] = [](MonMemChConsistencyDat const& dat){ return double(dat.getProblemsID()); };
    extList["problems_gain_zero"] = [](MonMemChConsistencyDat const& dat){ return double(dat.getProblemsGainZero()); };
    extList["problems_gain_switch"] = [](MonMemChConsistencyDat const& dat){ return double(dat.getProblemsGainSwitch()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  MemTTConsistencyReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonMemTTConsistencyDat const&)> extList;

    extList["processed_events"] = [](MonMemTTConsistencyDat const& dat){ return double(dat.getProcessedEvents()); };
    extList["problematic_events"] = [](MonMemTTConsistencyDat const& dat){ return double(dat.getProblematicEvents()); };
    extList["problems_id"] = [](MonMemTTConsistencyDat const& dat){ return double(dat.getProblemsID()); };
    extList["problems_size"] = [](MonMemTTConsistencyDat const& dat){ return double(dat.getProblemsSize()); };
    extList["problems_LV1"] = [](MonMemTTConsistencyDat const& dat){ return double(dat.getProblemsLV1()); };
    extList["problems_bunch_X"] = [](MonMemTTConsistencyDat const& dat){ return double(dat.getProblemsBunchX()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  LaserBlueReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonLaserBlueDat const&)> extList;

    extList["apd_mean"] = [](MonLaserBlueDat const& dat){ return double(dat.getAPDMean()); };
    extList["apd_rms"] = [](MonLaserBlueDat const& dat){ return double(dat.getAPDRMS()); };
    extList["apd_over_pn_mean"] = [](MonLaserBlueDat const& dat){ return double(dat.getAPDOverPNMean()); };
    extList["apd_over_pn_rms"] = [](MonLaserBlueDat const& dat){ return double(dat.getAPDOverPNRMS()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TimingLaserBlueCrystalReader::run(EcalCondDBInterface*, MonRunIOV&, std::string const&)
  {
    return ReturnType();
  }

  ReturnType
  PNBlueReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPNBlueDat const&)> extList;

    extList["adc_mean_g1"] = [](MonPNBlueDat const& dat){ return double(dat.getADCMeanG1()); };
    extList["adc_rms_g1"] = [](MonPNBlueDat const& dat){ return double(dat.getADCRMSG1()); };
    extList["ped_mean_g1"] = [](MonPNBlueDat const& dat){ return double(dat.getPedMeanG1()); };
    extList["ped_rms_g1"] = [](MonPNBlueDat const& dat){ return double(dat.getPedRMSG1()); };
    extList["adc_mean_g16"] = [](MonPNBlueDat const& dat){ return double(dat.getADCMeanG16()); };
    extList["adc_rms_g16"] = [](MonPNBlueDat const& dat){ return double(dat.getADCRMSG16()); };
    extList["ped_mean_g16"] = [](MonPNBlueDat const& dat){ return double(dat.getPedMeanG16()); };
    extList["ped_rms_g16"] = [](MonPNBlueDat const& dat){ return double(dat.getPedMeanG16()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  LaserGreenReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonLaserGreenDat const&)> extList;

    extList["apd_mean"] = [](MonLaserGreenDat const& dat){ return double(dat.getAPDMean()); };
    extList["apd_rms"] = [](MonLaserGreenDat const& dat){ return double(dat.getAPDRMS()); };
    extList["apd_over_pn_mean"] = [](MonLaserGreenDat const& dat){ return double(dat.getAPDOverPNMean()); };
    extList["apd_over_pn_rms"] = [](MonLaserGreenDat const& dat){ return double(dat.getAPDOverPNRMS()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TimingLaserGreenCrystalReader::run(EcalCondDBInterface*, MonRunIOV&, std::string const&)
  {
    return ReturnType();
  }

  ReturnType
  PNGreenReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPNGreenDat const&)> extList;

    extList["adc_mean_g1"] = [](MonPNGreenDat const& dat){ return double(dat.getADCMeanG1()); };
    extList["adc_rms_g1"] = [](MonPNGreenDat const& dat){ return double(dat.getADCRMSG1()); };
    extList["ped_mean_g1"] = [](MonPNGreenDat const& dat){ return double(dat.getPedMeanG1()); };
    extList["ped_rms_g1"] = [](MonPNGreenDat const& dat){ return double(dat.getPedRMSG1()); };
    extList["adc_mean_g16"] = [](MonPNGreenDat const& dat){ return double(dat.getADCMeanG16()); };
    extList["adc_rms_g16"] = [](MonPNGreenDat const& dat){ return double(dat.getADCRMSG16()); };
    extList["ped_mean_g16"] = [](MonPNGreenDat const& dat){ return double(dat.getPedMeanG16()); };
    extList["ped_rms_g16"] = [](MonPNGreenDat const& dat){ return double(dat.getPedMeanG16()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  LaserIRedReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonLaserIRedDat const&)> extList;

    extList["apd_mean"] = [](MonLaserIRedDat const& dat){ return double(dat.getAPDMean()); };
    extList["apd_rms"] = [](MonLaserIRedDat const& dat){ return double(dat.getAPDRMS()); };
    extList["apd_over_pn_mean"] = [](MonLaserIRedDat const& dat){ return double(dat.getAPDOverPNMean()); };
    extList["apd_over_pn_rms"] = [](MonLaserIRedDat const& dat){ return double(dat.getAPDOverPNRMS()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TimingLaserIRedCrystalReader::run(EcalCondDBInterface*, MonRunIOV&, std::string const&)
  {
    return ReturnType();
  }

  ReturnType
  PNIRedReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPNIRedDat const&)> extList;

    extList["adc_mean_g1"] = [](MonPNIRedDat const& dat){ return double(dat.getADCMeanG1()); };
    extList["adc_rms_g1"] = [](MonPNIRedDat const& dat){ return double(dat.getADCRMSG1()); };
    extList["ped_mean_g1"] = [](MonPNIRedDat const& dat){ return double(dat.getPedMeanG1()); };
    extList["ped_rms_g1"] = [](MonPNIRedDat const& dat){ return double(dat.getPedRMSG1()); };
    extList["adc_mean_g16"] = [](MonPNIRedDat const& dat){ return double(dat.getADCMeanG16()); };
    extList["adc_rms_g16"] = [](MonPNIRedDat const& dat){ return double(dat.getADCRMSG16()); };
    extList["ped_mean_g16"] = [](MonPNIRedDat const& dat){ return double(dat.getPedMeanG16()); };
    extList["ped_rms_g16"] = [](MonPNIRedDat const& dat){ return double(dat.getPedMeanG16()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  LaserRedReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonLaserRedDat const&)> extList;

    extList["apd_mean"] = [](MonLaserRedDat const& dat){ return double(dat.getAPDMean()); };
    extList["apd_rms"] = [](MonLaserRedDat const& dat){ return double(dat.getAPDRMS()); };
    extList["apd_over_pn_mean"] = [](MonLaserRedDat const& dat){ return double(dat.getAPDOverPNMean()); };
    extList["apd_over_pn_rms"] = [](MonLaserRedDat const& dat){ return double(dat.getAPDOverPNRMS()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TimingLaserRedCrystalReader::run(EcalCondDBInterface*, MonRunIOV&, std::string const&)
  {
    return ReturnType();
  }

  ReturnType
  PNRedReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPNRedDat const&)> extList;

    extList["adc_mean_g1"] = [](MonPNRedDat const& dat){ return double(dat.getADCMeanG1()); };
    extList["adc_rms_g1"] = [](MonPNRedDat const& dat){ return double(dat.getADCRMSG1()); };
    extList["ped_mean_g1"] = [](MonPNRedDat const& dat){ return double(dat.getPedMeanG1()); };
    extList["ped_rms_g1"] = [](MonPNRedDat const& dat){ return double(dat.getPedRMSG1()); };
    extList["adc_mean_g16"] = [](MonPNRedDat const& dat){ return double(dat.getADCMeanG16()); };
    extList["adc_rms_g16"] = [](MonPNRedDat const& dat){ return double(dat.getADCRMSG16()); };
    extList["ped_mean_g16"] = [](MonPNRedDat const& dat){ return double(dat.getPedMeanG16()); };
    extList["ped_rms_g16"] = [](MonPNRedDat const& dat){ return double(dat.getPedMeanG16()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  PedestalsReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPedestalsDat const&)> extList;

    extList["ped_mean_g1"] = [](MonPedestalsDat const& dat){ return double(dat.getPedMeanG1()); };
    extList["ped_rms_g1"] = [](MonPedestalsDat const& dat){ return double(dat.getPedRMSG1()); };
    extList["ped_mean_g6"] = [](MonPedestalsDat const& dat){ return double(dat.getPedMeanG6()); };
    extList["ped_rms_g6"] = [](MonPedestalsDat const& dat){ return double(dat.getPedRMSG6()); };
    extList["ped_mean_g12"] = [](MonPedestalsDat const& dat){ return double(dat.getPedMeanG12()); };
    extList["ped_rms_g12"] = [](MonPedestalsDat const& dat){ return double(dat.getPedRMSG12()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  PNPedReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPNPedDat const&)> extList;

    extList["ped_mean_g1"] = [](MonPNPedDat const& dat){ return double(dat.getPedMeanG1()); };
    extList["ped_rms_g1"] = [](MonPNPedDat const& dat){ return double(dat.getPedRMSG1()); };
    extList["ped_mean_g16"] = [](MonPNPedDat const& dat){ return double(dat.getPedMeanG16()); };
    extList["ped_rms_g16"] = [](MonPNPedDat const& dat){ return double(dat.getPedRMSG16()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  PedestalsOnlineReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPedestalsOnlineDat const&)> extList;

    extList["adc_mean_g12"] = [](MonPedestalsOnlineDat const& dat){ return double(dat.getADCMeanG12()); };
    extList["adc_rms_g12"] = [](MonPedestalsOnlineDat const& dat){ return double(dat.getADCRMSG12()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TestPulseReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonTestPulseDat const&)> extList;

    extList["adc_mean_g1"] = [](MonTestPulseDat const& dat){ return double(dat.getADCMeanG1()); };
    extList["adc_rms_g1"] = [](MonTestPulseDat const& dat){ return double(dat.getADCRMSG1()); };
    extList["adc_mean_g6"] = [](MonTestPulseDat const& dat){ return double(dat.getADCMeanG6()); };
    extList["adc_rms_g6"] = [](MonTestPulseDat const& dat){ return double(dat.getADCRMSG6()); };
    extList["adc_mean_g12"] = [](MonTestPulseDat const& dat){ return double(dat.getADCMeanG12()); };
    extList["adc_rms_g12"] = [](MonTestPulseDat const& dat){ return double(dat.getADCRMSG12()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  PulseShapeReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPulseShapeDat const&)> extList;

    extList["g1_avg_sample_01"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[0]); };
    extList["g1_avg_sample_02"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[1]); };
    extList["g1_avg_sample_03"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[2]); };
    extList["g1_avg_sample_04"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[3]); };
    extList["g1_avg_sample_05"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[4]); };
    extList["g1_avg_sample_06"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[5]); };
    extList["g1_avg_sample_07"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[6]); };
    extList["g1_avg_sample_08"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[7]); };
    extList["g1_avg_sample_09"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[8]); };
    extList["g1_avg_sample_10"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(1)[9]); };
    extList["g6_avg_sample_01"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[0]); };
    extList["g6_avg_sample_02"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[1]); };
    extList["g6_avg_sample_03"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[2]); };
    extList["g6_avg_sample_04"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[3]); };
    extList["g6_avg_sample_05"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[4]); };
    extList["g6_avg_sample_06"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[5]); };
    extList["g6_avg_sample_07"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[6]); };
    extList["g6_avg_sample_08"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[7]); };
    extList["g6_avg_sample_09"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[8]); };
    extList["g6_avg_sample_10"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(6)[9]); };
    extList["g12_avg_sample_01"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[0]); };
    extList["g12_avg_sample_02"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[1]); };
    extList["g12_avg_sample_03"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[2]); };
    extList["g12_avg_sample_04"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[3]); };
    extList["g12_avg_sample_05"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[4]); };
    extList["g12_avg_sample_06"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[5]); };
    extList["g12_avg_sample_07"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[6]); };
    extList["g12_avg_sample_08"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[7]); };
    extList["g12_avg_sample_09"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[8]); };
    extList["g12_avg_sample_10"] = [](MonPulseShapeDat const& dat){ return double(dat.getSamples(12)[9]); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  PNMGPAReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonPNMGPADat const&)> extList;

    extList["adc_mean_g1"] = [](MonPNMGPADat const& dat){ return double(dat.getADCMeanG1()); };
    extList["adc_rms_g1"] = [](MonPNMGPADat const& dat){ return double(dat.getADCRMSG1()); };
    extList["ped_mean_g1"] = [](MonPNMGPADat const& dat){ return double(dat.getPedMeanG1()); };
    extList["ped_rms_g1"] = [](MonPNMGPADat const& dat){ return double(dat.getPedRMSG1()); };
    extList["adc_mean_g16"] = [](MonPNMGPADat const& dat){ return double(dat.getADCMeanG16()); };
    extList["adc_rms_g16"] = [](MonPNMGPADat const& dat){ return double(dat.getADCRMSG16()); };
    extList["ped_mean_g16"] = [](MonPNMGPADat const& dat){ return double(dat.getPedMeanG16()); };
    extList["ped_rms_g16"] = [](MonPNMGPADat const& dat){ return double(dat.getPedRMSG16()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TimingCrystalReader::run(EcalCondDBInterface*, MonRunIOV&, std::string const&)
  {
    return ReturnType();
  }

  ReturnType
  Led1Reader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonLed1Dat const&)> extList;

    extList["vpt_mean"] = [](MonLed1Dat const& dat){ return double(dat.getVPTMean()); };
    extList["vpt_rms"] = [](MonLed1Dat const& dat){ return double(dat.getVPTRMS()); };
    extList["vpt_over_pn_mean"] = [](MonLed1Dat const& dat){ return double(dat.getVPTOverPNMean()); };
    extList["vpt_over_pn_rms"] = [](MonLed1Dat const& dat){ return double(dat.getVPTOverPNRMS()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TimingLed1CrystalReader::run(EcalCondDBInterface*, MonRunIOV&, std::string const&)
  {
    return ReturnType();
  }

  ReturnType
  Led2Reader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonLed2Dat const&)> extList;

    extList["vpt_mean"] = [](MonLed2Dat const& dat){ return double(dat.getVPTMean()); };
    extList["vpt_rms"] = [](MonLed2Dat const& dat){ return double(dat.getVPTRMS()); };
    extList["vpt_over_pn_mean"] = [](MonLed2Dat const& dat){ return double(dat.getVPTOverPNMean()); };
    extList["vpt_over_pn_rms"] = [](MonLed2Dat const& dat){ return double(dat.getVPTOverPNRMS()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }

  ReturnType
  TimingLed2CrystalReader::run(EcalCondDBInterface*, MonRunIOV&, std::string const&)
  {
    return ReturnType();
  }

  ReturnType
  OccupancyReader::run(EcalCondDBInterface* _db, MonRunIOV& _iov, std::string const& _formula)
  {
    std::map<std::string, double(*)(MonOccupancyDat const&)> extList;

    extList["events_over_low_threshold"] = [](MonOccupancyDat const& dat){ return double(dat.getEventsOverLowThreshold()); };
    extList["events_over_high_threshold"] = [](MonOccupancyDat const& dat){ return double(dat.getEventsOverHighThreshold()); };
    extList["avg_energy"] = [](MonOccupancyDat const& dat){ return double(dat.getAvgEnergy()); };

    return fetchAndFill(extList, _db, _iov, _formula);
  }
}
