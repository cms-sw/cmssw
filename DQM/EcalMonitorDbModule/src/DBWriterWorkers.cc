#include "../interface/DBWriterWorkers.h"
#include "../interface/LogicIDTranslation.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"
#include "DQM/EcalCommon/interface/MESetUtils.h"

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
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace ecaldqm {
  enum Quality {
    kBad = 0,
    kGood = 1,
    kUnknown = 2,
    kMBad = 3,
    kMGood = 4,
    kMUnknown = 5
  };

  bool
  qualityOK(int _quality)
  {
    return (_quality != kBad && _quality != kUnknown);
  }

  DBWriterWorker::DBWriterWorker(std::string const& _name, edm::ParameterSet const& _ps) :
    name_(_name),
    runTypes_(),
    source_(),
    active_(false)
  {
    edm::ParameterSet const& params(_ps.getUntrackedParameterSet(name_));

    std::vector<std::string> runTypes(params.getUntrackedParameter<std::vector<std::string> >("runTypes"));
    for(unsigned iT(0); iT < runTypes.size(); ++iT)
      runTypes_.insert(runTypes[iT]);

    if(!params.existsAs<edm::ParameterSet>("source", false)) return;

    edm::ParameterSet const& sourceParams(params.getUntrackedParameterSet("source"));
    std::vector<std::string> const& meNames(sourceParams.getParameterNames());
    for(unsigned iP(0); iP < meNames.size(); ++iP){
      std::string meName(meNames[iP]);
      edm::ParameterSet const& meParam(sourceParams.getUntrackedParameterSet(meName));
      source_.insert(meName, createMESet(meParam));
    }
  }

  void
  DBWriterWorker::retrieveSource(DQMStore::IGetter& _igetter)
  {
    std::string failedPath;
    for(MESetCollection::iterator sItr(this->source_.begin()); sItr != this->source_.end(); ++sItr){
      if(!sItr->second->retrieve(_igetter, &failedPath)){
        edm::LogError("EcalDQM") << name_ << ": MESet " << sItr->first << "@" << failedPath << " not found";
        this->active_ = false;
        return;
      }
    }
  }
 
  bool
  IntegrityWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    /*
      uses
      OccupancyTask.Digi (h_)
      PNDiodeTask.Occupancy (hmem_)
      IntegrityTask.Gain (h01_)
      IntegrityTask.ChId (h02_)
      IntegrityTask.GainSwitch (h03_)
      IntegrityTask.TowerId (h04_)
      IntegrityTask.BlockSize (h05_)
      RawDataTask.L1AFE
      RawDataTask.BXFE
      PNDiodeTask.MemChId (h06_)
      PNDiodeTask.MemGain (h07_)
      PNDiodeTask.MemTowerId (h08_)
      PNDiodeTask.MomBlockSize (h09_)
      IntegrityClient.Quality
      PNIntegrityClient.QualitySummary
    */

    bool result(true);

    std::map<EcalLogicID, MonCrystalConsistencyDat> crystalConsistencies;
    std::map<EcalLogicID, MonTTConsistencyDat> towerConsistencies;
    std::map<EcalLogicID, MonMemChConsistencyDat> memChannelConsistencies;
    std::map<EcalLogicID, MonMemTTConsistencyDat> memTowerConsistencies;

    MESet const& digiME(source_.at("Digi"));
    MESet const& gainME(source_.at("Gain"));
    MESet const& chidME(source_.at("ChId"));
    MESet const& gainswitchME(source_.at("GainSwitch"));
    MESet const& qualityME(source_.at("Quality"));

    MESet const& toweridME(source_.at("TowerId"));
    MESet const& blocksizeME(source_.at("BlockSize"));
    MESet const& l1aME(source_.at("L1AFE"));
    MESet const& bxME(source_.at("BXFE"));

    MESet const& memdigiME(source_.at("MEMDigi"));
    MESet const& memchidME(source_.at("MEMChId"));
    MESet const& memgainME(source_.at("MEMGain"));
    MESet const& pnqualityME(source_.at("PNQuality"));

    MESet const& memtoweridME(source_.at("MEMTowerId"));
    MESet const& memblocksizeME(source_.at("MEMBlockSize"));

    if(verbosity_ > 1) edm::LogInfo("EcalDQM") << " Looping over crystals";

    MESet::const_iterator dEnd(digiME.end());
    MESet::const_iterator qItr(qualityME);
    for(MESet::const_iterator dItr(digiME.beginChannel()); dItr != dEnd; dItr.toNextChannel()){
      DetId id(dItr->getId());

      int nDigis(dItr->getBinContent());
      int gain(gainME.getBinContent(id));
      int chid(chidME.getBinContent(id));
      int gainswitch(gainswitchME.getBinContent(id));
      qItr = dItr;

      if(gain > 0 || chid > 0 || gainswitch > 0){
        MonCrystalConsistencyDat& data(crystalConsistencies[crystalID(id)]);
        data.setProcessedEvents(nDigis);
        data.setProblematicEvents(gain + chid + gainswitch);
        data.setProblemsGainZero(gain);
        data.setProblemsID(chid);
        data.setProblemsGainSwitch(gainswitch);

        int channelStatus(qItr->getBinContent());
        bool channelBad(channelStatus == kBad || channelStatus == kMBad);
        data.setTaskStatus(channelBad);

        result &= qualityOK(channelStatus);
      }
    }

    if(verbosity_ > 1) edm::LogInfo("EcalDQM") << " Looping over towers";

    for(unsigned iDCC(kEEmLow); iDCC <= kEBpHigh; ++iDCC){
      for(unsigned iTower(1); iTower <= 68; ++iTower){
        if(!ccuExists(iDCC + 1, iTower)) continue;

        EcalElectronicsId eid(iDCC + 1, iTower, 1, 1);
        std::vector<DetId> channels(getElectronicsMap()->dccTowerConstituents(iDCC + 1, iTower));
        int nDigis(0);
        bool towerBad(false);
        for(unsigned iD(0); iD < channels.size(); ++iD){
          int n(digiME.getBinContent(channels[iD]));
          if(n > nDigis) nDigis = n;
          int channelStatus(qualityME.getBinContent(channels[iD]));
          if(channelStatus == kBad || channelStatus == kMBad) towerBad = true;
        }

        int towerid(toweridME.getBinContent(eid));
        int blocksize(blocksizeME.getBinContent(eid));
        int l1a(l1aME.getBinContent(iDCC + 1, iTower));
        int bx(bxME.getBinContent(iDCC + 1, iTower));

        if(towerid > 0 || blocksize > 0 || l1a > 0 || bx > 0){
          MonTTConsistencyDat& data(towerConsistencies[towerID(eid)]);
          data.setProcessedEvents(nDigis);
          data.setProblematicEvents(towerid + blocksize + l1a + bx);
          data.setProblemsID(towerid);
          data.setProblemsSize(blocksize);
          data.setProblemsLV1(l1a);
          data.setProblemsBunchX(bx);
          data.setTaskStatus(towerBad);

          result &= !towerBad;
        }
      }
    }

    if(verbosity_ > 1) edm::LogInfo("EcalDQM") << " Looping over MEM channels and towers";

    for(unsigned iMD(0); iMD < memDCC.size(); ++iMD){
      unsigned iDCC(memDCC[iMD]);

      int subdet(iDCC <= kEEmHigh || iDCC >= kEEpLow ? EcalEndcap : EcalBarrel);

      for(unsigned iPN(1); iPN <= 10; ++iPN){
        EcalPnDiodeDetId pnid(subdet, iDCC + 1, iPN);

        int nDigis(memdigiME.getBinContent(pnid));
        int memchid(memchidME.getBinContent(pnid));
        int memgain(memgainME.getBinContent(pnid));

        if(memchid > 0 || memgain > 0){
          MonMemChConsistencyDat& data(memChannelConsistencies[memChannelID(pnid)]);

          data.setProcessedEvents(nDigis);
          data.setProblematicEvents(memchid + memgain);
          data.setProblemsID(memchid);
          data.setProblemsGainZero(memgain);

          int channelStatus(pnqualityME.getBinContent(pnid));
          bool channelBad(channelStatus == kBad || channelStatus == kMBad);
          data.setTaskStatus(channelBad);

          result &= qualityOK(channelStatus);
        }
      }

      for(unsigned iTower(69); iTower <= 70; ++iTower){
        EcalElectronicsId eid(iDCC + 1, iTower, 1, 1);

        int nDigis(0);
        bool towerBad(false);
        for(unsigned iPN(1); iPN <= 10; ++iPN){
          EcalPnDiodeDetId pnid(subdet, iDCC + 1, iPN);
          int n(memdigiME.getBinContent(pnid));
          if(n > nDigis) nDigis = n;
          int channelStatus(pnqualityME.getBinContent(pnid));
          if(channelStatus == kBad || channelStatus == kMBad) towerBad = true;
        }

        int towerid(memtoweridME.getBinContent(eid));
        int blocksize(memblocksizeME.getBinContent(eid));

        if(towerid > 0 || blocksize > 0){
          MonMemTTConsistencyDat& data(memTowerConsistencies[memTowerID(eid)]);

          data.setProcessedEvents(nDigis);
          data.setProblematicEvents(towerid + blocksize);
          data.setProblemsID(towerid);
          data.setProblemsSize(blocksize);
          data.setTaskStatus(towerBad);

          result &= !towerBad;
        }
      }
    }

    if(verbosity_ > 1) edm::LogInfo("EcalDQM") << " Inserting data";

    try{
      if(!crystalConsistencies.empty()){
        if(verbosity_ > 2) edm::LogInfo("EcalDQM") << " crystalConsistencies";
        _db->insertDataArraySet(&crystalConsistencies, &_iov);
      }
      if(!towerConsistencies.empty()){
        if(verbosity_ > 2) edm::LogInfo("EcalDQM") << " towerConsistencies";
        _db->insertDataArraySet(&towerConsistencies, &_iov);
      }
      if(!memChannelConsistencies.empty()){
        if(verbosity_ > 2) edm::LogInfo("EcalDQM") << " memChannelConsistencies";
        _db->insertDataArraySet(&memChannelConsistencies, &_iov);
      }
      if(!memTowerConsistencies.empty()){
        if(verbosity_ > 2) edm::LogInfo("EcalDQM") << " memTowerConsistencies";
        _db->insertDataArraySet(&memTowerConsistencies, &_iov);
      }
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return result;
  }

  LaserWriter::LaserWriter(edm::ParameterSet const& _ps) :
    DBWriterWorker("Laser", _ps),
    wlToME_()
  {
    std::vector<int> laserWavelengths(_ps.getUntrackedParameter<std::vector<int> >("laserWavelengths"));

    // wavelengths are not necessarily ordered
    // create a map wl -> MESet index
    // using Amplitude here but any multi-wavelength plot is fine

    MESet::PathReplacements repl;

    MESetMulti const& amplitude(static_cast<MESetMulti const&>(source_.at("Amplitude")));
    unsigned nWL(laserWavelengths.size());
    for(unsigned iWL(0); iWL != nWL; ++iWL){
      int wl(laserWavelengths[iWL]);
      if(wl <= 0 || wl >= 5) throw cms::Exception("InvalidConfiguration") << "Laser Wavelength";
      repl["wl"] = std::to_string(wl);
      wlToME_[wl] = amplitude.getIndex(repl);
    }
  }

  bool
  LaserWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    /*
      uses
      LaserTask.Amplitude (h01, h03, h05, h07)
      LaserTask.AOverP (h02, h04, h06, h08)
      LaserTask.Timing (h09, h10, h11, h12)
      LaserClient.Quality (meg01, meg02, meg03, meg04)
      LaserTask.PNAmplitude (i09, i10, i11, i12)
      LaserClient.PNQualitySummary (meg09, meg10, meg11, meg12)
      PNDiodeTask.Pedestal (i13, i14, i15, i16)
    */

    bool result(true);

    std::map<EcalLogicID, MonLaserBlueDat> l1Amp;
    std::map<EcalLogicID, MonTimingLaserBlueCrystalDat> l1Time;
    std::map<EcalLogicID, MonPNBlueDat> l1PN;
    std::map<EcalLogicID, MonLaserGreenDat> l2Amp;
    std::map<EcalLogicID, MonTimingLaserGreenCrystalDat> l2Time;
    std::map<EcalLogicID, MonPNGreenDat> l2PN;
    std::map<EcalLogicID, MonLaserIRedDat> l3Amp;
    std::map<EcalLogicID, MonTimingLaserIRedCrystalDat> l3Time;
    std::map<EcalLogicID, MonPNIRedDat> l3PN;
    std::map<EcalLogicID, MonLaserRedDat> l4Amp;
    std::map<EcalLogicID, MonTimingLaserRedCrystalDat> l4Time;
    std::map<EcalLogicID, MonPNRedDat> l4PN;

    MESet const& ampME(source_.at("Amplitude"));
    MESet const& aopME(source_.at("AOverP"));
    MESet const& timeME(source_.at("Timing"));
    MESet const& qualityME(source_.at("Quality"));

    MESet const& pnME(source_.at("PNAmplitude"));
    MESet const& pnQualityME(source_.at("PNQuality"));
    MESet const& pnPedestalME(source_.at("PNPedestal"));

    for(std::map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
      int wl(wlItr->first);
      unsigned iM(wlItr->second);

      static_cast<MESetMulti const&>(ampME).use(iM);
      static_cast<MESetMulti const&>(aopME).use(iM);
      static_cast<MESetMulti const&>(timeME).use(iM);
      static_cast<MESetMulti const&>(qualityME).use(iM);
      static_cast<MESetMulti const&>(pnME).use(iM);
      static_cast<MESetMulti const&>(pnQualityME).use(iM);

      MESet::const_iterator aEnd(ampME.end());
      MESet::const_iterator qItr(qualityME);
      MESet::const_iterator oItr(aopME);
      MESet::const_iterator tItr(timeME);
      for(MESet::const_iterator aItr(ampME.beginChannel()); aItr != aEnd; aItr.toNextChannel()){
        float aEntries(aItr->getBinEntries());
        if(aEntries < 1.) continue;

        qItr = aItr;
        oItr = aItr;
        tItr = aItr;

        DetId id(aItr->getId());

        float ampMean(aItr->getBinContent());
        float ampRms(aItr->getBinError() * std::sqrt(aEntries));

        float aopEntries(oItr->getBinEntries());
        float aopMean(oItr->getBinContent());
        float aopRms(oItr->getBinError() * std::sqrt(aopEntries));

        float timeEntries(tItr->getBinEntries());
        float timeMean(tItr->getBinContent());
        float timeRms(tItr->getBinError() * std::sqrt(timeEntries));

        int channelStatus(qItr->getBinContent());
        bool channelBad(channelStatus == kBad || channelStatus == kMBad);

        EcalLogicID logicID(crystalID(id));

        switch(wl){
        case 1:
          {
            MonLaserBlueDat& aData(l1Amp[logicID]);
            aData.setAPDMean(ampMean);
            aData.setAPDRMS(ampRms);
            aData.setAPDOverPNMean(aopMean);
            aData.setAPDOverPNRMS(aopRms);
            aData.setTaskStatus(channelBad);
 
            MonTimingLaserBlueCrystalDat& tData(l1Time[logicID]);
            tData.setTimingMean(timeMean);
            tData.setTimingRMS(timeRms);
            tData.setTaskStatus(channelBad);
          }
          break;
        case 2:
          {
            MonLaserGreenDat& aData(l2Amp[logicID]);
            aData.setAPDMean(ampMean);
            aData.setAPDRMS(ampRms);
            aData.setAPDOverPNMean(aopMean);
            aData.setAPDOverPNRMS(aopRms);
            aData.setTaskStatus(channelBad);
 
            MonTimingLaserGreenCrystalDat& tData(l2Time[logicID]);
            tData.setTimingMean(timeMean);
            tData.setTimingRMS(timeRms);
            tData.setTaskStatus(channelBad);
          }
          break;
        case 3:
          {
            MonLaserIRedDat& aData(l3Amp[logicID]);
            aData.setAPDMean(ampMean);
            aData.setAPDRMS(ampRms);
            aData.setAPDOverPNMean(aopMean);
            aData.setAPDOverPNRMS(aopRms);
            aData.setTaskStatus(channelBad);

            MonTimingLaserIRedCrystalDat& tData(l3Time[logicID]);
            tData.setTimingMean(timeMean);
            tData.setTimingRMS(timeRms);
            tData.setTaskStatus(channelBad);
          }
          break;
        case 4:
          {
            MonLaserRedDat& aData(l4Amp[logicID]);
            aData.setAPDMean(ampMean);
            aData.setAPDRMS(ampRms);
            aData.setAPDOverPNMean(aopMean);
            aData.setAPDOverPNRMS(aopRms);
            aData.setTaskStatus(channelBad);

            MonTimingLaserRedCrystalDat& tData(l4Time[logicID]);
            tData.setTimingMean(timeMean);
            tData.setTimingRMS(timeRms);
            tData.setTaskStatus(channelBad);
          }
          break;
        }
        result &= qualityOK(channelStatus);
      }

      for(unsigned iMD(0); iMD < memDCC.size(); ++iMD){
        unsigned iDCC(memDCC[iMD]);

        int subdet(iDCC <= kEEmHigh || iDCC >= kEEpLow ? EcalEndcap : EcalBarrel);

        for(unsigned iPN(1); iPN <= 10; ++iPN){
          EcalPnDiodeDetId pnid(subdet, iDCC + 1, iPN);

          float entries(pnME.getBinEntries(pnid));
          if(entries < 1.) continue;

          float mean(pnME.getBinContent(pnid));
          float rms(pnME.getBinError(pnid) * std::sqrt(entries));

          float pedestalEntries(pnPedestalME.getBinEntries(pnid));
          float pedestalMean(pnPedestalME.getBinContent(pnid));
          float pedestalRms(pnPedestalME.getBinError(pnid) * std::sqrt(pedestalEntries));

          int channelStatus(pnQualityME.getBinContent(pnid));
          bool channelBad(channelStatus == kBad || channelStatus == kMBad);

          switch(wl){
          case 1:
            {
              MonPNBlueDat& data(l1PN[lmPNID(pnid)]);
              data.setADCMeanG1(-1.);
              data.setADCRMSG1(-1.);
              data.setPedMeanG1(-1.);
              data.setPedRMSG1(-1.);
              data.setADCMeanG16(mean);
              data.setADCRMSG16(rms);
              data.setPedMeanG16(pedestalMean);
              data.setPedRMSG16(pedestalRms);
              data.setTaskStatus(channelBad);
            }
            break;
          case 2:
            {
              MonPNGreenDat& data(l2PN[lmPNID(pnid)]);
              data.setADCMeanG1(-1.);
              data.setADCRMSG1(-1.);
              data.setPedMeanG1(-1.);
              data.setPedRMSG1(-1.);
              data.setADCMeanG16(mean);
              data.setADCRMSG16(rms);
              data.setPedMeanG16(pedestalMean);
              data.setPedRMSG16(pedestalRms);
              data.setTaskStatus(channelBad);
            }
            break;
          case 3:
            {
              MonPNIRedDat& data(l3PN[lmPNID(pnid)]);
              data.setADCMeanG1(-1.);
              data.setADCRMSG1(-1.);
              data.setPedMeanG1(-1.);
              data.setPedRMSG1(-1.);
              data.setADCMeanG16(mean);
              data.setADCRMSG16(rms);
              data.setPedMeanG16(pedestalMean);
              data.setPedRMSG16(pedestalRms);
              data.setTaskStatus(channelBad);
            }
            break;
          case 4:
            {
              MonPNRedDat& data(l4PN[lmPNID(pnid)]);
              data.setADCMeanG1(-1.);
              data.setADCRMSG1(-1.);
              data.setPedMeanG1(-1.);
              data.setPedRMSG1(-1.);
              data.setADCMeanG16(mean);
              data.setADCRMSG16(rms);
              data.setPedMeanG16(pedestalMean);
              data.setPedRMSG16(pedestalRms);
              data.setTaskStatus(channelBad);
            }
            break;
          }

          result &= qualityOK(channelStatus);

        }
      }
    }

    try{
      if(!l1Amp.empty())
        _db->insertDataArraySet(&l1Amp, &_iov);
      if(!l1Time.empty())
        _db->insertDataArraySet(&l1Time, &_iov);
      if(!l1PN.empty())
        _db->insertDataArraySet(&l1PN, &_iov);
      if(!l2Amp.empty())
        _db->insertDataArraySet(&l2Amp, &_iov);
      if(!l2Time.empty())
        _db->insertDataArraySet(&l2Time, &_iov);
      if(!l2PN.empty())
        _db->insertDataArraySet(&l2PN, &_iov);
      if(!l3Amp.empty())
        _db->insertDataArraySet(&l3Amp, &_iov);
      if(!l3Time.empty())
        _db->insertDataArraySet(&l3Time, &_iov);
      if(!l3PN.empty())
        _db->insertDataArraySet(&l3PN, &_iov);
      if(!l4Amp.empty())
        _db->insertDataArraySet(&l4Amp, &_iov);
      if(!l4Time.empty())
        _db->insertDataArraySet(&l4Time, &_iov);
      if(!l4PN.empty())
        _db->insertDataArraySet(&l4PN, &_iov);
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return result;
  }

  PedestalWriter::PedestalWriter(edm::ParameterSet const& _ps) :
    DBWriterWorker("Pedestal", _ps),
    gainToME_(),
    pnGainToME_()
  {
    std::vector<int> MGPAGains(_ps.getUntrackedParameter<std::vector<int> >("MGPAGains"));
    std::vector<int> MGPAGainsPN(_ps.getUntrackedParameter<std::vector<int> >("MGPAGainsPN"));

    MESet::PathReplacements repl;

    MESetMulti const& pedestal(static_cast<MESetMulti const&>(source_.at("Pedestal")));
    unsigned nG(MGPAGains.size());
    for(unsigned iG(0); iG != nG; ++iG){
      int gain(MGPAGains[iG]);
      if(gain != 1 && gain != 6 && gain != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain";
      repl["gain"] = std::to_string(gain);
      gainToME_[gain] = pedestal.getIndex(repl);
    }

    repl.clear();

    MESetMulti const& pnPedestal(static_cast<MESetMulti const&>(source_.at("PNPedestal")));
    unsigned nGPN(MGPAGainsPN.size());
    for(unsigned iG(0); iG != nGPN; ++iG){
      int gain(MGPAGainsPN[iG]);
      if(gain != 1 && gain != 16) throw cms::Exception("InvalidConfiguration") << "PN MGPA gain";
      repl["pngain"] = std::to_string(gain);
      pnGainToME_[gain] = pnPedestal.getIndex(repl);
    }
  }

  bool
  PedestalWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    /*
      uses
      PedestalTask.Pedestal (h01, h02, h03)
      PedestalTask.PNPedestal (i01, i02)
      PedestalClient.Quality (meg01, meg02, meg03)
      PedestalClient.PNQualitySummary (meg04, meg05)
    */

    bool result(true);

    std::map<EcalLogicID, MonPedestalsDat> pedestals;
    std::map<EcalLogicID, MonPNPedDat> pnPedestals;

    MESet const& pedestalME(source_.at("Pedestal"));
    MESet const& qualityME(source_.at("Quality"));

    MESet const& pnPedestalME(source_.at("PNPedestal"));
    MESet const& pnQualityME(source_.at("PNQuality"));

    for(std::map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      int gain(gainItr->first);
      int iM(gainItr->second);

      static_cast<MESetMulti const&>(pedestalME).use(iM);
      static_cast<MESetMulti const&>(qualityME).use(iM);

      MESet::const_iterator pEnd(pedestalME.end());
      MESet::const_iterator qItr(qualityME);
      for(MESet::const_iterator pItr(pedestalME.beginChannel()); pItr != pEnd; pItr.toNextChannel()){
        float entries(pItr->getBinEntries());
        if(entries < 1.) continue;

        qItr = pItr;

        float mean(pItr->getBinContent());
        float rms(pItr->getBinError() * std::sqrt(entries));

        EcalLogicID logicID(crystalID(pItr->getId()));
        if(pedestals.find(logicID) == pedestals.end()){
          MonPedestalsDat& insertion(pedestals[logicID]);
          insertion.setPedMeanG1(-1.);
          insertion.setPedRMSG1(-1.);
          insertion.setPedMeanG6(-1.);
          insertion.setPedRMSG6(-1.);
          insertion.setPedMeanG12(-1.);
          insertion.setPedRMSG12(-1.);
          insertion.setTaskStatus(false);
        }

        MonPedestalsDat& data(pedestals[logicID]);
        switch(gain){
        case 1:
          data.setPedMeanG1(mean);
          data.setPedRMSG1(rms);
          break;
        case 6:
          data.setPedMeanG6(mean);
          data.setPedRMSG6(rms);
          break;
        case 12:
          data.setPedMeanG12(mean);
          data.setPedRMSG12(rms);
          break;
        }

        int channelStatus(qItr->getBinContent());
        bool channelBad(channelStatus == kBad || channelStatus == kMBad);
        if(channelBad)
          data.setTaskStatus(true);

        result &= qualityOK(channelStatus);
      }
    }

    for(std::map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      int gain(gainItr->first);
      int iM(gainItr->second);

      static_cast<MESetMulti const&>(pnPedestalME).use(iM);
      static_cast<MESetMulti const&>(pnQualityME).use(iM);

      for(unsigned iMD(0); iMD < memDCC.size(); ++iMD){
        unsigned iDCC(memDCC[iMD]);

        int subdet(iDCC <= kEEmHigh || iDCC >= kEEpLow ? EcalEndcap : EcalBarrel);

        for(unsigned iPN(1); iPN <= 10; ++iPN){
          EcalPnDiodeDetId pnid(subdet, iDCC + 1, iPN);

          float entries(pnPedestalME.getBinEntries(pnid));
          if(entries < 1.) continue;

          float mean(pnPedestalME.getBinContent(pnid));
          float rms(pnPedestalME.getBinError(pnid) * std::sqrt(entries));

          EcalLogicID logicID(lmPNID(pnid));
          if(pnPedestals.find(logicID) == pnPedestals.end()){
            MonPNPedDat& insertion(pnPedestals[logicID]);
            insertion.setPedMeanG1(-1.);
            insertion.setPedRMSG1(-1.);
            insertion.setPedMeanG16(-1.);
            insertion.setPedRMSG16(-1.);
            insertion.setTaskStatus(false);
          }

          MonPNPedDat& data(pnPedestals[lmPNID(pnid)]);
          switch(gain){
          case 1:
            data.setPedMeanG1(mean);
            data.setPedRMSG1(rms);
            break;
          case 16:
            data.setPedMeanG16(mean);
            data.setPedRMSG16(rms);
            break;
          }

          int channelStatus(pnQualityME.getBinContent(pnid));
          bool channelBad(channelStatus == kBad || channelStatus == kMBad);
          if(channelBad)
            data.setTaskStatus(true);

          result &= qualityOK(channelStatus);
        }
      }
    }

    try{
      if(!pedestals.empty())
        _db->insertDataArraySet(&pedestals, &_iov);
      if(!pnPedestals.empty())
        _db->insertDataArraySet(&pnPedestals, &_iov);
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return result;
  }

  bool
  PresampleWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    /*
      uses
      PresampleTask.Pedestal (h03)
      PresampleClient.Quality (meg03)
    */

    bool result(true);

    std::map<EcalLogicID, MonPedestalsOnlineDat> pedestals;

    MESet const& pedestalME(source_.at("Pedestal"));
    MESet const& qualityME(source_.at("Quality"));

    MESet::const_iterator pEnd(pedestalME.end());
    MESet::const_iterator qItr(qualityME);
    for(MESet::const_iterator pItr(pedestalME.beginChannel()); pItr != pEnd; pItr.toNextChannel()){
      float entries(pItr->getBinEntries());
      if(entries < 1.) continue;

      qItr = pItr;

      float mean(pItr->getBinContent());
      float rms(pItr->getBinError() * std::sqrt(entries));

      MonPedestalsOnlineDat& data(pedestals[crystalID(pItr->getId())]);
      data.setADCMeanG12(mean);
      data.setADCRMSG12(rms);

      int channelStatus(qItr->getBinContent());
      bool channelBad(channelStatus == kBad || channelStatus == kMBad);
      data.setTaskStatus(channelBad);

      result &= qualityOK(channelStatus);
    }

    try{
      if(!pedestals.empty())
        _db->insertDataArraySet(&pedestals, &_iov);
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return result;
  }

  TestPulseWriter::TestPulseWriter(edm::ParameterSet const& _ps) :
    DBWriterWorker("TestPulse", _ps),
    gainToME_(),
    pnGainToME_()
  {
    std::vector<int> MGPAGains(_ps.getUntrackedParameter<std::vector<int> >("MGPAGains"));
    std::vector<int> MGPAGainsPN(_ps.getUntrackedParameter<std::vector<int> >("MGPAGainsPN"));

    MESet::PathReplacements repl;

    MESetMulti const& amplitude(static_cast<MESetMulti const&>(source_.at("Amplitude")));
    unsigned nG(MGPAGains.size());
    for(unsigned iG(0); iG != nG; ++iG){
      int gain(MGPAGains[iG]);
      if(gain != 1 && gain != 6 && gain != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain";
      repl["gain"] = std::to_string(gain);
      gainToME_[gain] = amplitude.getIndex(repl);
    }

    repl.clear();

    MESetMulti const& pnAmplitude(static_cast<MESetMulti const&>(source_.at("PNAmplitude")));
    unsigned nGPN(MGPAGainsPN.size());
    for(unsigned iG(0); iG != nGPN; ++iG){
      int gain(MGPAGainsPN[iG]);
      if(gain != 1 && gain != 16) throw cms::Exception("InvalidConfiguration") << "PN MGPA gain";
      repl["pngain"] = std::to_string(gain);
      pnGainToME_[gain] = pnAmplitude.getIndex(repl);
    }
  }

  bool
  TestPulseWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    /*
      uses
      TestPulseTask.Amplitude (ha01, ha02, ha03)
      TestPulseTask.Shape (me_hs01, me_hs02, me_hs03)
      TestPulseTask.PNAmplitude (i01, i02)
      PNDiodeTask.Pedestal (i03, i04)
      TestPulseClient.Quality (meg01, meg02, meg03)
      TestPulseClient.PNQualitySummary (meg04, meg05)
    */

    bool result(true);

    std::map<EcalLogicID, MonTestPulseDat> amplitude;
    std::map<EcalLogicID, MonPulseShapeDat> shape;
    std::map<EcalLogicID, MonPNMGPADat> pnAmplitude;

    MESet const& amplitudeME(source_.at("Amplitude"));
    MESet const& shapeME(source_.at("Shape"));
    MESet const& qualityME(source_.at("Quality"));

    MESet const& pnAmplitudeME(source_.at("PNAmplitude"));
    MESet const& pnPedestalME(source_.at("PNPedestal"));
    MESet const& pnQualityME(source_.at("PNQuality"));

    for(std::map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      int gain(gainItr->first);
      int iM(gainItr->second);

      static_cast<MESetMulti const&>(amplitudeME).use(iM);
      static_cast<MESetMulti const&>(shapeME).use(iM);
      static_cast<MESetMulti const&>(qualityME).use(iM);

      MESet::const_iterator aEnd(amplitudeME.end());
      MESet::const_iterator qItr(qualityME);
      for(MESet::const_iterator aItr(amplitudeME.beginChannel()); aItr != aEnd; aItr.toNextChannel()){
        float entries(aItr->getBinEntries());
        if(entries < 1.) continue;

        qItr = aItr;

        float mean(aItr->getBinContent());
        float rms(aItr->getBinError() * std::sqrt(entries));

        EcalLogicID logicID(crystalID(aItr->getId()));
        if(amplitude.find(logicID) == amplitude.end()){
          MonTestPulseDat& insertion(amplitude[logicID]);
          insertion.setADCMeanG1(-1.);
          insertion.setADCRMSG1(-1.);
          insertion.setADCMeanG6(-1.);
          insertion.setADCRMSG6(-1.);
          insertion.setADCMeanG12(-1.);
          insertion.setADCRMSG12(-1.);
          insertion.setTaskStatus(false);
        }

        MonTestPulseDat& data(amplitude[logicID]);
        switch(gain){
        case 1:
          data.setADCMeanG1(mean);
          data.setADCRMSG1(rms);
          break;
        case 6:
          data.setADCMeanG6(mean);
          data.setADCRMSG6(rms);
          break;
        case 12:
          data.setADCMeanG12(mean);
          data.setADCRMSG12(rms);
          break;
        }

        int channelStatus(qItr->getBinContent());
        bool channelBad(channelStatus == kBad || channelStatus == kMBad);
        if(channelBad)
          data.setTaskStatus(true);

        result &= qualityOK(channelStatus);
      }

      for(unsigned iSM(0); iSM < 54; ++iSM){
        std::vector<float> samples(10, 0.);
        std::vector<DetId> ids(getElectronicsMap()->dccConstituents(iSM + 1));
        unsigned nId(ids.size());
        unsigned nChannels(0);
        EcalLogicID logicID;
        for(unsigned iD(0); iD < nId; ++iD){
          DetId& id(ids[iD]);

          if(iD == 0) logicID = crystalID(id);

          if(shapeME.getBinEntries(id, 1) < 1.) continue;

          ++nChannels;

          for(int i(0); i < 10; ++i)
            samples[i] += shapeME.getBinContent(id, i + 1);
        }

	if(nChannels == 0) continue;

        for(int i(0); i < 10; ++i)
          samples[i] /= nChannels;

        if(shape.find(logicID) == shape.end()){
          MonPulseShapeDat& insertion(shape[logicID]);
          std::vector<float> defval(10, -1.);
          insertion.setSamples(defval, 1);
          insertion.setSamples(defval, 6);
          insertion.setSamples(defval, 12);
        }

        MonPulseShapeDat& data(shape[logicID]);
        data.setSamples(samples, gain);
      }
    }

    for(std::map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      int gain(gainItr->first);
      int iM(gainItr->second);

      static_cast<MESetMulti const&>(pnAmplitudeME).use(iM);
      static_cast<MESetMulti const&>(pnQualityME).use(iM);

      for(unsigned iMD(0); iMD < memDCC.size(); ++iMD){
        unsigned iDCC(memDCC[iMD]);

        int subdet(iDCC <= kEEmHigh || iDCC >= kEEpLow ? EcalEndcap : EcalBarrel);

        for(unsigned iPN(1); iPN <= 10; ++iPN){
          EcalPnDiodeDetId pnid(subdet, iDCC + 1, iPN);

          float entries(pnAmplitudeME.getBinEntries(pnid));
          if(entries < 1.) continue;

          float mean(pnAmplitudeME.getBinContent(pnid));
          float rms(pnAmplitudeME.getBinError(pnid) * std::sqrt(entries));
          float pedestalEntries(pnPedestalME.getBinEntries(pnid));
          float pedestalMean(pnPedestalME.getBinContent(pnid));
          float pedestalRms(pnPedestalME.getBinError(pnid) * std::sqrt(pedestalEntries));

          EcalLogicID logicID(lmPNID(pnid));
          if(pnAmplitude.find(logicID) == pnAmplitude.end()){
            MonPNMGPADat& insertion(pnAmplitude[logicID]);
            insertion.setADCMeanG1(-1.);
            insertion.setADCRMSG1(-1.);
            insertion.setPedMeanG1(-1.);
            insertion.setPedRMSG1(-1.);
            insertion.setADCMeanG16(-1.);
            insertion.setADCRMSG16(-1.);
            insertion.setPedMeanG16(-1.);
            insertion.setPedRMSG16(-1.);
            insertion.setTaskStatus(false);
          }

          MonPNMGPADat& data(pnAmplitude[lmPNID(pnid)]);
          switch(gain){
          case 1:
            data.setADCMeanG1(mean);
            data.setADCRMSG1(rms);
            // dynamic pedestal not measured for G1
//             data.setPedMeanG1(pedestalMean);
//             data.setPedRMSG1(pedestalRms);
            break;
          case 16:
            data.setADCMeanG16(mean);
            data.setADCRMSG16(rms);
            data.setPedMeanG16(pedestalMean);
            data.setPedRMSG16(pedestalRms);
            break;
          }

          int channelStatus(pnQualityME.getBinContent(pnid));
          bool channelBad(channelStatus == kBad || channelStatus == kMBad);
          if(channelBad)
            data.setTaskStatus(true);

          result &= qualityOK(channelStatus);
        }
      }
    }

    try{
      if(!amplitude.empty())
        _db->insertDataArraySet(&amplitude, &_iov);
      if(!shape.empty())
        _db->insertDataSet(&shape, &_iov);
      if(!pnAmplitude.empty())
        _db->insertDataArraySet(&pnAmplitude, &_iov);
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return result;
  }

  bool
  TimingWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    /*
      uses
      TimingTask.TimeMap (h01)
      TimingClient.Quality (meg01)
    */

    bool result(true);

    std::map<EcalLogicID, MonTimingCrystalDat> timing;

    MESet const& timingME(source_.at("Timing"));
    MESet const& qualityME(source_.at("Quality"));

    MESet::const_iterator tEnd(timingME.end());
    MESet::const_iterator qItr(qualityME);
    for(MESet::const_iterator tItr(timingME.beginChannel()); tItr != tEnd; tItr.toNextChannel()){
      float entries(tItr->getBinEntries());
      if(entries < 1.) continue;

      qItr = tItr;

      float mean(tItr->getBinContent());
      float rms(tItr->getBinError() * std::sqrt(entries));

      MonTimingCrystalDat& data(timing[crystalID(tItr->getId())]);
      data.setTimingMean(mean);
      data.setTimingRMS(rms);

      int channelStatus(qItr->getBinContent());
      bool channelBad(channelStatus == kBad || channelStatus == kMBad);
      data.setTaskStatus(channelBad);

      result &= qualityOK(channelStatus);
    }

    try{
      if(!timing.empty())
        _db->insertDataArraySet(&timing, &_iov);
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return result;
  }

  LedWriter::LedWriter(edm::ParameterSet const& _ps) :
    DBWriterWorker("Led", _ps),
    wlToME_()
  {
    std::vector<int> ledWavelengths(_ps.getUntrackedParameter<std::vector<int> >("ledWavelengths"));

    // wavelengths are not necessarily ordered
    // create a map wl -> MESet index
    // using Amplitude here but any multi-wavelength plot is fine

    MESet::PathReplacements repl;

    MESetMulti const& amplitude(static_cast<MESetMulti const&>(source_.at("Amplitude")));
    unsigned nWL(ledWavelengths.size());
    for(unsigned iWL(0); iWL != nWL; ++iWL){
      int wl(ledWavelengths[iWL]);
      if(wl != 1 && wl != 2) throw cms::Exception("InvalidConfiguration") << "Led Wavelength";
      repl["wl"] = std::to_string(wl);
      wlToME_[wl] = amplitude.getIndex(repl);
    }
  }

  bool
  LedWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    /*
      uses
      LedTask.Amplitude (h01, h03)
      LedTask.AOverP (h02, h04)
      LedTask.Timing (h09, h10)
      LedClient.Quality (meg01, meg02)
      LedTask.PNAmplitude (i09, i10)
x      LedClient.PNQualitySummary (meg09, meg10)
x      PNDiodeTask.Pedestal (i13, i14)
    */

    bool result(true);

    std::map<EcalLogicID, MonLed1Dat> l1Amp;
    std::map<EcalLogicID, MonTimingLed1CrystalDat> l1Time;
//     std::map<EcalLogicID, MonPNLed1Dat> l1PN;
    std::map<EcalLogicID, MonLed2Dat> l2Amp;
    std::map<EcalLogicID, MonTimingLed2CrystalDat> l2Time;
//     std::map<EcalLogicID, MonPNLed2Dat> l2PN;

    MESet const& ampME(source_.at("Amplitude"));
    MESet const& aopME(source_.at("AOverP"));
    MESet const& timeME(source_.at("Timing"));
    MESet const& qualityME(source_.at("Quality"));

//     MESet const& pnME(source_.at("PNAmplitude"));
//     MESet const& pnQualityME(source_.at("PNQuality"));
//     MESet const& pnPedestalME(source_.at("PNPedestal"));

    for(std::map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
      int wl(wlItr->first);
      unsigned iM(wlItr->second);

      static_cast<MESetMulti const&>(ampME).use(iM);
      static_cast<MESetMulti const&>(aopME).use(iM);
      static_cast<MESetMulti const&>(timeME).use(iM);
      static_cast<MESetMulti const&>(qualityME).use(iM);
//       static_cast<MESetMulti const&>(pnME).use(iM);
//       static_cast<MESetMulti const&>(pnQualityME).use(iM);

      MESet::const_iterator aEnd(ampME.end());
      MESet::const_iterator qItr(qualityME);
      MESet::const_iterator oItr(aopME);
      MESet::const_iterator tItr(timeME);
      for(MESet::const_iterator aItr(ampME.beginChannel()); aItr != aEnd; aItr.toNextChannel()){
        float aEntries(aItr->getBinEntries());
        if(aEntries < 1.) continue;

        qItr = aItr;
        oItr = aItr;
        tItr = aItr;

        DetId id(aItr->getId());

        float ampMean(aItr->getBinContent());
        float ampRms(aItr->getBinError() * std::sqrt(aEntries));

        float aopEntries(oItr->getBinEntries());
        float aopMean(oItr->getBinContent());
        float aopRms(oItr->getBinError() * std::sqrt(aopEntries));

        float timeEntries(tItr->getBinEntries());
        float timeMean(tItr->getBinContent());
        float timeRms(tItr->getBinError() * std::sqrt(timeEntries));

        int channelStatus(qItr->getBinContent());
        bool channelBad(channelStatus == kBad || channelStatus == kMBad);

        EcalLogicID logicID(crystalID(id));

        switch(wl){
        case 1:
          {
            MonLed1Dat& aData(l1Amp[logicID]);
            aData.setVPTMean(ampMean);
            aData.setVPTRMS(ampRms);
            aData.setVPTOverPNMean(aopMean);
            aData.setVPTOverPNRMS(aopRms);
            aData.setTaskStatus(channelBad);
 
            MonTimingLed1CrystalDat& tData(l1Time[logicID]);
            tData.setTimingMean(timeMean);
            tData.setTimingRMS(timeRms);
            tData.setTaskStatus(channelBad);
          }
          break;
        case 2:
          {
            MonLed2Dat& aData(l2Amp[logicID]);
            aData.setVPTMean(ampMean);
            aData.setVPTRMS(ampRms);
            aData.setVPTOverPNMean(aopMean);
            aData.setVPTOverPNRMS(aopRms);
            aData.setTaskStatus(channelBad);
 
            MonTimingLed2CrystalDat& tData(l2Time[logicID]);
            tData.setTimingMean(timeMean);
            tData.setTimingRMS(timeRms);
            tData.setTaskStatus(channelBad);
          }
          break;
        }
        result &= qualityOK(channelStatus);
      }

//       for(unsigned iMD(0); iMD < memDCC.size(); ++iMD){
//         unsigned iDCC(memDCC[iMD]);

//         if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;

//         for(unsigned iPN(1); iPN <= 10; ++iPN){
//           EcalPnDiodeDetId pnid(EcalEndcap, iDCC + 1, iPN);

//           float entries(pnME.getBinEntries(pnid));
//           if(entries < 1.) continue;

//           float mean(pnME.getBinContent(pnid));
//           float rms(pnME.getBinError(pnid) * std::sqrt(entries));

//           float pedestalEntries(pnPedestalME.getBinEntries(pnid));
//           float pedestalMean(pnPedestalME.getBinContent(pnid));
//           float pedestalRms(pnPedestalME.getBinError(pnid) * std::sqrt(pedestalEntries));

//           int channelStatus(pnQualityME.getBinContent(pnid));
//           bool channelBad(channelStatus == kBad || channelStatus == kMBad);

//           switch(wl){
//           case 1:
//             {
//               MonPNLed1Dat& data(l1PN[lmPNID(pnid)]);
//               data.setADCMeanG1(-1.);
//               data.setADCRMSG1(-1.);
//               data.setPedMeanG1(-1.);
//               data.setPedRMSG1(-1.);
//               data.setADCMeanG16(mean);
//               data.setADCRMSG16(rms);
//               data.setPedMeanG16(pedestalMean);
//               data.setPedRMSG16(pedestalRms);
//               data.setTaskStatus(channelBad);
//             }
//             break;
//           case 2:
//             {
//               MonPNLed2Dat& data(l2PN[lmPNID(pnid)]);
//               data.setADCMeanG1(-1.);
//               data.setADCRMSG1(-1.);
//               data.setPedMeanG1(-1.);
//               data.setPedRMSG1(-1.);
//               data.setADCMeanG16(mean);
//               data.setADCRMSG16(rms);
//               data.setPedMeanG16(pedestalMean);
//               data.setPedRMSG16(pedestalRms);
//               data.setTaskStatus(channelBad);
//             }
//             break;
//           }

//           result &= qualityOK(channelStatus);

//         }
//       }
    }

    try{
      if(!l1Amp.empty())
        _db->insertDataArraySet(&l1Amp, &_iov);
      if(!l1Time.empty())
        _db->insertDataArraySet(&l1Time, &_iov);
//       if(l1PN.size() > 0)
//         _db->insertDataArraySet(&l1PN, &_iov);
      if(!l2Amp.empty())
        _db->insertDataArraySet(&l2Amp, &_iov);
      if(!l2Time.empty())
        _db->insertDataArraySet(&l2Time, &_iov);
//       if(l2PN.size() > 0)
//         _db->insertDataArraySet(&l2PN, &_iov);
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return result;
  }

  bool
  OccupancyWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    /*
      uses
      OccupancyTask.Digi (i01)
      EnergyTask.HitMap (i02)
    */
    std::map<EcalLogicID, MonOccupancyDat> occupancy;

    MESet const& occupancyME(source_.at("Occupancy"));
    MESet const& energyME(source_.at("Energy"));

    MESet::const_iterator oEnd(occupancyME.end());
    MESet::const_iterator eItr(energyME);
    for(MESet::const_iterator oItr(occupancyME.beginChannel()); oItr != oEnd; oItr.toNextChannel()){

      if(oItr->getME()->getTH1()->GetEntries() < 1000.) continue;

      int entries(oItr->getBinContent());
      if(entries < 10) continue;

      eItr = oItr;

      int eEntries(eItr->getBinEntries());
      float energy(eEntries > 10 ? eItr->getBinContent() : -1.);

      MonOccupancyDat& data(occupancy[crystalID(oItr->getId())]);
      data.setEventsOverLowThreshold(entries);
      data.setEventsOverHighThreshold(eEntries);
      data.setAvgEnergy(energy);
    }

    try{
      if(!occupancy.empty())
        _db->insertDataArraySet(&occupancy, &_iov);
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return true;
  }

  bool
  SummaryWriter::run(EcalCondDBInterface* _db, MonRunIOV& _iov)
  {
    std::map<EcalLogicID, MonRunDat> dataset;
    MonRunDat& runDat(dataset[ecalID()]);

    // CURRENTLY ONLY WRITING SUCCESS
    MonRunOutcomeDef outcomeDef;
    outcomeDef.setShortDesc("success");

    runDat.setNumEvents(processedEvents_);
    runDat.setMonRunOutcomeDef(outcomeDef);
    runDat.setTaskList(taskList_);
    runDat.setTaskOutcome(outcome_);

    try{
      _db->insertDataSet(&dataset, &_iov);
    }
    catch(std::runtime_error& e){
      if(std::string(e.what()).find("unique constraint") != std::string::npos)
        edm::LogWarning("EcalDQM") << e.what();
      else
        throw cms::Exception("DBError") << e.what();
    }

    return true;
  }
}
