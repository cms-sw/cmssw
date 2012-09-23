#include "DQM/EcalCommon/interface/EcalCondDBWriter.h"

#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "OnlineDB/EcalCondDB/interface/MonCrystalConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTTConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonMemChConsistencyDat.h"
#include "OnlineDB/EcalCondDB/interface/MonMemTTConsistencyDat.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

enum Quality {
  kBad = 0,
  kGood = 1,
  kUnknown = 2,
  kMBad = 3,
  kMGood = 4,
  kMUnknown = 5
};

enum Constants {
  nChannels = EBDetId::kSizeForDenseIndexing + EEDetId::kSizeForDenseIndexing,
  nTowers = EcalTrigTowerDetId::kEBTotalTowers + EcalScDetId::kSizeForDenseIndexing
};

EcalLogicID
crystalID(DetId const& _id)
{
  using namespace ecaldqm;
  unsigned iDCC(dccId(_id) - 1);
  if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
    int ism(iDCC <= kEEmHigh ? 10 + iDCC - kEEmLow : 1 + iDCC - kEEpLow);
    EEDetId eeid(_id);
    int index(eeid.ix() * 1000 + eeid.iy());
    return LogicID::getEcalLogicID("EE_crystal_number", ism, index);
  }
  else{
    int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
    EBDetId ebid(_id);
    return LogicID::getEcalLogicID("EB_crystal_number", ism, ebid.ic());
  }
}

EcalLogicID
towerID(EcalElectronicsId const& _id)
{
  using namespace ecaldqm;
  unsigned iDCC(_id.dccId() - 1);
  if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
    int ism(iDCC <= kEEmHigh ? 10 + iDCC - kEEmLow : 1 + iDCC - kEEpLow);
    return LogicID::getEcalLogicID("EE_readout_tower", ism, _id.towerId());
  }
  else{
    int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
    return LogicID::getEcalLogicID("EB_trigger_tower", ism, _id.towerId());
  }
}

EcalLogicID
memChannelID(EcalPnDiodeDetId const& _id)
{
  // using the PN ID degenerates the logic ID - 50 time samples are actually split into 5 channels each
  using namespace ecaldqm;
  unsigned iDCC(_id.iDCCId() - 1);
  int pnid(_id.iPnId());
  if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
    int ism(iDCC <= kEEmHigh ? 10 + iDCC - kEEmLow : 1 + iDCC - kEEpLow);
    return LogicID::getEcalLogicID("EE_mem_channel", ism, (pnid - 1) % 5 + ((pnid - 1) / 5) * 25 + 1);
  }
  else{
    int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
    return LogicID::getEcalLogicID("EB_mem_channel", ism, (pnid - 1) % 5 + ((pnid - 1) / 5) * 25 + 1);
  }
}

EcalLogicID
memTowerID(EcalElectronicsId const& _id)
{
  using namespace ecaldqm;
  unsigned iDCC(_id.dccId() - 1);
  if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
    int ism(iDCC <= kEEmHigh ? 10 + iDCC - kEEmLow : 1 + iDCC - kEEpLow);
    return LogicID::getEcalLogicID("EE_mem_TT", ism, _id.towerId());
  }
  else{
    int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
    return LogicID::getEcalLogicID("EB_mem_TT", ism, _id.towerId());
  }
}

bool
EcalCondDBWriter::writeIntegrity(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  using namespace ecaldqm;
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

  MESet const* digiME(_meSets["Digi"]);
  MESet const* gainME(_meSets["Gain"]);
  MESet const* chidME(_meSets["ChId"]);
  MESet const* gainswitchME(_meSets["GainSwitch"]);
  MESet const* qualityME(_meSets["Quality"]);
  if(!digiME || !gainME || !chidME || !gainswitchME || !qualityME)
    throw cms::Exception("Configuration") << "Channel integrity MEs not found";

  MESet const* toweridME(_meSets["TowerId"]);
  MESet const* blocksizeME(_meSets["BlockSize"]);
  MESet const* l1aME(_meSets["L1AFE"]);
  MESet const* bxME(_meSets["BXFE"]);
  if(!toweridME || !blocksizeME || !l1aME || !bxME)
    throw cms::Exception("Configuration") << "Tower integrity MEs not found";

  MESet const* memdigiME(_meSets["MEMDigi"]);
  MESet const* memchidME(_meSets["MEMChId"]);
  MESet const* memgainME(_meSets["MEMGain"]);
  MESet const* pnqualityME(_meSets["PNQuality"]);
  if(!memdigiME || !memchidME || !memgainME || !pnqualityME)
    throw cms::Exception("Configuration") << "MEM channel integrity MEs not found";

  MESet const* memtoweridME(_meSets["MEMTowerId"]);
  MESet const* memblocksizeME(_meSets["MEMBlockSize"]);
  if(!memtoweridME || !memblocksizeME)
    throw cms::Exception("Configuration") << "MEM tower integrity MEs not found";

  MESet::const_iterator dEnd(digiME->end());
  MESet::const_iterator qItr(qualityME);
  for(MESet::const_iterator dItr(digiME->beginChannel()); dItr != dEnd; dItr.toNextChannel()){
    DetId id(dItr->getId());

    int nDigis(dItr->getBinContent());
    int gain(gainME->getBinContent(id));
    int chid(chidME->getBinContent(id));
    int gainswitch(gainswitchME->getBinContent(id));
    qItr = dItr;

    if(gain > 0 || chid > 0 || gainswitch > 0){
      MonCrystalConsistencyDat& data(crystalConsistencies[crystalID(id)]);
      data.setProcessedEvents(nDigis);
      data.setProblematicEvents(gain + chid + gainswitch);
      data.setProblemsGainZero(gain);
      data.setProblemsID(chid);
      data.setProblemsGainSwitch(gainswitch);

      int channelStatus(qItr->getBinContent());
      bool channelGood(channelStatus != kBad && channelStatus != kMBad);
      data.setTaskStatus(channelGood);

      result &= channelGood;
    }
  }

  for(unsigned iDCC(kEEmLow); iDCC <= kEBpHigh; ++iDCC){
    for(unsigned iTower(1); iTower <= 68; ++iTower){
      if(!ccuExists(iDCC + 1, iTower)) continue;

      EcalElectronicsId eid(iDCC + 1, iTower, 1, 1);
      std::vector<DetId> channels(getElectronicsMap()->dccTowerConstituents(iDCC + 1, iTower));
      int nDigis(0);
      bool towerGood(false);
      for(unsigned iD(0); iD < channels.size(); ++iD){
        int n(digiME->getBinContent(channels[iD]));
        if(n > nDigis) nDigis = n;
        int channelStatus(qualityME->getBinContent(channels[iD]));
        if(channelStatus != kBad && channelStatus != kMBad) towerGood = true;
      }

      int towerid(toweridME->getBinContent(eid));
      int blocksize(blocksizeME->getBinContent(eid));
      int l1a(l1aME->getBinContent(iDCC + 1, iTower));
      int bx(bxME->getBinContent(iDCC + 1, iTower));

      if(towerid > 0 || blocksize > 0 || l1a > 0 || bx > 0){
        MonTTConsistencyDat& data(towerConsistencies[towerID(eid)]);
        data.setProcessedEvents(nDigis);
        data.setProblematicEvents(towerid + blocksize + l1a + bx);
        data.setProblemsID(towerid);
        data.setProblemsSize(blocksize);
        data.setProblemsLV1(l1a);
        data.setProblemsBunchX(bx);
        data.setTaskStatus(towerGood);

        result &= towerGood;
      }
    }

    int subdet(iDCC <= kEEmHigh || iDCC >= kEEpLow ? EcalEndcap : EcalBarrel);

    for(unsigned iPN(1); iPN <= 10; ++iPN){
      EcalPnDiodeDetId pnid(subdet, iDCC + 1, iPN);

      int nDigis(memdigiME->getBinContent(pnid));
      int memchid(memchidME->getBinContent(pnid));
      int memgain(memgainME->getBinContent(pnid));

      if(memchid > 0 || memgain > 0){
        MonMemChConsistencyDat& data(memChannelConsistencies[memChannelID(pnid)]);

        data.setProcessedEvents(nDigis);
        data.setProblematicEvents(memchid + memgain);
        data.setProblemsID(memchid);
        data.setProblemsGainZero(memgain);

        int channelStatus(pnqualityME->getBinContent(pnid));
        bool channelGood(channelStatus != kBad && channelStatus != kMBad);
        data.setTaskStatus(channelGood);

        result &= channelGood;
      }
    }

    for(unsigned iTower(69); iTower <= 70; ++iTower){
      EcalElectronicsId eid(iDCC + 1, iTower, 1, 1);

      int nDigis(0);
      bool towerGood(false);
      for(unsigned iPN(1); iPN <= 10; ++iPN){
        EcalPnDiodeDetId pnid(subdet, iDCC + 1, iPN);
        int n(memdigiME->getBinContent(pnid));
        if(n > nDigis) nDigis = n;
        int channelStatus(pnqualityME->getBinContent(pnid));
        if(channelStatus != kBad && channelStatus != kMBad) towerGood = true;
      }

      int towerid(memtoweridME->getBinContent(eid));
      int blocksize(memblocksizeME->getBinContent(eid));

      if(towerid > 0 || blocksize > 0){
        MonMemTTConsistencyDat& data(memTowerConsistencies[memTowerID(eid)]);

        data.setProcessedEvents(nDigis);
        data.setProblematicEvents(towerid + blocksize);
        data.setProblemsID(towerid);
        data.setProblemsSize(blocksize);
        data.setTaskStatus(towerGood);

        result &= towerGood;
      }
    }
  }

  try{
    db_->insertDataArraySet(&crystalConsistencies, &_iov);
    db_->insertDataArraySet(&towerConsistencies, &_iov);
    db_->insertDataArraySet(&memChannelConsistencies, &_iov);
    db_->insertDataArraySet(&memTowerConsistencies, &_iov);
  }
  catch(std::runtime_error& e){
    throw cms::Exception("DBError") << e.what();
  }

  return result;
}

bool
EcalCondDBWriter::writeLaser(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  return true;
}

bool
EcalCondDBWriter::writePedestal(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  return true;
}

bool
EcalCondDBWriter::writePresample(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  return true;
}

bool
EcalCondDBWriter::writeTestPulse(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  return true;
}

bool
EcalCondDBWriter::writeTiming(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  return true;
}

bool
EcalCondDBWriter::writeLed(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  return true;
}

bool
EcalCondDBWriter::writeRawData(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  return true;
}

bool
EcalCondDBWriter::writeOccupancy(PtrMap<std::string, ecaldqm::MESet const> const& _meSets, MonRunIOV& _iov)
{
  return true;
}
