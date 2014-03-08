#include "DQM/EcalCommon/interface/StatusManager.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CommonTools/Utils/interface/Exception.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

#include "TPRegexp.h"
#include "TObjArray.h"
#include "TString.h"

namespace ecaldqm
{

  StatusManager::StatusManager() :
    dictionary_(),
    status_()
  {
    dictionary_["CH_ID_ERROR"] = 0x1 << EcalDQMStatusHelper::CH_ID_ERROR;
    dictionary_["CH_GAIN_ZERO_ERROR"] = 0x1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR;
    dictionary_["CH_GAIN_SWITCH_ERROR"] = 0x1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR;
    dictionary_["TT_ID_ERROR"] = 0x1 << EcalDQMStatusHelper::TT_ID_ERROR;
    dictionary_["TT_SIZE_ERROR"] = 0x1 << EcalDQMStatusHelper::TT_SIZE_ERROR;

    dictionary_["PEDESTAL_LOW_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR;
    dictionary_["PEDESTAL_MIDDLE_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR;
    dictionary_["PEDESTAL_HIGH_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR;
    dictionary_["PEDESTAL_LOW_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR;
    dictionary_["PEDESTAL_MIDDLE_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR;
    dictionary_["PEDESTAL_HIGH_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR;

    dictionary_["PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR;
    dictionary_["PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;

    dictionary_["TESTPULSE_LOW_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR;
    dictionary_["TESTPULSE_MIDDLE_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR;
    dictionary_["TESTPULSE_HIGH_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR;
    dictionary_["TESTPULSE_LOW_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR;
    dictionary_["TESTPULSE_MIDDLE_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR;
    dictionary_["TESTPULSE_HIGH_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR;

    dictionary_["LASER_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::LASER_MEAN_ERROR;
    dictionary_["LASER_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::LASER_RMS_ERROR;
    dictionary_["LASER_TIMING_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR;
    dictionary_["LASER_TIMING_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR;

    dictionary_["LED_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::LED_MEAN_ERROR;
    dictionary_["LED_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::LED_RMS_ERROR;
    dictionary_["LED_TIMING_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR;
    dictionary_["LED_TIMING_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR;

    dictionary_["STATUS_FLAG_ERROR"] = 0x1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR;

    dictionary_["PHYSICS_BAD_CHANNEL_WARNING"] = 0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING;
    dictionary_["PHYSICS_BAD_CHANNEL_ERROR"] = 0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR;

    dictionary_["disabled_channel"] =
      0x1 << EcalDQMStatusHelper::TT_SIZE_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LASER_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LED_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
      0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR;

    dictionary_["dead_channel"] =
      0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR | 
      0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::LASER_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LASER_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::LED_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LED_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR | 
      0x1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
      0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR;

    dictionary_["pedestal_problem"] =
      0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;

    dictionary_["testpulse_problem"] =
      0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR;

    dictionary_["laser_problem"] =
      0x1 << EcalDQMStatusHelper::LASER_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LASER_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR;

    dictionary_["led_problem"] =
      0x1 << EcalDQMStatusHelper::LED_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LED_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR |
      0x1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR;

    dictionary_["noise_problem"] =
      0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
      0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR;
  }

  void
  StatusManager::readFromStream(std::istream& _input)
  {
    TPRegexp linePat("^[ ]*(Crystal|TT|PN)[ ]+(EB[0-9+-]*|EE[0-9+-]*|[0-9]+)[ ]+([0-9]+)[ ]([a-zA-Z_]+)");

    std::string line;
    while(true){
      std::getline(_input, line);
      if(!_input.good()) break;

      if(!linePat.MatchB(line)) continue;

      TObjArray* matches(linePat.MatchS(line));
      TString channelType(matches->At(1)->GetName());
      TString module(matches->At(2)->GetName());
      unsigned channel(TString(matches->At(3)->GetName()).Atoi());
      TString statusName(matches->At(4)->GetName());
      delete matches;

      std::map<std::string, uint32_t>::const_iterator dItr(dictionary_.find(statusName.Data()));
      if(dItr == dictionary_.end()) continue;
      uint32_t statusVal(dItr->second);

      if(channelType == "Crystal"){
        // module: Subdetector name, channel: dense ID
        // Store using EBDetId and EEDetId as keys (following EcalDQMChannelStatus)

        if(module == "EB"){
          if(!EBDetId::validDenseIndex(channel)) continue;
          status_.insert(std::pair<uint32_t, uint32_t>(EBDetId::unhashIndex(channel).rawId(), statusVal));
        }
        else if(module == "EE"){
          if(!EEDetId::validDenseIndex(channel)) continue;
          status_.insert(std::pair<uint32_t, uint32_t>(EEDetId::unhashIndex(channel).rawId(), statusVal));
        }
      }
      else if(channelType == "TT"){
        // module: Supermodule name, channel: RU ID (electronics ID tower)
        // Store using EcalTrigTowerDetId and EcalScDetId as keys (following EcalDQMTowerStatus)

        if(module.Contains("EB")){
          /* TODO CHECK THIS */

          int iEta((channel - 1) / 4 + 1);
          int zside(0);
          int iPhi(0);
          if(module(3) == '-'){
            zside = -1;
            iPhi = (channel - 1) % 4 + 1;
          }
          else{
            zside = 1;
            iPhi = (68 - channel) % 4 + 1;
          }

          status_.insert(std::pair<uint32_t, uint32_t>(EcalTrigTowerDetId(zside, EcalBarrel, iEta, iPhi).rawId(), statusVal));
        }
        else if(module.Contains("EE")){
          std::vector<EcalScDetId> scIds(getElectronicsMap()->getEcalScDetId(dccId(module.Data()), channel, false));
          for(unsigned iS(0); iS != scIds.size(); ++iS)
            status_.insert(std::pair<uint32_t, uint32_t>(scIds[iS].rawId(), statusVal));
        }
      }
      else if(channelType == "PN"){
        // module: DCC ID, channel: iPN
        // Store using EcalPnDiodeDetId as keys
        unsigned iDCC(module.Atoi() - 1);
        int subdet(iDCC <= kEEmHigh || iDCC >= kEEpLow ? EcalEndcap : EcalBarrel);
        status_.insert(std::pair<uint32_t, uint32_t>(EcalPnDiodeDetId(subdet, iDCC + 1, channel).rawId(), statusVal));
      }
    }
  }

  void
  StatusManager::readFromObj(EcalDQMChannelStatus const& _channelStatus, EcalDQMTowerStatus const& _towerStatus)
  {
    EcalDQMChannelStatus::Items const& barrelChStatus(_channelStatus.barrelItems());
    for(unsigned iC(0); iC != EBDetId::kSizeForDenseIndexing; ++iC)
      status_.insert(std::pair<uint32_t, uint32_t>(EBDetId::unhashIndex(iC).rawId(), barrelChStatus[iC].getStatusCode()));

    EcalDQMChannelStatus::Items const& endcapChStatus(_channelStatus.endcapItems());
    for(unsigned iC(0); iC != EEDetId::kSizeForDenseIndexing; ++iC)
      status_.insert(std::pair<uint32_t, uint32_t>(EEDetId::unhashIndex(iC).rawId(), endcapChStatus[iC].getStatusCode()));

    EcalDQMTowerStatus::Items const& barrelTowStatus(_towerStatus.barrelItems());
    for(unsigned iC(0); iC != EcalTrigTowerDetId::kEBTotalTowers; ++iC)
      status_.insert(std::pair<uint32_t, uint32_t>(EcalTrigTowerDetId::detIdFromDenseIndex(iC).rawId(), barrelTowStatus[iC].getStatusCode()));

    EcalDQMTowerStatus::Items const& endcapTowStatus(_towerStatus.endcapItems());
    for(unsigned iC(0); iC != EcalScDetId::kSizeForDenseIndexing; ++iC)
      status_.insert(std::pair<uint32_t, uint32_t>(EcalScDetId::unhashIndex(iC).rawId(), endcapTowStatus[iC].getStatusCode()));
  }

  void
  StatusManager::writeToStream(std::ostream& _output) const
  {

  }

  void
  StatusManager::writeToObj(EcalDQMChannelStatus& _channelStatus, EcalDQMTowerStatus& _towerStatus) const
  {
    for(unsigned iC(0); iC != EBDetId::kSizeForDenseIndexing; ++iC){
      uint32_t key(EBDetId::unhashIndex(iC).rawId());
      _channelStatus.setValue(key, EcalDQMStatusCode(getStatus(key)));
    }

    for(unsigned iC(0); iC != EEDetId::kSizeForDenseIndexing; ++iC){
      uint32_t key(EEDetId::unhashIndex(iC).rawId());
      _channelStatus.setValue(key, EcalDQMStatusCode(getStatus(key)));
    }

    for(unsigned iC(0); iC != EcalTrigTowerDetId::kEBTotalTowers; ++iC){
      uint32_t key(EcalTrigTowerDetId::detIdFromDenseIndex(iC));
      _towerStatus.setValue(key, EcalDQMStatusCode(getStatus(key)));
    }

    for(unsigned iC(0); iC != EcalScDetId::kSizeForDenseIndexing; ++iC){
      uint32_t key(EcalScDetId::unhashIndex(iC));
      _towerStatus.setValue(key, EcalDQMStatusCode(getStatus(key)));
    }
  }

  uint32_t
  StatusManager::getStatus(uint32_t _key) const
  {
    std::map<uint32_t, uint32_t>::const_iterator itr(status_.find(_key));
    if(itr == status_.end()) return 0;
    return itr->second;
  }

}
