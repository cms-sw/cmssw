#ifndef SISTRIPPOPCON_UNITTEST_HANDLER_GAIN_H
#define SISTRIPPOPCON_UNITTEST_HANDLER_GAIN_H

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/CondDB/interface/Types.h"

#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"

#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>
#include <ctime>

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

namespace popcon {

  template <typename T>
  class SiStripPopConHandlerUnitTestGain : public popcon::PopConSourceHandler<T> {
  public:
    enum DataType { UNDEFINED = 0, _Cabling = 1, _Pedestal = 2, _Noise = 3, _Threshold = 4, _BadStrip = 5, _Gain = 6 };

    //---------------------------------------
    //
    SiStripPopConHandlerUnitTestGain(const edm::ParameterSet& pset)
        : m_name(pset.getUntrackedParameter<std::string>("name", "SiStripPopPopConConfigDbObjHandler")),
          m_since(pset.getUntrackedParameter<uint32_t>("since", 5)),
          m_debugMode(pset.getUntrackedParameter<bool>("debug", true)){};

    //---------------------------------------
    //
    ~SiStripPopConHandlerUnitTestGain() override{};

    //---------------------------------------
    //
    void getNewObjects() override {
      edm::LogInfo("SiStripPopPopConConfigDbObjHandler") << "[getNewObjects] for PopCon application " << m_name;

      if (m_debugMode) {
        std::stringstream ss;
        ss << "\n\n------- " << m_name << " - > getNewObjects\n";
        if (this->tagInfo().size) {
          //check whats already inside of database
          ss << "got offlineInfo" << this->tagInfo().name << ", size " << this->tagInfo().size
             << " , last object valid since " << this->tagInfo().lastInterval.since << " token "
             << this->tagInfo().lastInterval.payloadId << "\n\n UserText " << this->userTextLog() << "\n LogDBEntry \n"
             << this->logDBEntry().logId << "\n"
             << this->logDBEntry().destinationDB << "\n"
             << this->logDBEntry().provenance << "\n"
             << this->logDBEntry().usertext << "\n"
             << this->logDBEntry().iovtag << "\n"
             << this->logDBEntry().iovtimetype << "\n"
             << this->logDBEntry().payloadIdx << "\n"
             << this->logDBEntry().payloadClass << "\n"
             << this->logDBEntry().payloadToken << "\n"
             << this->logDBEntry().exectime << "\n"
             << this->logDBEntry().execmessage << "\n"
             << "\n\n-- user text "
             << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of('@'));
        } else {
          ss << " First object for this tag ";
        }
        edm::LogInfo("SiStripPopPopConConfigDbObjHandler") << ss.str();
      }
      if (isTransferNeeded())
        setForTransfer();

      edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
          << "[getNewObjects] for PopCon application " << m_name << " Done\n--------------\n";
    }

    //---------------------------------------
    //
    std::string id() const override { return m_name; }

  private:
    //methods

    DataType getDataType() {
      if (typeid(T) == typeid(SiStripFedCabling)) {
        edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
            << "[getDataType] for PopCon application " << m_name << " " << typeid(T).name();
        return _Cabling;
      }
      return UNDEFINED;
    }

    //---------------------------------------
    //
    bool isTransferNeeded() {
      edm::LogInfo("SiStripPopPopConConfigDbObjHandler") << "[isTransferNeeded] checking for transfer" << std::endl;
      std::stringstream ss_logdb, ss;
      std::stringstream ss1;

      //get log information from previous upload
      if (this->tagInfo().size)
        ss_logdb << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of('@'));
      else
        ss_logdb << "";

      ss << "@ " << clock();

      if (!strcmp(ss.str().c_str(), ss_logdb.str().c_str())) {
        //string are equal, no need to do transfer
        edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
            << "[isTransferNeeded] the selected conditions are already uploaded in the last iov ("
            << this->tagInfo().lastInterval.since << ") open for the object " << this->logDBEntry().payloadClass
            << " in the db " << this->logDBEntry().destinationDB << " parameters: " << ss.str()
            << "\n NO TRANSFER NEEDED";
        return false;
      }
      this->m_userTextLog = ss.str();
      edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
          << "[isTransferNeeded] the selected conditions will be uploaded: " << ss.str() << "\n A- " << ss.str()
          << "\n B- " << ss_logdb.str() << "\n Fine";

      return true;
    }

    //---------------------------------------
    //
    void setForTransfer() {
      edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
          << "[setForTransfer] " << m_name << " getting data to be transferred " << std::endl;

      T* obj = nullptr;

      fillObject(obj);

      if (!this->tagInfo().size)
        m_since = 1;
      else if (m_debugMode)
        m_since = this->tagInfo().lastInterval.since + 1;

      if (obj != nullptr) {
        edm::LogInfo("SiStripPopPopConConfigDbObjHandler") << "setting since = " << m_since << std::endl;
        this->m_to_transfer.push_back(std::make_pair(obj, m_since));
      } else {
        edm::LogError("SiStripPopPopConConfigDbObjHandler")
            << "[setForTransfer] " << m_name << "  : NULL pointer of obj " << typeid(T).name()
            << " reported by SiStripCondObjBuilderFromDb\n Transfer aborted" << std::endl;
      }
    }

  private:
    // data members
    std::string m_name;
    unsigned long long m_since;
    bool m_debugMode;
    edm::Service<SiStripCondObjBuilderFromDb> condObjBuilder;

    void fillObject(T*& obj) {
      std::cout << __LINE__ << std::endl;

      if (typeid(T) == typeid(SiStripApvGain)) {
        std::cout << __LINE__ << std::endl;

        obj = new SiStripApvGain();

        const auto detInfo =
            SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
        int count = -1;
        for (const auto& it : detInfo.getAllData()) {
          count++;
          //Generate Gains for det detid
          SiStripApvGain::InputVector inputApvGain;
          for (int apv = 0; apv < it.second.nApvs; ++(++apv)) {
            float MeanTick = 555.;
            float RmsTick = 55.;

            float tick = CLHEP::RandGauss::shoot(MeanTick, RmsTick);

            if (count < 6)
              edm::LogInfo("SiStripGainBuilder") << "detid " << it.first << " \t"
                                                 << " APV " << apv << " \t" << tick << " \t" << std::endl;
            inputApvGain.push_back(tick);  //APV0
            inputApvGain.push_back(tick);  //APV1
          }

          SiStripApvGain::Range gain_range(inputApvGain.begin(), inputApvGain.end());
          if (!obj->put(it.first, gain_range))
            edm::LogError("SiStripGainBuilder") << "[SiStripGainBuilder::analyze] detid already exists" << std::endl;
        }
      }
    }
  };

}  // namespace popcon

#endif  //SISTRIPPOPCON_UNITTEST_HANDLER_H
