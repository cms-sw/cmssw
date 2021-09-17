#ifndef SISTRIPPOPCON_CONFIGDB_HANDLER_H
#define SISTRIPPOPCON_CONFIGDB_HANDLER_H

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

namespace popcon {

  template <typename T>
  class SiStripPopConConfigDbObjHandler : public popcon::PopConSourceHandler<T> {
  public:
    //---------------------------------------
    //
    SiStripPopConConfigDbObjHandler(const edm::ParameterSet& pset)
        : m_name(pset.getUntrackedParameter<std::string>("name", "SiStripPopPopConConfigDbObjHandler")),
          m_since(pset.getUntrackedParameter<uint32_t>("since", 5)),
          m_debugMode(pset.getUntrackedParameter<bool>("debug", false)){};

    //---------------------------------------
    //
    ~SiStripPopConConfigDbObjHandler() override{};

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

    //---------------------------------------
    //
    bool isTransferNeeded() {
      edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
          << "[isTransferNeeded] checking for transfer: " << typeid(T).name() << std::endl;
      std::stringstream ss_logdb, ss;

      //get log information from previous upload
      if (!this->logDBEntry().usertext.empty())
        ss_logdb << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_first_of('@'));
      else
        ss_logdb << "";

      std::string label = "";
      if (typeid(T) == typeid(SiStripFedCabling))
        label = "Cabling";

      if (typeid(T) == typeid(SiStripApvGain))
        label = "ApvTiming";

      if (typeid(T) == typeid(SiStripLatency))
        label = "ApvLatency";

      if (!condObjBuilder->checkForCompatibility(ss_logdb, ss, label)) {
        //string are equal, no need to do transfer
        edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
            << "[isTransferNeeded] the selected conditions are already uploaded in the last iov ("
            << this->tagInfo().lastInterval.since << ") open for the object " << this->logDBEntry().payloadClass
            << " in the db " << this->logDBEntry().destinationDB << " parameters: " << ss_logdb.str()
            << "\n NO TRANSFER NEEDED";
        return false;
      }

      this->m_userTextLog = ss.str();
      edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
          << "[isTransferNeeded] the selected conditions will be uploaded: " << ss.str()
          << "\n Going to Upload: " << ss.str() << "\n Last Upload: " << ss_logdb.str() << "\n Fine";

      return true;
    }

    //---------------------------------------
    //
    void setForTransfer() {
      edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
          << "[setForTransfer] " << m_name << " getting data to be transferred " << std::endl;

      T* obj = nullptr;
      condObjBuilder->getValue(obj);

      edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
          << "[setForTransfer] " << m_name << " got data to be transferred from condObjBuilder " << std::endl;

      if (!this->tagInfo().size)
        m_since = 1;
      else if (m_debugMode)
        m_since = this->tagInfo().lastInterval.since + 1;
      edm::LogInfo("SiStripPopPopConConfigDbObjHandler") << "[setForTransfer] setting since = " << m_since << std::endl;

      if (obj != nullptr) {
        edm::LogInfo("SiStripPopPopConConfigDbObjHandler") << "[setForTransfer] filling map m_to_transfer" << std::endl;
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
  };
}  // namespace popcon

#endif  //SISTRIPPOPCON_CONFIGDB_HANDLER_H
