#ifndef SISTRIPPOPCON_UNITTEST_HANDLER_H
#define SISTRIPPOPCON_UNITTEST_HANDLER_H

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
#include <time.h>

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

namespace popcon {

  template <typename T>
  class SiStripPopConHandlerUnitTest : public popcon::PopConSourceHandler<T> {
  public:
    enum DataType { UNDEFINED = 0, _Cabling = 1, _Pedestal = 2, _Noise = 3, _Threshold = 4, _BadStrip = 5 };

    //---------------------------------------
    //
    SiStripPopConHandlerUnitTest(const edm::ParameterSet& pset)
        : m_name(pset.getUntrackedParameter<std::string>("name", "SiStripPopPopConConfigDbObjHandler")),
          m_since(pset.getUntrackedParameter<uint32_t>("since", 5)),
          m_debugMode(pset.getUntrackedParameter<bool>("debug", false)){};

    //---------------------------------------
    //
    ~SiStripPopConHandlerUnitTest(){};

    //---------------------------------------
    //
    void getNewObjects() {
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
             << this->logDBEntry().payloadName << "\n"
             << this->logDBEntry().payloadToken << "\n"
             << this->logDBEntry().payloadContainer << "\n"
             << this->logDBEntry().exectime << "\n"
             << this->logDBEntry().execmessage << "\n"
             << "\n\n-- user text "
             << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@"));
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
    std::string id() const { return m_name; }

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
        ss_logdb << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@"));
      else
        ss_logdb << "";

      ss << "@ " << clock();

      if (!strcmp(ss.str().c_str(), ss_logdb.str().c_str())) {
        //string are equal, no need to do transfer
        edm::LogInfo("SiStripPopPopConConfigDbObjHandler")
            << "[isTransferNeeded] the selected conditions are already uploaded in the last iov ("
            << this->tagInfo().lastInterval.since << ") open for the object " << this->logDBEntry().payloadName
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

      T* obj = 0;

      fillObject(obj);

      if (!this->tagInfo().size)
        m_since = 1;
      else if (m_debugMode)
        m_since = this->tagInfo().lastInterval.since + 1;

      if (obj != 0) {
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
      if (typeid(T) == typeid(SiStripNoises)) {
        obj = new SiStripNoises();

        const auto detInfo =
            SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
        int count = -1;
        for (const auto& it : detInfo.getAllData()) {
          count++;
          //Generate Noise for det detid
          SiStripNoises::InputVector theSiStripVector;
          for (int strip = 0; strip < 128 * it.second.nApvs; ++strip) {
            float MeanNoise = 5;
            float RmsNoise = 1;

            float noise = CLHEP::RandGauss::shoot(MeanNoise, RmsNoise);

            //double badStripProb = .5;
            //bool disable = (CLHEP::RandFlat::shoot(1.) < badStripProb ? true:false);

            obj->setData(noise, theSiStripVector);
            if (count < 6)
              edm::LogInfo("SiStripNoisesBuilder") << "detid " << it.first << " \t"
                                                   << " strip " << strip << " \t" << noise << " \t"
                                                   << theSiStripVector.back() / 10 << " \t" << std::endl;
          }

          if (!obj->put(it.first, theSiStripVector))
            edm::LogError("SiStripNoisesBuilder")
                << "[SiStripNoisesBuilder::analyze] detid already exists" << std::endl;
        }
      }
    }
  };

}  // namespace popcon

#endif  //SISTRIPPOPCON_UNITTEST_HANDLER_H
