#ifndef DQMOffline_CalibTracker_SiStripDQMPopConSourceHandler_H
#define DQMOffline_CalibTracker_SiStripDQMPopConSourceHandler_H

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
  @class SiStripDQMPopConSourceHandler
  @author M. De Mattia
  @author P. David (merge service functionality into base class)

  Base class for SiStrip popcon::PopConSourceHandler (reading from DQM) and writing in the Database.
*/
template<typename T>
class SiStripDQMPopConSourceHandler : public popcon::PopConSourceHandler<T>
{
public:
  explicit SiStripDQMPopConSourceHandler(const edm::ParameterSet& pset)
    : m_name{pset.getUntrackedParameter<std::string>("name", "SiStripPopConDbObjHandler")}
    , m_since{pset.getUntrackedParameter<uint32_t>("since", 5)}
    , m_runNumber{pset.getParameter<uint32_t>("RunNb")}
    , m_iovSequence{pset.getUntrackedParameter<bool>("iovSequence", true)} // flag: check compatibility
    , m_debugMode{pset.getUntrackedParameter<bool>("debug", false)}
  {}

  ~SiStripDQMPopConSourceHandler() override {}

  // popcon::PopConSourceHandler interface methods
  void getNewObjects() override;
  std::string id() const override { return m_name; }

  virtual T* getObj() const = 0;

  virtual std::string getMetaDataString() const;
  virtual bool checkForCompatibility( const std::string otherMetaData ) const { return otherMetaData != getMetaDataString(); }

  // additional methods needed for SiStripPopConDQMEDHarvester
  virtual void initES(const edm::EventSetup&) {}
  virtual void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {}

protected:
  uint32_t getRunNumber() const { return m_runNumber; }

private:
  std::string m_name;
  unsigned long long m_since;
  uint32_t m_runNumber;
  bool m_iovSequence;
  bool m_debugMode;

  // helper methods
  bool isTransferNeeded();
  void setForTransfer();
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

template<typename T>
void SiStripDQMPopConSourceHandler<T>::getNewObjects()
{
  edm::LogInfo("SiStripPopConDbObjHandler") << "[SiStripPopConDbObjHandler::getNewObjects] for PopCon application " << m_name;

  if ( m_debugMode ) {
    std::stringstream ss;
    ss << "\n\n------- " << m_name
       << " - > getNewObjects\n";
    if ( this->tagInfo().size ){
      //check whats already inside of database
      ss << "\ngot offlineInfo" << this->tagInfo().name
         << "\n size " << this->tagInfo().size
         << "\n" << this->tagInfo().token
         << "\n last object valid since " << this->tagInfo().lastInterval.first
         << "\n token "    << this->tagInfo().lastPayloadToken
         << "\n UserText " << this->userTextLog()
         << "\n LogDBEntry \n"
         << this->logDBEntry().logId         << "\n"
         << this->logDBEntry().destinationDB << "\n"
         << this->logDBEntry().provenance    << "\n"
         << this->logDBEntry().usertext      << "\n"
         << this->logDBEntry().iovtag        << "\n"
         << this->logDBEntry().iovtimetype   << "\n"
         << this->logDBEntry().payloadIdx    << "\n"
         << this->logDBEntry().payloadClass  << "\n"
         << this->logDBEntry().payloadToken  << "\n"
         << this->logDBEntry().exectime      << "\n"
         << this->logDBEntry().execmessage   << "\n";
      if ( this->logDBEntry().usertext != "" )
        ss<< "\n-- user text " << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@")) ;
    } else {
      ss << " First object for this tag ";
    }
    edm::LogInfo("SiStripPopConDbObjHandler") << ss.str();
  }

  if (isTransferNeeded())
    setForTransfer();

  edm::LogInfo("SiStripPopConDbObjHandler") << "[SiStripPopConDbObjHandler::getNewObjects] for PopCon application " << m_name << " Done\n--------------\n";
}

template<typename T>
bool SiStripDQMPopConSourceHandler<T>::isTransferNeeded()
{
  edm::LogInfo("SiStripPopConDbObjHandler") << "[SiStripPopConDbObjHandler::isTransferNeeded] checking for transfer ";

  if ( m_iovSequence && ( m_since <= this->tagInfo().lastInterval.first ) ) {
    edm::LogInfo   ("SiStripPopConDbObjHandler")
      << "[SiStripPopConDbObjHandler::isTransferNeeded] \nthe current starting iov " << m_since
      << "\nis not compatible with the last iov ("
      << this->tagInfo().lastInterval.first << ") open for the object "
      << this->logDBEntry().payloadClass << " \nin the db "
      << this->logDBEntry().destinationDB << " \n NO TRANSFER NEEDED";
    return false;
  }

  std::string ss_logdb{};

  //get log information from previous upload
  if ( this->logDBEntry().usertext != "" )
    ss_logdb = this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@")+2);

  std::string ss = getMetaDataString();
  if ( ( ! m_iovSequence ) || checkForCompatibility(ss_logdb) ) {
    this->m_userTextLog = "@ " + ss;

    edm::LogInfo   ("SiStripPopConDbObjHandler")
      << "[SiStripPopConDbObjHandler::isTransferNeeded] \nthe selected conditions will be uploaded: " << ss
      << "\n Current MetaData - " << ss << "\n Last Uploaded MetaData- " << ss_logdb << "\n Fine";

    return true;
  } else if ( m_iovSequence ) {
    edm::LogInfo   ("SiStripPopConDbObjHandler")
      << "[SiStripPopConDbObjHandler::isTransferNeeded] \nthe current MetaData conditions " << ss
      << "\nare not compatible with the MetaData Conditions of the last iov ("
      << this->tagInfo().lastInterval.first << ") open for the object "
      << this->logDBEntry().payloadClass << " \nin the db "
      << this->logDBEntry().destinationDB << " \nConditions: "  << ss_logdb << "\n NO TRANSFER NEEDED";
    return false;
  } else {
    return true;
  }
}

template<typename T>
void SiStripDQMPopConSourceHandler<T>::setForTransfer()
{
  edm::LogInfo   ("SiStripPopConDbObjHandler") << "[SiStripPopConDbObjHandler::setForTransfer] " << m_name << " getting data to be transferred ";

  if ( ! this->tagInfo().size )
    m_since=1;
  else
    if (m_debugMode)
      m_since = this->tagInfo().lastInterval.first+1;

  T* obj = this->getObj();
  if ( obj ) {
    edm::LogInfo   ("SiStripPopConDbObjHandler") <<"setting since = "<< m_since;
    this->m_to_transfer.push_back(std::make_pair(obj,m_since));
  } else {
    edm::LogError   ("SiStripPopConDbObjHandler") <<"[SiStripPopConDbObjHandler::setForTransfer] " << m_name << "  : NULL pointer of obj " << typeid(T).name() << " reported by SiStripCondObjBuilderFromDb\n Transfer aborted";
  }
}

template <class T>
std::string SiStripDQMPopConSourceHandler<T>::getMetaDataString() const
{
  std::cout << "SiStripPedestalsDQMService::getMetaDataString" << std::endl;
  std::stringstream ss;
  ss << "Run " << m_runNumber << std::endl;
  return ss.str();
}

#endif // DQMOffline_CalibTracker_SiStripDQMPopConSourceHandler_H
