#ifndef STOR_INITMSG_COLLECTION_H
#define STOR_INITMSG_COLLECTION_H

/**
 * This class is used to manage the set of INIT messages that have
 * been received by the storage manager and will be sent to event
 * consumers and written to output streams.
 *
 * $Id$
 */

#include "EventFilter/StorageManager/interface/ConsumerID.h"

#include "IOPool/Streamer/interface/InitMessage.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"
#include <vector>
#include <map>
#include <string>

namespace stor
{
  typedef std::vector<unsigned char> InitMsgBuffer;
  typedef boost::shared_ptr<InitMsgBuffer> InitMsgSharedPtr;

  class InitMsgCollection
  {

  public:

    InitMsgCollection();
    ~InitMsgCollection();

    bool addIfUnique(InitMsgView const& initMsgView);
    InitMsgSharedPtr getElementForOutputModule(std::string requestedOMLabel);
    InitMsgSharedPtr getLastElement();
    InitMsgSharedPtr getElementAt(unsigned int index);
    InitMsgSharedPtr getFullCollection() { return serializedFullSet_; }

    bool registerConsumer( ConsumerID cid, const std::string& hltModule );
    InitMsgSharedPtr getElementForConsumer( ConsumerID cid );

    void clear();
    int size();

    std::string getSelectionHelpString();
    std::string getOutputModuleName(uint32 outputModuleId);
    static std::string stringsToText(Strings const& list,
                                     unsigned int maxCount = 0);

  private:

    void add(InitMsgView const& initMsgView);

    std::vector<InitMsgSharedPtr> initMsgList_;
    InitMsgSharedPtr serializedFullSet_;

    std::map<uint32, std::string> outModNameTable_;
    boost::mutex listLock_;

    std::map<ConsumerID, std::string> consumerOutputModuleMap_;
    boost::mutex consumerMapLock_;
  };
}

#endif
