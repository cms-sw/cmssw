// $Id: StateMachine.h,v 1.7 2009/07/14 10:34:44 dshpakov Exp $
/// @file: InitMsgCollection.h 

#ifndef StorageManager_InitMsgCollection_h
#define StorageManager_InitMsgCollection_h

#include "EventFilter/StorageManager/interface/ConsumerID.h"

#include "IOPool/Streamer/interface/InitMessage.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"
#include <vector>
#include <map>
#include <string>

namespace stor
{

  /**
     This class is used to manage the unique set of INIT messages
     that have been received by the storage manager and will be sent
     to event consumers and written to output streams.

     $Author: dshpakov $
     $Revision: 1.4 $
     $Date: 2009/07/14 10:34:44 $
  */

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

#endif // StorageManager_InitMsgCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
