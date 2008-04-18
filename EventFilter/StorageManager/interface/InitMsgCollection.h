#ifndef STOR_INITMSG_COLLECTION_H
#define STOR_INITMSG_COLLECTION_H

/**
 * This class is used to manage the set of INIT messages that have
 * been received by the storage manager and will be sent to event
 * consumers and written to output streams.
 *
 * $Id: InitMsgCollection.h,v 1.2 2008/02/11 15:07:37 biery Exp $
 */

#include "IOPool/Streamer/interface/InitMessage.h"
#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"
#include <vector>

namespace stor
{
  typedef std::vector<unsigned char> InitMsgBuffer;
  typedef boost::shared_ptr<InitMsgBuffer> InitMsgSharedPtr;

  class InitMsgCollection
  {

  public:

    InitMsgCollection();
    ~InitMsgCollection();

    bool testAndAddIfUnique(InitMsgView const& initMsgView);
    InitMsgSharedPtr getElementForSelection(Strings const& triggerSelection);
    InitMsgSharedPtr getLastElement();
    InitMsgSharedPtr getElementAt(unsigned int index);
    InitMsgSharedPtr getFullCollection() { return serializedFullSet_; }

    void clear();
    int size();

    std::string getSelectionHelpString();
    static std::string stringsToText(Strings const& list,
                                     unsigned int maxCount = 0);

  private:

    void add(InitMsgView const& initMsgView);

    std::vector<InitMsgSharedPtr> initMsgList_;
    InitMsgSharedPtr serializedFullSet_;

    boost::mutex listLock_;

  };
}

#endif
