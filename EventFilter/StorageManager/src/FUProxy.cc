// $Id: FUProxy.cc,v 1.7 2011/04/07 08:01:40 mommsen Exp $
/// @file: FUProxy.cc

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FUProxy.h"
#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include <toolbox/mem/Reference.h>
#include <toolbox/mem/MemoryPoolFactory.h>
#include <toolbox/mem/exception/Exception.h>

#include <i2o/Method.h>
#include <i2o/utils/AddressMap.h>

#include <xcept/tools.h>

#include <iostream> 
#include <sstream>

namespace stor {
  
  FUProxy::FUProxy
  (
    xdaq::ApplicationDescriptor *smAppDesc,
    xdaq::ApplicationDescriptor *fuAppDesc,
    xdaq::ApplicationContext    *smAppContext,
    toolbox::mem::Pool          *msgPool
  ) 
  : smAppDesc_(smAppDesc),
    fuAppDesc_(fuAppDesc),
    smAppContext_(smAppContext),
    msgPool_(msgPool)
  {}
  
  void FUProxy::sendDQMDiscard(const int& rbBufferId)
  {
    sendDiscardMsg(rbBufferId, I2O_FU_DQM_DISCARD, sizeof(I2O_FU_DQM_DISCARD_MESSAGE_FRAME));
  }
  
  void FUProxy::sendDataDiscard(const int& rbBufferId)
  {
    sendDiscardMsg(rbBufferId, I2O_FU_DATA_DISCARD, sizeof(I2O_FU_DATA_DISCARD_MESSAGE_FRAME));
  }
  
  void FUProxy::sendDiscardMsg
  (
    const int& rbBufferId,
    const int& msgId,
    const size_t& msgSize
  )
  {
    try {
      toolbox::mem::Reference* bufRef=
        toolbox::mem::getMemoryPoolFactory()->getFrame(msgPool_,msgSize);
      
      // message pointer
      I2O_MESSAGE_FRAME* stdMsg = 
        (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
      I2O_PRIVATE_MESSAGE_FRAME* pvtMsg =
        (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
      I2O_FU_DATA_DISCARD_MESSAGE_FRAME* msg =
        (I2O_FU_DATA_DISCARD_MESSAGE_FRAME*)stdMsg;
      
      // set message frame
      stdMsg->MessageSize     = msgSize >> 2;
      stdMsg->InitiatorAddress= i2o::utils::getAddressMap()->getTid(smAppDesc_);
      stdMsg->TargetAddress   = i2o::utils::getAddressMap()->getTid(fuAppDesc_);
      stdMsg->Function        = I2O_PRIVATE_MESSAGE;
      stdMsg->VersionOffset   = 0;
      stdMsg->MsgFlags        = 0;  
      
      // set private message frame
      pvtMsg->XFunctionCode   = msgId;
      pvtMsg->OrganizationID  = XDAQ_ORGANIZATION_ID;
      
      // set fu data discard message frame
      msg->rbBufferID         = rbBufferId;
      
      // set buffer size
      bufRef->setDataSize(msgSize);
      
      // post frame
      smAppContext_->postFrame(bufRef,smAppDesc_,fuAppDesc_);
    }
    catch (toolbox::mem::exception::Exception& e)
    {
      XCEPT_RETHROW(exception::IgnoredDiscard,
        "FUProxy failed to allocate buffer reference!", e);
    }
    catch (xcept::Exception &e)
    {
      XCEPT_RETHROW(exception::IgnoredDiscard,
        "FUProxy failed to post discard message!", e);
    }
  }
  
} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -


