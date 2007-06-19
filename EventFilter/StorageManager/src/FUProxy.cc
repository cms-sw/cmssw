// Created by Markus Klute on 2007 Mar 22.
// $Id: FUProxy.cc,v 1.2 2007/06/06 21:41:08 meschi Exp $
////////////////////////////////////////////////////////////////////////////////

#include <EventFilter/StorageManager/interface/FUProxy.h>

#include <toolbox/include/toolbox/mem/Reference.h>
#include <toolbox/include/toolbox/mem/MemoryPoolFactory.h>
#include <toolbox/include/toolbox/mem/exception/Exception.h>

#include <i2o/include/i2o/Method.h>
#include <i2o/utils/include/i2o/utils/AddressMap.h>

#include <xcept/include/xcept/tools.h>

#include <iostream> 
#include <sstream>

using stor::FUProxy;


////////////////////////////////////////////////////////////////////////////////
FUProxy::FUProxy(xdaq::ApplicationDescriptor *smAppDesc,
		 xdaq::ApplicationDescriptor *fuAppDesc,
		 xdaq::ApplicationContext    *smAppContext,
		 toolbox::mem::Pool          *i2oPool) 
  : smAppDesc_(smAppDesc)
  , fuAppDesc_(fuAppDesc)
  , smAppContext_(smAppContext)
  , i2oPool_(i2oPool)
{

}


////////////////////////////////////////////////////////////////////////////////
FUProxy::~FUProxy()
{

}


////////////////////////////////////////////////////////////////////////////////
int FUProxy::sendDQMDiscard(int fuId)
{
  return sendDiscard(fuId, I2O_FU_DQM_DISCARD);
}


////////////////////////////////////////////////////////////////////////////////
int FUProxy::sendDataDiscard(int fuId)
{
  return sendDiscard(fuId, I2O_FU_DATA_DISCARD);
}


////////////////////////////////////////////////////////////////////////////////
int FUProxy::sendDiscard(int fuId, int msgId)
  throw (evf::Exception)
{
  try {
    size_t msgSize=sizeof(I2O_FU_DATA_DISCARD_MESSAGE_FRAME);
    
    toolbox::mem::Reference *bufRef=
      toolbox::mem::getMemoryPoolFactory()->getFrame(i2oPool_,msgSize);

    // message pointer
    I2O_MESSAGE_FRAME *stdMsg = 
      (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
    I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
      (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
    I2O_FU_DATA_DISCARD_MESSAGE_FRAME *msg =
      (I2O_FU_DATA_DISCARD_MESSAGE_FRAME*)stdMsg;
    
    // set message frame
    stdMsg->MessageSize     = msgSize >> 2;
    stdMsg->InitiatorAddress= i2o::utils::getAddressMap()->getTid(smAppDesc_);
    stdMsg->TargetAddress   = i2o::utils::getAddressMap()->getTid(fuAppDesc_);
    stdMsg->Function        = I2O_PRIVATE_MESSAGE;
    stdMsg->VersionOffset   = 0;
    stdMsg->MsgFlags        = 0;  
 
    // set private message frame
    pvtMsg->XFunctionCode   =  msgId;
    pvtMsg->OrganizationID  = XDAQ_ORGANIZATION_ID;

    // set fu data discard message frame
    msg->fuID               = fuId;

     // set buffer size
    bufRef->setDataSize(msgSize);

    // post frame
    smAppContext_->postFrame(bufRef,smAppDesc_,fuAppDesc_);
  }
  catch (xdaq::exception::Exception &e) {
    std::ostringstream oss;
    oss << "FUProxy failed to post message! " 
	<< " What: "              << e.what()
	<< " Exception history: " << xcept::stdformat_exception_history(e);
    LOG4CPLUS_ERROR(smAppContext_->getLogger(), oss.str());
    XCEPT_RETHROW(evf::Exception, oss.str(), e);
  }
  catch(toolbox::mem::exception::Exception& e)	{
    std::ostringstream oss;
    oss << "FUProxy failed to allocate buffer reference! " 
	<< " What: "              << e.what()
	<< " Exception history: " << xcept::stdformat_exception_history(e);
    LOG4CPLUS_ERROR(smAppContext_->getLogger(), oss.str());
    XCEPT_RETHROW(evf::Exception, oss.str(), e);
  }
  return 0;
}


