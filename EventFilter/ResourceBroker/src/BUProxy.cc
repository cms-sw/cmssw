////////////////////////////////////////////////////////////////////////////////
//
// BUProxy
// -------
//
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/BUProxy.h"

#include "xdaq/include/xdaq/Application.h"

#include "toolbox/include/toolbox/mem/Reference.h"
#include "toolbox/include/toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/include/toolbox/mem/exception/Exception.h"

#include "i2o/include/i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"

#include "xcept/include/xcept/tools.h"


#include "interface/evb/include/i2oEVBMsgs.h" 
#include "interface/shared/include/i2oXFunctionCodes.h"

#include <iostream>


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
BUProxy::BUProxy(xdaq::ApplicationDescriptor *fuAppDesc,
		 xdaq::ApplicationDescriptor *buAppDesc,
		 xdaq::ApplicationContext    *appContext,
		 toolbox::mem::Pool          *i2oPool,
		 UInt_t                       dataBufSize)
  : fuAppDesc_(fuAppDesc)
  , buAppDesc_(buAppDesc)
  , appContext_(appContext)
  , i2oPool_(i2oPool)
  , dataBufSize_(dataBufSize)
{
  if(i2oPool_==0) cout<<"BUProxy ctor ERROR: no memory pool!"<<endl;
}


//______________________________________________________________________________
BUProxy::~BUProxy()
{

}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void BUProxy::sendAllocate(const UIntVec_t& fuResourceIds)
{
  Logger log=appContext_->getLogger();

  try {
    size_t msgSize=
      sizeof(I2O_BU_ALLOCATE_MESSAGE_FRAME)+
      sizeof(BU_ALLOCATE)*(fuResourceIds.size()-1);
    
    toolbox::mem::Reference *bufRef=
      toolbox::mem::getMemoryPoolFactory()->getFrame(i2oPool_,msgSize);
  
    I2O_MESSAGE_FRAME             *stdMsg;
    I2O_PRIVATE_MESSAGE_FRAME     *pvtMsg;
    I2O_BU_ALLOCATE_MESSAGE_FRAME *msg;
    
    stdMsg=(I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
    pvtMsg=(I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
    msg   =(I2O_BU_ALLOCATE_MESSAGE_FRAME*)stdMsg; 
    
    stdMsg->MessageSize     =msgSize >> 2;
    stdMsg->InitiatorAddress=i2o::utils::getAddressMap()->getTid(fuAppDesc_);
    stdMsg->TargetAddress   =i2o::utils::getAddressMap()->getTid(buAppDesc_);
    stdMsg->Function        =I2O_PRIVATE_MESSAGE;
    stdMsg->VersionOffset   =0;
    stdMsg->MsgFlags        =0;  // Point-to-point
    
    pvtMsg->XFunctionCode   =I2O_BU_ALLOCATE;
    pvtMsg->OrganizationID  =XDAQ_ORGANIZATION_ID;
    
    msg->n                  =fuResourceIds.size();
    
    for(UInt_t i=0;i<fuResourceIds.size();i++) {
      msg->allocate[i].fuTransactionId=fuResourceIds[i];
      msg->allocate[i].fset           =1; // IGNORED!!!
    }

    bufRef->setDataSize(msgSize);

    appContext_->postFrame(bufRef,fuAppDesc_,buAppDesc_);
  }
  catch(toolbox::mem::exception::Exception e)	{
    LOG4CPLUS_ERROR(log,"Error allocating buffer reference: "
		    <<xcept::stdformat_exception_history(e));
  }
  catch(xdaq::exception::ApplicationDescriptorNotFound e) {
    LOG4CPLUS_ERROR(log,"Error getting source tid: " 
		   <<xcept::stdformat_exception_history(e));
  }
  catch(xdaq::exception::Exception &e) {
    LOG4CPLUS_ERROR(log,"Error posting 'sendAllocate' message frame: "<<e.what());
  }

  return;
}


//______________________________________________________________________________
void BUProxy::sendCollect(UInt_t /* fuResourceId */)
{
  LOG4CPLUS_INFO(appContext_->getLogger(),"BUProxy::sendCollect");
}


//______________________________________________________________________________
void BUProxy::sendDiscard(UInt_t buResourceId)
{
  Logger log=appContext_->getLogger();
  
  try {
    size_t msgSize=sizeof(I2O_BU_DISCARD_MESSAGE_FRAME);
    
    toolbox::mem::Reference *bufRef=
      toolbox::mem::getMemoryPoolFactory()->getFrame(i2oPool_,msgSize);
    
    I2O_MESSAGE_FRAME         *stdMsg=(I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
    I2O_PRIVATE_MESSAGE_FRAME *pvtMsg=(I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
    I2O_BU_DISCARD_MESSAGE_FRAME *msg=(I2O_BU_DISCARD_MESSAGE_FRAME*)stdMsg;
    
    stdMsg->MessageSize     =msgSize >> 2;
    stdMsg->InitiatorAddress=i2o::utils::getAddressMap()->getTid(fuAppDesc_);
    stdMsg->TargetAddress   =i2o::utils::getAddressMap()->getTid(buAppDesc_);
    stdMsg->Function        =I2O_PRIVATE_MESSAGE;
    stdMsg->VersionOffset   =0;
    stdMsg->MsgFlags        =0;  // Point-to-point
    
    pvtMsg->XFunctionCode   =I2O_BU_DISCARD;
    pvtMsg->OrganizationID  =XDAQ_ORGANIZATION_ID;
    
    msg->n                  =1;
    msg->buResourceId[0]    =buResourceId;
    
    bufRef->setDataSize(msgSize);

    appContext_->postFrame(bufRef,fuAppDesc_,buAppDesc_);
  }
  catch(toolbox::mem::exception::Exception e)	{
    LOG4CPLUS_ERROR(log,"Error allocating buffer reference: "
		    <<xcept::stdformat_exception_history(e));
  }
  catch(xdaq::exception::ApplicationDescriptorNotFound e) {
    LOG4CPLUS_ERROR(log,"Error getting source tid: " 
		   <<xcept::stdformat_exception_history(e));
  }
  catch(xdaq::exception::Exception &e) {
    LOG4CPLUS_ERROR(log,"Error posting 'discard' message frame: "<<e.what());
  }

  return;
}
  
