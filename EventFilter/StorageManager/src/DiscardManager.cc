// $Id: DiscardManager.cc,v 1.5.10.1 2011/02/28 17:56:06 mommsen Exp $
/// @file: DiscardManager.cc

#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FUProxy.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

#include "IOPool/Streamer/interface/MsgHeader.h"

#include "toolbox/mem/MemoryPoolFactory.h"

using namespace stor;

DiscardManager::DiscardManager(xdaq::ApplicationContext* ctx,
                               xdaq::ApplicationDescriptor* desc,
                               DataSenderMonitorCollection& dsmc):
  appContext_(ctx),
  appDescriptor_(desc),
  dataSenderMonCollection_(dsmc)
{
}

void DiscardManager::configure()
{
  Strings nameList = toolbox::mem::getMemoryPoolFactory()->getMemoryPoolNames();
  for (unsigned int idx = 0; idx < nameList.size(); ++idx) {
    //std::cout << "POOL NAME2 = " << nameList[idx] << std::endl;
    if (idx == 0 || nameList[idx].find("TCP") != std::string::npos) {
      toolbox::net::URN poolURN(nameList[idx]);
      toolbox::mem::Pool* thePool =
        toolbox::mem::getMemoryPoolFactory()->findPool(poolURN);
      if (thePool != 0) 
        {
          pool_ = thePool;
          proxyCache_.clear();
        }
    }
  }
}

bool DiscardManager::sendDiscardMessage(I2OChain const& i2oMessage)
{
  if (i2oMessage.messageCode() == Header::INVALID)
    {
      dataSenderMonCollection_.incrementSkippedDiscardCount(i2oMessage);
      return false;
    }

  unsigned int rbBufferId = i2oMessage.rbBufferId();
  std::string hltClassName = i2oMessage.hltClassName();
  unsigned int hltInstance = i2oMessage.hltInstance();
  FUProxyPtr fuProxyPtr = getProxyFromCache(hltClassName, hltInstance);
  if (fuProxyPtr.get() == 0)
    {
      dataSenderMonCollection_.incrementSkippedDiscardCount(i2oMessage);
      std::stringstream msg;
      msg << "Unable to find the resource broker corresponding to ";
      msg << "classname = \"";
      msg << hltClassName;
      msg << "\" and instance = \"";
      msg << hltInstance;
      msg << "\".";
      XCEPT_RAISE(stor::exception::RBLookupFailed, msg.str());
    }
  else
    {
      if (i2oMessage.messageCode() == Header::DQM_EVENT)
        {
          fuProxyPtr->sendDQMDiscard(rbBufferId);
          dataSenderMonCollection_.incrementDQMDiscardCount(i2oMessage);
        }
      else
        {
          fuProxyPtr->sendDataDiscard(rbBufferId);	
          dataSenderMonCollection_.incrementDataDiscardCount(i2oMessage);
        }
    }

  return true;
}

DiscardManager::FUProxyPtr
DiscardManager::getProxyFromCache(std::string hltClassName,
                                  unsigned int hltInstance)
{
  HLTSenderKey mapKey = std::make_pair(hltClassName, hltInstance);
  FUProxyMap::const_iterator cacheIter;
  cacheIter = proxyCache_.find(mapKey);

  if (cacheIter != proxyCache_.end())
    {
      return cacheIter->second;
    }
  else
    {
      FUProxyPtr fuProxyPtr = makeNewFUProxy(hltClassName, hltInstance);
      if (fuProxyPtr.get() != 0)
        {
          proxyCache_[mapKey] = fuProxyPtr;
        }
      return fuProxyPtr;
    }
}

DiscardManager::FUProxyPtr
DiscardManager::makeNewFUProxy(std::string hltClassName,
                               unsigned int hltInstance)
{
  FUProxyPtr proxyPtr;
  std::set<xdaq::ApplicationDescriptor*> setOfRBs=
    appContext_->getDefaultZone()->
    getApplicationDescriptors(hltClassName.c_str());

  std::set<xdaq::ApplicationDescriptor*>::iterator iter;
  std::set<xdaq::ApplicationDescriptor*>::iterator iterEnd = setOfRBs.end();

  for (iter = setOfRBs.begin(); iter != iterEnd; ++iter)
    {
      if ((*iter)->getInstance() == hltInstance)
	{
          proxyPtr.reset(new stor::FUProxy(appDescriptor_, *iter,
                                           appContext_, pool_));
          break;
	}
    }

  return proxyPtr;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
