// $Id: DiscardManager.cc,v 1.3 2009/06/24 19:11:21 biery Exp $
/// @file: DiscardManager.cc

#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

#include "toolbox/mem/MemoryPoolFactory.h"

using namespace stor;

DiscardManager::DiscardManager(xdaq::ApplicationContext* ctx,
                               xdaq::ApplicationDescriptor* desc,
                               DataSenderMonitorCollection& dsmc):
  _appContext(ctx),
  _appDescriptor(desc),
  _dataSenderMonCollection(dsmc)
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
          _pool = thePool;
          _proxyCache.clear();
        }
    }
  }
}

bool DiscardManager::sendDiscardMessage(I2OChain const& i2oMessage)
{
  if (i2oMessage.messageCode() == Header::INVALID)
    {
      _dataSenderMonCollection.incrementSkippedDiscardCount(i2oMessage);
      return false;
    }

  unsigned int rbBufferId = i2oMessage.rbBufferId();
  std::string hltClassName = i2oMessage.hltClassName();
  unsigned int hltInstance = i2oMessage.hltInstance();
  FUProxyPtr fuProxyPtr = getProxyFromCache(hltClassName, hltInstance);
  if (fuProxyPtr.get() == 0)
    {
      _dataSenderMonCollection.incrementSkippedDiscardCount(i2oMessage);
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
          _dataSenderMonCollection.incrementDQMDiscardCount(i2oMessage);
        }
      else
        {
          fuProxyPtr->sendDataDiscard(rbBufferId);	
          _dataSenderMonCollection.incrementDataDiscardCount(i2oMessage);
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
  cacheIter = _proxyCache.find(mapKey);

  if (cacheIter != _proxyCache.end())
    {
      return cacheIter->second;
    }
  else
    {
      FUProxyPtr fuProxyPtr = makeNewFUProxy(hltClassName, hltInstance);
      if (fuProxyPtr.get() != 0)
        {
          _proxyCache[mapKey] = fuProxyPtr;
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
    _appContext->getDefaultZone()->
    getApplicationDescriptors(hltClassName.c_str());

  std::set<xdaq::ApplicationDescriptor*>::iterator iter;
  std::set<xdaq::ApplicationDescriptor*>::iterator iterEnd = setOfRBs.end();

  for (iter = setOfRBs.begin(); iter != iterEnd; ++iter)
    {
      if ((*iter)->getInstance() == hltInstance)
	{
          proxyPtr.reset(new stor::FUProxy(_appDescriptor, *iter,
                                           _appContext, _pool));
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
