// $Id: DiscardManager.cc,v 1.7 2011/04/07 08:01:40 mommsen Exp $
/// @file: DiscardManager.cc

#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FUProxy.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

#include "IOPool/Streamer/interface/MsgHeader.h"

#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/net/URN.h"


namespace stor {
  
  DiscardManager::DiscardManager
  (
    xdaq::ApplicationContext* ctx,
    xdaq::ApplicationDescriptor* desc,
    DataSenderMonitorCollection& dsmc
  ):
  appContext_(ctx),
  appDescriptor_(desc),
  dataSenderMonCollection_(dsmc)
  {
    std::ostringstream poolName;
    poolName << desc->getClassName() << desc->getInstance();
    toolbox::net::URN urn("toolbox-mem-pool", poolName.str());
    toolbox::mem::HeapAllocator* a = new toolbox::mem::HeapAllocator();
    
    msgPool_ = toolbox::mem::getMemoryPoolFactory()->createPool(urn, a);
  }
  
  void DiscardManager::configure()
  {
    proxyCache_.clear();
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
      XCEPT_RAISE(exception::RBLookupFailed, msg.str());
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
  DiscardManager::getProxyFromCache
  (
    std::string const& hltClassName,
    unsigned int const& hltInstance
  )
  {
    HLTSenderKey mapKey = std::make_pair(hltClassName, hltInstance);
    FUProxyMap::iterator pos = proxyCache_.lower_bound(mapKey);
    
    if (pos == proxyCache_.end() || (proxyCache_.key_comp()(mapKey, pos->first)))
    {
      // Use pos as a hint to insert a new record, so it can avoid another lookup
      FUProxyPtr fuProxyPtr = makeNewFUProxy(hltClassName, hltInstance);
      if (fuProxyPtr.get() != 0)
        pos = proxyCache_.insert(pos, FUProxyMap::value_type(mapKey, fuProxyPtr));

      return fuProxyPtr;
    }
    else
    {
      return pos->second;
    }
  }
  
  DiscardManager::FUProxyPtr
  DiscardManager::makeNewFUProxy
  (
    std::string const& hltClassName,
    unsigned int const& hltInstance
  )
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
        proxyPtr.reset(new FUProxy(appDescriptor_, *iter,
            appContext_, msgPool_));
        break;
      }
    }
    
    return proxyPtr;
  }
  
} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
