// $Id: InitMsgCollection.cc,v 1.17 2012/04/20 10:48:02 mommsen Exp $
/// @file: InitMsgCollection.cc

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/DumpTools.h"
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include "boost/algorithm/string/trim.hpp"
#include <iostream>

using namespace stor;
using namespace edm;

InitMsgCollection::InitMsgCollection()
{
  clear();
}


InitMsgCollection::~InitMsgCollection()
{
}


bool InitMsgCollection::addIfUnique(InitMsgView const& initMsgView)
{
  boost::mutex::scoped_lock sl(listLock_);
  
  // check if the outputModuleId was already seen
  const uint32_t outputModuleId = initMsgView.outputModuleId();
  InitMsgMap::iterator pos = initMsgMap_.lower_bound(outputModuleId);
  if ( pos != initMsgMap_.end() && !(initMsgMap_.key_comp()(outputModuleId, pos->first)))
    return false; // init message already exists
  
  checkOutputModuleLabel(initMsgView);
  
  // add the message to the internal list
  InitMsgSharedPtr serializedProds(new InitMsgBuffer(initMsgView.size()));
  std::copy(
    initMsgView.startAddress(),
    initMsgView.startAddress()+initMsgView.size(),
    &(*serializedProds)[0]
  );
  initMsgMap_.insert(pos, InitMsgMap::value_type(outputModuleId,serializedProds));
  
  return true; // new init message
}


bool InitMsgCollection::addIfUnique(I2OChain const& i2oChain, InitMsgSharedPtr& serializedProds)
{
  boost::mutex::scoped_lock sl(listLock_);

  // check if the outputModuleId was already seen
  const uint32_t outputModuleId = i2oChain.outputModuleId();
  InitMsgMap::iterator pos = initMsgMap_.lower_bound(outputModuleId);
  if ( pos != initMsgMap_.end() && !(initMsgMap_.key_comp()(outputModuleId, pos->first)))
    return false; // init message already exists
  
  // build the init message view
  serializedProds.reset( new InitMsgBuffer(i2oChain.totalDataSize()) );
  i2oChain.copyFragmentsIntoBuffer( *serializedProds );
  InitMsgView initMsgView(&(*serializedProds)[0]);
  
  checkOutputModuleLabel(initMsgView);
  
  // add the message to the internal list
  initMsgMap_.insert(pos, InitMsgMap::value_type(outputModuleId,serializedProds));

  return true; // new init message
}


void InitMsgCollection::checkOutputModuleLabel(InitMsgView const& initMsgView) const
{
  const std::string inputOMLabel = initMsgView.outputModuleLabel();
  const std::string trimmedOMLabel = boost::algorithm::trim_copy(inputOMLabel);
  if (trimmedOMLabel.empty()) {
    throw cms::Exception("InitMsgCollection", "addIfUnique:")
      << "Invalid INIT message: the HLT output module label is empty!"
        << std::endl;
  }
}


InitMsgSharedPtr
InitMsgCollection::getElementForOutputModuleId(const uint32_t& requestedOutputModuleId) const
{
  boost::mutex::scoped_lock sl(listLock_);
  InitMsgSharedPtr serializedProds;
  
  InitMsgMap::const_iterator it = initMsgMap_.find(requestedOutputModuleId);
  if (it != initMsgMap_.end())
    serializedProds = it->second;

  return serializedProds;
}


InitMsgSharedPtr
InitMsgCollection::getElementForOutputModuleLabel(const std::string& requestedOutputModuleLabel) const
{
  boost::mutex::scoped_lock sl(listLock_);
  InitMsgSharedPtr serializedProds;

  // handle the special case of an empty request
  // (If we want to use class methods to check the collection size and
  // fetch the last element in the collection, then we would need to 
  // switch to recursive mutexes...)
  if (requestedOutputModuleLabel.empty()) {
    if (initMsgMap_.size() == 1) {
      serializedProds = initMsgMap_.begin()->second;
    }
    else if (initMsgMap_.size() > 1) {
      std::string msg = "Invalid INIT message lookup: the requested ";
      msg.append("HLT output module label is empty but there are multiple ");
      msg.append("HLT output modules to choose from.");
      throw cms::Exception("InitMsgCollection", "getElementForOutputModule:")
        << msg << std::endl;
    }
  }

  else {
    // loop over the existing messages
    for (InitMsgMap::const_iterator msgIter = initMsgMap_.begin(),
           msgIterEnd = initMsgMap_.end();
         msgIter != msgIterEnd;
         msgIter++)
    {
      InitMsgSharedPtr workingMessage = msgIter->second;
      InitMsgView existingInitMsg(&(*workingMessage)[0]);
      std::string existingOMLabel = existingInitMsg.outputModuleLabel();
      
      // check if the output module labels match
      if (requestedOutputModuleLabel == existingOMLabel) {
        serializedProds = workingMessage;
        break;
      }
    }
  }

  return serializedProds;
}


InitMsgSharedPtr InitMsgCollection::getElementAt(const unsigned int index) const
{
  boost::mutex::scoped_lock sl(listLock_);

  InitMsgSharedPtr ptrToElement;
  try
  {
    ptrToElement = initMsgMap_.at(index);
  }
  catch (std::out_of_range& e)
  { }

  return ptrToElement;
}


void InitMsgCollection::clear()
{
  boost::mutex::scoped_lock sl(listLock_);
  initMsgMap_.clear();
}


size_t InitMsgCollection::size() const
{
  boost::mutex::scoped_lock sl(listLock_);
  return initMsgMap_.size();
}


std::string InitMsgCollection::getSelectionHelpString() const
{
  boost::mutex::scoped_lock sl(listLock_);

  // nothing much we can say if the collection is empty
  if (initMsgMap_.empty()) {
    return "No information is available about the available triggers.";
  }

  // list the full set of available triggers
  std::string helpString;
  helpString.append("The full list of trigger paths is the following:");

  // we can just use the list from the first entry since all
  // subsequent entries are forced to be the same
  InitMsgSharedPtr serializedProds = initMsgMap_.begin()->second;
  InitMsgView existingInitMsg(&(*serializedProds)[0]);
  Strings existingTriggerList;
  existingInitMsg.hltTriggerNames(existingTriggerList);
  for (unsigned int idx = 0; idx < existingTriggerList.size(); idx++) {
    helpString.append("\n    " + existingTriggerList[idx]);
  }

  // list the output modules (INIT messages)
  helpString.append("\nThe registered HLT output modules and their ");
  helpString.append("trigger selections are the following:");

  // loop over the existing messages
    for (InitMsgMap::const_iterator msgIter = initMsgMap_.begin(),
           msgIterEnd = initMsgMap_.end();
         msgIter != msgIterEnd;
         msgIter++)
    {
    serializedProds = msgIter->second;
    InitMsgView workingInitMsg(&(*serializedProds)[0]);
    helpString.append("\n  *** Output module \"");
    helpString.append(workingInitMsg.outputModuleLabel());
    helpString.append("\" ***");
    Strings workingSelectionList;
    workingInitMsg.hltTriggerSelections(workingSelectionList);
    for (unsigned int idx = 0; idx < workingSelectionList.size(); idx++) {
      helpString.append("\n    " + workingSelectionList[idx]);
    }
  }

  // return the result
  return helpString;
}


std::string InitMsgCollection::getOutputModuleName(const uint32_t outputModuleId) const
{
  boost::mutex::scoped_lock sl(listLock_);

  InitMsgMap::const_iterator it = initMsgMap_.find(outputModuleId);

  if (it == initMsgMap_.end())
  {
    return "";
  }
  else {
    InitMsgSharedPtr serializedProds = it->second;
    const InitMsgView initMsgView(&(*serializedProds)[0]);
    return initMsgView.outputModuleLabel();
  }
}


std::string InitMsgCollection::stringsToText(Strings const& list,
                                             unsigned int maxCount)
{
  std::string resultString = "";
  unsigned int elementCount = list.size();
  if (maxCount > 0 && maxCount < elementCount) {elementCount = maxCount;}
  for (unsigned int idx = 0; idx < elementCount; idx++)
  {
    resultString.append(list[idx]);
    if (idx < (elementCount-1)) {
      resultString.append(", ");
    }
  }
  if (elementCount < list.size())
  {
    resultString.append(", ...");
  }
  return resultString;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
