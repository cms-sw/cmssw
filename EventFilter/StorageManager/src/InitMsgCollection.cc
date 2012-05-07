// $Id: InitMsgCollection.cc,v 1.15 2011/03/07 15:31:32 mommsen Exp $
/// @file: InitMsgCollection.cc

#include "DataFormats/Streamer/interface/StreamedProducts.h"
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

  // test the output module label for validity
  std::string inputOMLabel = initMsgView.outputModuleLabel();
  std::string trimmedOMLabel = boost::algorithm::trim_copy(inputOMLabel);
  if (trimmedOMLabel.empty()) {
    throw cms::Exception("InitMsgCollection", "addIfUnique:")
      << "Invalid INIT message: the HLT output module label is empty!"
      << std::endl;
  }

  // initially, assume that we will want to include the new message
  bool addToList = true;

  // if this is the first INIT message that we've seen, we just add it
  if (initMsgList_.empty()) {
    this->add(initMsgView);
  }

  // if this is a subsequent INIT message, check if it is unique
  else {

    // loop over the existing messages
    for (InitMsgList::iterator msgIter = initMsgList_.begin(),
           msgIterEnd = initMsgList_.end();
         msgIter != msgIterEnd;
         msgIter++)
    {
      InitMsgSharedPtr serializedProds = msgIter->first;
      InitMsgView existingInitMsg(&(*serializedProds)[0]);
      std::string existingOMLabel = existingInitMsg.outputModuleLabel();

      // check if the output module labels match
      if (inputOMLabel == existingOMLabel) {
        // we already have a copy of the INIT message
        addToList = false;
        ++msgIter->second;
        break;
      }
    }

    // if we've found no problems, add the new message to the collection
    if (addToList) {
      this->add(initMsgView);
    }
  }

  // indicate whether the message was added or not
  return addToList;
}


InitMsgSharedPtr
InitMsgCollection::getElementForOutputModule(const std::string& requestedOMLabel) const
{
  boost::mutex::scoped_lock sl(listLock_);
  InitMsgSharedPtr serializedProds;

  // handle the special case of an empty request
  // (If we want to use class methods to check the collection size and
  // fetch the last element in the collection, then we would need to 
  // switch to recursive mutexes...)
  if (requestedOMLabel.empty()) {
    if (initMsgList_.size() == 1) {
      serializedProds = initMsgList_.back().first;
    }
    else if (initMsgList_.size() > 1) {
      std::string msg = "Invalid INIT message lookup: the requested ";
      msg.append("HLT output module label is empty but there are multiple ");
      msg.append("HLT output modules to choose from.");
      throw cms::Exception("InitMsgCollection", "getElementForOutputModule:")
        << msg << std::endl;
    }
  }

  else {
    // loop over the existing messages
    for (InitMsgList::const_iterator msgIter = initMsgList_.begin(),
           msgIterEnd = initMsgList_.end();
         msgIter != msgIterEnd;
         msgIter++)
    {
      InitMsgSharedPtr workingMessage = msgIter->first;
      InitMsgView existingInitMsg(&(*workingMessage)[0]);
      std::string existingOMLabel = existingInitMsg.outputModuleLabel();
      
      // check if the output module labels match
      if (requestedOMLabel == existingOMLabel) {
        serializedProds = workingMessage;
        break;
      }
    }
  }

  return serializedProds;
}


InitMsgSharedPtr InitMsgCollection::getLastElement() const
{
  boost::mutex::scoped_lock sl(listLock_);

  InitMsgSharedPtr ptrToLast;
  if (!initMsgList_.empty()) {
    ptrToLast = initMsgList_.back().first;
  }
  return ptrToLast;
}


InitMsgSharedPtr InitMsgCollection::getElementAt(const unsigned int index) const
{
  boost::mutex::scoped_lock sl(listLock_);

  InitMsgSharedPtr ptrToElement;
  try
  {
    ptrToElement = initMsgList_.at(index).first;
  }
  catch (std::out_of_range& e)
  { }

  return ptrToElement;
}


void InitMsgCollection::clear()
{
  boost::mutex::scoped_lock sl(listLock_);
  initMsgList_.clear();
  outModNameTable_.clear();
}


size_t InitMsgCollection::size() const
{
  boost::mutex::scoped_lock sl(listLock_);
  return initMsgList_.size();
}


size_t InitMsgCollection::initMsgCount(const std::string& outputModuleLabel) const
{
  boost::mutex::scoped_lock sl(listLock_);

  for (InitMsgList::const_iterator msgIter = initMsgList_.begin(),
         msgIterEnd = initMsgList_.end();
       msgIter != msgIterEnd;
       msgIter++)
  {
    InitMsgSharedPtr workingMessage = msgIter->first;
    InitMsgView existingInitMsg(&(*workingMessage)[0]);
    std::string existingOMLabel = existingInitMsg.outputModuleLabel();
      
    // check if the output module labels match
    if (outputModuleLabel == existingOMLabel) {
      return msgIter->second;
    }
  }
  return 0;
}


size_t InitMsgCollection::maxMsgCount() const
{
  boost::mutex::scoped_lock sl(listLock_);

  size_t maxCount = 0;

  for (InitMsgList::const_iterator msgIter = initMsgList_.begin(),
         msgIterEnd = initMsgList_.end();
       msgIter != msgIterEnd;
       msgIter++)
  {
    if (msgIter->second > maxCount)
      maxCount = msgIter->second;
  }
  return maxCount;
}


std::string InitMsgCollection::getSelectionHelpString() const
{
  boost::mutex::scoped_lock sl(listLock_);

  // nothing much we can say if the collection is empty
  if (initMsgList_.empty()) {
    return "No information is available about the available triggers.";
  }

  // list the full set of available triggers
  std::string helpString;
  helpString.append("The full list of trigger paths is the following:");

  // we can just use the list from the first entry since all
  // subsequent entries are forced to be the same
  InitMsgSharedPtr serializedProds = initMsgList_[0].first;
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
    for (InitMsgList::const_iterator msgIter = initMsgList_.begin(),
           msgIterEnd = initMsgList_.end();
         msgIter != msgIterEnd;
         msgIter++)
    {
    serializedProds = msgIter->first;
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

  OutModTable::const_iterator it = outModNameTable_.find(outputModuleId);

  if (it == outModNameTable_.end())
  {
    return "";
  }
  else {
    return it->second;
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


void InitMsgCollection::add(InitMsgView const& initMsgView)
{
  // add the message to the internal list
  InitMsgSharedPtr serializedProds(new InitMsgBuffer(initMsgView.size()));
  initMsgList_.push_back( std::make_pair(serializedProds,1) );
  std::copy(initMsgView.startAddress(),
            initMsgView.startAddress()+initMsgView.size(),
            &(*serializedProds)[0]);

  // add the module ID name to the name map
  outModNameTable_[initMsgView.outputModuleId()] =
    initMsgView.outputModuleLabel();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
