/**
 * This class is used to manage the set of INIT messages that have
 * been received by the storage manager and will be sent to event
 * consumers and written to output disk files.
 *
 * $Id: InitMsgCollection.cc,v 1.7 2009/06/10 08:15:27 dshpakov Exp $
/// @file: InitMsgCollection.cc
 */

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

/**
 * InitMsgCollection constructor.
 */
InitMsgCollection::InitMsgCollection()
{
  FDEBUG(5) << "Executing constructor for InitMsgCollection" << std::endl;
  initMsgList_.clear();
  outModNameTable_.clear();
  consumerOutputModuleMap_.clear();
  serializedFullSet_.reset(new InitMsgBuffer(2 * sizeof(Header)));
  OtherMessageBuilder fullSetMsg(&(*serializedFullSet_)[0], Header::INIT_SET);
}

/**
 * InitMsgCollection destructor.
 */
InitMsgCollection::~InitMsgCollection()
{
  FDEBUG(5) << "Executing destructor for InitMsgCollection" << std::endl;
}

/**
 * Adds the specified INIT message to the collection if it has a unique
 * HLT output module label.
 *
 * If we already have an INIT message with the same output module label
 * as the input INIT message, the duplicate
 * message is *not* added to the collection, and this method returns false.
 *
 * If the output module label inside the INIT message is empty, an
 * exception is thrown.
 *
 * @param initMsgView The INIT message to be added to the collection.
 * @return true if the message was added to the collection, false otherwise.
 * @throws cms::Exception if one of the consistency checks fails.
 */
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
  if (initMsgList_.size() == 0) {
    this->add(initMsgView);
  }

  // if this is a subsequent INIT message, check if it is unique
  else {

    // loop over the existing messages
    std::vector<InitMsgSharedPtr>::const_iterator msgIter;
    for (msgIter = initMsgList_.begin(); msgIter != initMsgList_.end(); msgIter++) {
      InitMsgSharedPtr serializedProds = *msgIter;
      InitMsgView existingInitMsg(&(*serializedProds)[0]);
      std::string existingOMLabel = existingInitMsg.outputModuleLabel();

      // check if the output module labels match
      if (inputOMLabel == existingOMLabel) {
        // we already have a copy of the INIT message
        addToList = false;
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

/**
 * Fetches the single INIT message that matches the requested HLT output
 * module label.  If no messages match the request, an empty pointer
 * is returned.
 *
 * If the requested HLT output module label is empty, and there is only
 * one INIT message in the collection, that INIT message is returned.
 * However, if there is more than one INIT message in the collection, and
 * an empty request is passed into this method, an exception will be thrown.
 * (How can we decide which is the best match when we have nothing to work
 * with?)
 *
 * @param requestedOMLabel The HLT output module label of the INIT
 *        message to be returned.
 * @return a pointer to the INIT message that matches.  If no
 *         matching INIT message is found, and empty pointer is returned.
 * @throws cms::Exception if the input HLT output module label string is
 *         empty and there is more than one INIT message in the collection.
 */
InitMsgSharedPtr
InitMsgCollection::getElementForOutputModule(std::string requestedOMLabel)
{
  boost::mutex::scoped_lock sl(listLock_);
  InitMsgSharedPtr serializedProds;

  // handle the special case of an empty request
  // (If we want to use class methods to check the collection size and
  // fetch the last element in the collection, then we would need to 
  // switch to recursive mutexes...)
  if (requestedOMLabel.empty()) {
    if (initMsgList_.size() == 1) {
      serializedProds = initMsgList_.back();
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
    std::vector<InitMsgSharedPtr>::const_iterator msgIter;
    for (msgIter = initMsgList_.begin(); msgIter != initMsgList_.end(); msgIter++) {
      InitMsgSharedPtr workingMessage = *msgIter;
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

/**
 * Returns a shared pointer to the last element in the collection
 * or an empty pointer if the collection has no elements.
 *
 * @return the last InitMsgSharedPtr in the collection.
 */
InitMsgSharedPtr InitMsgCollection::getLastElement()
{
  boost::mutex::scoped_lock sl(listLock_);

  InitMsgSharedPtr ptrToLast;
  if (initMsgList_.size() > 0) {
    ptrToLast = initMsgList_.back();
  }
  return ptrToLast;
}

/**
 * Returns a shared pointer to the requested element in the collection
 * or an empty pointer if the requested index if out of bounds.
 *
 * @param index The index of the requested element.
 * @return the InitMsgSharedPtr at the requested index in the collection.
 */
InitMsgSharedPtr InitMsgCollection::getElementAt(unsigned int index)
{
  boost::mutex::scoped_lock sl(listLock_);

  InitMsgSharedPtr ptrToElement;
  if (index >= 0 && index < initMsgList_.size()) {
    ptrToElement = initMsgList_[index];
  }
  return ptrToElement;
}

/**
 * Removes all entries from the collection.
 */
void InitMsgCollection::clear()
{
  {
    boost::mutex::scoped_lock sl(consumerMapLock_);
    consumerOutputModuleMap_.clear();
  }
  {
    boost::mutex::scoped_lock sl(listLock_);
    initMsgList_.clear();
    outModNameTable_.clear();
  }
}

/**
 * Returns the number of INIT messages in the collection.
 *
 * @return the integer number of messages.
 */
int InitMsgCollection::size()
{
  boost::mutex::scoped_lock sl(listLock_);
  return initMsgList_.size();
}

/**
 * Returns a string with information on which selections are available.
 *
 * @return the help string.
 */
std::string InitMsgCollection::getSelectionHelpString()
{
  // nothing much we can say if the collection is empty
  if (initMsgList_.size() == 0) {
    return "No information is available about the available triggers.";
  }

  // list the full set of available triggers
  std::string helpString;
  helpString.append("The full list of trigger paths is the following:");

  // we can just use the list from the first entry since all
  // subsequent entries are forced to be the same
  InitMsgSharedPtr serializedProds = initMsgList_[0];
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
  std::vector<InitMsgSharedPtr>::const_iterator msgIter;
  for (msgIter = initMsgList_.begin(); msgIter != initMsgList_.end(); msgIter++) {
    serializedProds = *msgIter;
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

/**
 * Returns the name of the output module with the specified module ID,
 * or an empty string of the specified module ID is not known.
 *
 * @return the output module label or an empty string
 */
std::string InitMsgCollection::getOutputModuleName(uint32 outputModuleId)
{
  if (outModNameTable_.find(outputModuleId) == outModNameTable_.end())
  {
    return "";
  }
  else {
    return outModNameTable_[outputModuleId];
  }
}

/**
 * Creates a single text string from the elements in the specified
 * list of strings.  The specified maximum number of elements are
 * included, however a zero value for the maximum number will include
 * all elements.
 *
 * @param list the list of strings to include (vector of strings);
 * @param maxCount the maximum number of list elements to include.
 * @return the text string with the formatted list elements.
 */
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

/**
 * Adds the specified INIT message to the collection (unconditionally).
 *
 * @param initMsgView The INIT message to add to the collection.
 */
void InitMsgCollection::add(InitMsgView const& initMsgView)
{
  // add the message to the internal list
  InitMsgSharedPtr serializedProds(new InitMsgBuffer(initMsgView.size()));
  initMsgList_.push_back(serializedProds);
  std::copy(initMsgView.startAddress(),
            initMsgView.startAddress()+initMsgView.size(),
            &(*serializedProds)[0]);

  // add the module ID name to the name map
  outModNameTable_[initMsgView.outputModuleId()] =
    initMsgView.outputModuleLabel();

  // calculate various sizes needed for adding the message to
  // the serialized version of the full set
  OtherMessageView fullSetView(&(*serializedFullSet_)[0]);
  unsigned int oldBodySize = fullSetView.bodySize();
  unsigned int oldBufferSize = serializedFullSet_->size();
  unsigned int newBodySize = oldBodySize + initMsgView.size();
  unsigned int newBufferSize = oldBufferSize + initMsgView.size();

  // add the message to the serialized full set of messages
  serializedFullSet_->resize(newBufferSize);
  OtherMessageBuilder fullSetMsg(&(*serializedFullSet_)[0],
                                 Header::INIT_SET,
                                 newBodySize);
  uint8 *copyPtr = fullSetMsg.msgBody() + oldBodySize;
  std::copy(initMsgView.startAddress(),
            initMsgView.startAddress()+initMsgView.size(),
            copyPtr);
}

////////////////////////////
//// Register consumer: ////
////////////////////////////
bool InitMsgCollection::registerConsumer( ConsumerID cid, const std::string& hltModule )
{
  if ( ! cid.isValid() ) { return false; }
  if ( hltModule == "" ) { return false; }

  boost::mutex::scoped_lock sl( consumerMapLock_ );
  consumerOutputModuleMap_[ cid ] = hltModule;
  return true;
}

/**
 * Fetches the single INIT message that matches the requested consumer ID.
 * If no messages match the request, an empty pointer is returned.
 */
InitMsgSharedPtr InitMsgCollection::getElementForConsumer( ConsumerID cid )
{
  std::string outputModuleLabel;
  {
    boost::mutex::scoped_lock sl( consumerMapLock_ );

    std::map< ConsumerID, std::string >::const_iterator mapIter;
    mapIter = consumerOutputModuleMap_.find( cid );

    if ( mapIter != consumerOutputModuleMap_.end() &&
         mapIter->second != "" )
      {
        outputModuleLabel = mapIter->second;
      }
    else
      {
        InitMsgSharedPtr serializedProds;
        return serializedProds;
      }
  }

  return getElementForOutputModule(outputModuleLabel);
}
