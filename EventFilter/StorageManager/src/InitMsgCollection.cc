/**
 * This class is used to manage the set of INIT messages that have
 * been received by the storage manager and will be sent to event
 * consumers and written to output disk files.
 *
 * $Id: InitMsgCollection.cc,v 1.4 2008/04/16 01:39:59 biery Exp $
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

#if 0
    // 29-Apr-2008 KAB - replaced the following methods as part of the switch
    // to the newer HLT output module selection scheme (in which the HLT
    // output module needs to be explicitly specified)
    //
    // testAndAddIfUnique() replaced by addIfUnique()
    // getElementForSelection() replaced by getElementForOutputModule()
    //
    bool testAndAddIfUnique(InitMsgView const& initMsgView);
    InitMsgSharedPtr getElementForSelection(Strings const& triggerSelection);

/**
 * Adds the specified INIT message to the collection if it is unique and
 * if it passes a number of consistency checks.  The consistency checks
 * include tests like requiring that all INIT messages have the same
 * HLT full trigger list and requiring that the trigger selections from
 * multiple messages do not overlap.
 *
 * If the message fails a consistency check, an exception is thrown.  If
 * the consistency checks pass, but we already have a copy of the input INIT
 * message (from a different filter unit, presumably), the duplicate
 * message is *not* added to the collection, and this method returns false.
 * If the consistency checks pass and the message is unique, it is added
 * to the collection, and the method returns true;
 *
 * @param initMsgView The INIT message to be added to the collection.
 * @return true if the message was added to the collection, false otherwise.
 * @throws cms::Exception if one of the consistency checks fails.
 */
bool InitMsgCollection::testAndAddIfUnique(InitMsgView const& initMsgView)
{
  boost::mutex::scoped_lock sl(listLock_);

  // initially, assume that we will want to include the new message
  bool addToList = true;

  // test the message to verify that its full trigger list is not empty
  std::string inputOMLabel = initMsgView.outputModuleLabel();
  Strings inputTriggerList;
  initMsgView.hltTriggerNames(inputTriggerList);
  if (inputTriggerList.size() == 0) {
    addToList = false; // useless unless we remove the exception...
    throw cms::Exception("InitMsgCollection", "testAndAddIfUnique:")
      << "The full trigger list specified for the \"" << inputOMLabel
      << "\" output module is empty!" << std::endl;
  }

  // test the message to verify that its trigger selection is valid
  // in the context of its full trigger list
  Strings inputSelectionList;
  initMsgView.hltTriggerSelections(inputSelectionList);
  if (! EventSelector::selectionIsValid(inputSelectionList,
                                        inputTriggerList)) {
    addToList = false; // useless unless we remove the exception...
    throw cms::Exception("InitMsgCollection", "testAndAddIfUnique:")
      << "The trigger selection specified for the \"" << inputOMLabel
      << "\" output module is not valid for the full trigger list!"
      << std::endl;
  }

  // if this is the first INIT message that we've seen, we just add it
  if (initMsgList_.size() == 0) {
    this->add(initMsgView);
  }

  // if this is a subsequent INIT message, we have to run some tests
  else {

    // loop over the existing messages
    std::vector<InitMsgSharedPtr>::const_iterator msgIter;
    for (msgIter = initMsgList_.begin(); msgIter != initMsgList_.end(); msgIter++) {
      InitMsgSharedPtr serializedProds = *msgIter;
      InitMsgView existingInitMsg(&(*serializedProds)[0]);
      std::string existingOMLabel = existingInitMsg.outputModuleLabel();

      // independent of everything else that we check, the full trigger list
      // in *every* INIT message must match the full trigger list in every
      // other INIT message.  If the list of trigger paths are not the same,
      // how could we ever hope to compare selection lists or trigger results?

      // check that the full trigger lists are identical
      Strings existingTriggerList;
      existingInitMsg.hltTriggerNames(existingTriggerList);
      if (inputTriggerList != existingTriggerList) {
        addToList = false; // useless unless we remove the exception...
        throw cms::Exception("InitMsgCollection", "testAndAddIfUnique:")
          << "INIT messages from the \"" << inputOMLabel << "\" and \""
          << existingOMLabel << "\" output modules have "
          << "different HLT full trigger lists!" << std::endl;
      }

      // fetch the trigger selection list to be used in later tests
      Strings existingSelectionList;
      existingInitMsg.hltTriggerSelections(existingSelectionList);

      // check if the output module labels match
      if (inputOMLabel == existingOMLabel) {

        // if the output module labels match, and everything else in the
        // messages match, then we presume that the new message is just
        // a duplicate sent from the Nth filter unit.  If something in
        // the INIT messages do not match, we have the
        // odd situation in which multiple filter units are sending us
        // INIT messages from what they claim is same output module, but
        // is not really.

        // check that the trigger selections are identical
        if (EventSelector::testSelectionOverlap(inputSelectionList,
                                                existingSelectionList,
                                                existingTriggerList) !=
            evtSel::ExactMatch) {
          addToList = false; // useless unless we remove the exception...
          throw cms::Exception("InitMsgCollection", "testAndAddIfUnique:")
            << "INIT messages from the \"" << inputOMLabel
            << "\" output module in different filter units have "
            << "different HLT trigger selections!" << std::endl;
        }

        // check that the product lists are identical
        std::auto_ptr<SendJobHeader> header =
          StreamerInputSource::deserializeRegistry(initMsgView);
        std::auto_ptr<SendJobHeader> refheader =
          StreamerInputSource::deserializeRegistry(existingInitMsg);
        if (! registryIsSubset(*header, *refheader) ||
            ! registryIsSubset(*refheader, *header)) {
          addToList = false; // useless unless we remove the exception...
          throw cms::Exception("InitMsgCollection", "testAndAddIfUnique:")
            << "INIT messages from the \"" << inputOMLabel
            << "\" output module in different filter units have "
            << "different product lists!" << std::endl;
        }

        // at this point, do we return or continue with tests against
        // the remaining INIT messages in the list?
        // Let's return because A) the new message exactly matches an
        // existing one in the list, and B) the messages in the list should
        // have passed all of the remaining tests to get in the list.
        addToList = false;  // not really needed since we return...
        return addToList;
      }
      else {

        // if the output module labels don't match, check if the trigger
        // selection in the new INIT message overlaps with the selection
        // in the existing message.  If it does, that is a problem because
        // we need the trigger selection lists to be non-overlapping (so
        // that masked TriggerResults can possibly only match one
        // selection list)

        // check that the trigger selections do not overlap
        if (EventSelector::testSelectionOverlap(inputSelectionList,
                                                existingSelectionList,
                                                existingTriggerList) !=
            evtSel::NoOverlap) {
          addToList = false; // useless unless we remove the exception...
          throw cms::Exception("InitMsgCollection", "testAndAddIfUnique:")
            << "INIT messages from the \"" << inputOMLabel << "\" and \""
            << existingOMLabel << "\" output modules have "
            << "overlapping HLT trigger selections: ("
            << stringsToText(inputSelectionList, 10) << ") and ("
            << stringsToText(existingSelectionList, 10) << ")."
            << std::endl;
        }
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
 * Fetches the single INIT message that matches the input trigger selection.
 * If the trigger selection is not valid for the full trigger list contained
 * in the INIT messages in this collection or if multiple INIT messages
 * match the selection, an exception is thrown.  If no messages match
 * the selection, an empty pointer is returned.
 *
 * @param triggerSelection The trigger selection list to use when
 *        searching for the appropriate INIT message (vector of strings).
 * @return a pointer to the INIT message that matches.  If no
 *         matching INIT message is found, and empty pointer is returned.
 * @throws cms::Exception if the input selection is not valid in the
 *         context of the full trigger list referenced by the INIT
 *         messages in the collection OR if multiple INIT messages match
 *         the selection.
 */
InitMsgSharedPtr
InitMsgCollection::getElementForSelection(Strings const& triggerSelection)
{
  boost::mutex::scoped_lock sl(listLock_);

  InitMsgSharedPtr serializedProds;
  if (initMsgList_.size() > 0) {

    // check that the input trigger selection is valid
    InitMsgSharedPtr workingMessage = initMsgList_[0];
    InitMsgView existingInitMsg(&(*workingMessage)[0]);
    Strings fullTriggerList;
    existingInitMsg.hltTriggerNames(fullTriggerList);
    if (! EventSelector::selectionIsValid(triggerSelection,
                                          fullTriggerList)) {
      std::string msg = "The specified trigger selection list (";
      msg.append(stringsToText(triggerSelection, 10));
      msg.append(") contains paths not in the full trigger list!");
      throw cms::Exception("InitMsgCollection", "getElementForSelection:")
        << msg << std::endl;
    }

    // loop over the existing messages
    std::vector<InitMsgSharedPtr>::const_iterator msgIter;
    for (msgIter = initMsgList_.begin(); msgIter != initMsgList_.end(); msgIter++) {
      workingMessage = *msgIter;
      InitMsgView workingInitMsg(&(*workingMessage)[0]);

      Strings workingSelectionList;
      workingInitMsg.hltTriggerSelections(workingSelectionList);

      evtSel::OverlapResult overlapResult =
        EventSelector::testSelectionOverlap(triggerSelection,
                                            workingSelectionList,
                                            fullTriggerList);
      if (overlapResult == evtSel::ExactMatch ||
          overlapResult == evtSel::PartialOverlap) {
        if (serializedProds.get() == NULL) {
          serializedProds = workingMessage;
        }
        else {
          std::string msg = "The specified trigger selection list (";
          msg.append(stringsToText(triggerSelection, 10));
          msg.append(") matches triggers from more than one HLT output module!");
          throw cms::Exception("InitMsgCollection", "getElementForSelection:")
            << msg << std::endl;
        }
      }
    }
  }

  return serializedProds;
}

#endif

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
  boost::mutex::scoped_lock sl(listLock_);
  initMsgList_.clear();
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
