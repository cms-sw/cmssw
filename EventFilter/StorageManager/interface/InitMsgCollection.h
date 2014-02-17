// $Id: InitMsgCollection.h,v 1.15 2012/04/20 10:48:18 mommsen Exp $
/// @file: InitMsgCollection.h 

#ifndef EventFilter_StorageManager_InitMsgCollection_h
#define EventFilter_StorageManager_InitMsgCollection_h

#include "IOPool/Streamer/interface/InitMessage.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace stor
{

  class I2OChain;

  /**
     This class is used to manage the unique set of INIT messages
     that have been received by the storage manager and will be sent
     to event consumers and written to output streams.

     $Author: mommsen $
     $Revision: 1.15 $
     $Date: 2012/04/20 10:48:18 $
  */

  typedef std::vector<unsigned char> InitMsgBuffer;
  typedef boost::shared_ptr<InitMsgBuffer> InitMsgSharedPtr;

  class InitMsgCollection
  {

  public:

    /**
     * InitMsgCollection constructor.
     */
    InitMsgCollection();

    /**
     * InitMsgCollection destructor.
     */
    ~InitMsgCollection();

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
    bool addIfUnique(InitMsgView const& initMsgView);

    /**
     * Adds the INIT message contained in the specified I2OChain to the
     * collection if it has a unique HLT output module label.
     *
     * If we already have an INIT message with the same output module label
     * as the input INIT message, the duplicate
     * message is *not* added to the collection, and this method returns false.
     *
     * If the INIT message is new, the InitMsgView will contain the view of
     * INIT message and the method returns true.
     *
     * If the output module label inside the INIT message is empty, an
     * exception is thrown.
     *
     * @param i2oChain the I2OChain containing an INIT message
     * @param initMsgView the INIT view which was added to the collection.
     * @return true if the message was added to the collection, false otherwise.
     * @throws cms::Exception if one of the consistency checks fails.
     */
    bool addIfUnique(I2OChain const&, InitMsgSharedPtr&);

    /**
     * Fetches the single INIT message that matches the requested HLT output
     * module id.  If no messages match the request, an empty pointer
     * is returned.
     *
     * @param requestedOutputModuleId The HLT output module id of the INIT
     *        message to be returned.
     * @return a pointer to the INIT message that matches.  If no
     *         matching INIT message is found, and empty pointer is returned.
     */
    InitMsgSharedPtr getElementForOutputModuleId(const uint32_t&) const;

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
     * @param requestedOutputModuleLabel The HLT output module label of the INIT
     *        message to be returned.
     * @return a pointer to the INIT message that matches.  If no
     *         matching INIT message is found, and empty pointer is returned.
     * @throws cms::Exception if the input HLT output module label string is
     *         empty and there is more than one INIT message in the collection.
     */
    InitMsgSharedPtr getElementForOutputModuleLabel(const std::string&) const;

    /**
     * Returns a shared pointer to the requested element in the collection
     * or an empty pointer if the requested index if out of bounds.
     *
     * @param index The index of the requested element.
     * @return the InitMsgSharedPtr at the requested index in the collection.
     */
    InitMsgSharedPtr getElementAt(const unsigned int index) const;
 
    /**
     * Removes all entries from the collection.
     */
    void clear();

    /**
     * Returns the number of unique INIT messages in the collection.
     *
     * @return the integer number of messages.
     */
    size_t size() const;

    /**
     * Returns a string with information on which selections are available.
     *
     * @return the help string.
     */
    std::string getSelectionHelpString() const;

    /**
     * Returns the name of the output module with the specified module ID,
     * or an empty string of the specified module ID is not known.
     *
     * @return the output module label or an empty string
     */
    std::string getOutputModuleName(const uint32_t outputModuleId) const;

    /**
     * Creates a single text string from the elements in the specified
     * list of strings.  The specified maximum number of elements are
     * included, however a zero value for the maximum number will include
     * all elements.
     *
     * @param list the list of strings to include (std::vector of strings);
     * @param maxCount the maximum number of list elements to include.
     * @return the text string with the formatted list elements.
     */
    static std::string stringsToText(Strings const& list,
                                     unsigned int maxCount = 0);

  private:

    void checkOutputModuleLabel(InitMsgView const&) const;

    typedef std::map<uint32_t,InitMsgSharedPtr> InitMsgMap;
    InitMsgMap initMsgMap_;
    mutable boost::mutex listLock_;
  };
  
  typedef boost::shared_ptr<InitMsgCollection> InitMsgCollectionPtr;

} //namespace stor

#endif // EventFilter_StorageManager_InitMsgCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
