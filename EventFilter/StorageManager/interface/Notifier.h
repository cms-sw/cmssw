// $Id: Notifier.h,v 1.6 2009/07/13 14:51:13 mommsen Exp $
/// @file: Notifier.h 

#ifndef StorageManager_Notifier_h
#define StorageManager_Notifier_h

#include <string>

#include "xdaq/Application.h"
#include "xdaq/exception/Exception.h"

namespace stor
{

  /**
    Interface class for handling RCMS notifier
    
    $Author: mommsen $
    $Revision: 1.6 $
    $Date: 2009/07/13 14:51:13 $
  */

  class Notifier
  {

  public:

    /**
       Constructor
    */
    Notifier() {}

    /**
       Destructor
    */
    virtual ~Notifier() {};

    /**
       Report new state to RCMS
    */
    virtual void reportNewState( const std::string& stateName ) = 0;

    /**
       Access logger
    */
    virtual Logger& getLogger() = 0;

    /**
       Send message to sentinel
    */
    virtual void tellSentinel( const std::string& level, xcept::Exception& e ) = 0;

    /**
       Write message to a file in /tmp (last resort when everything
       else fails)
    */
    void localDebug( const std::string& message ) const;

  protected:

    /**
       Storage manager instance number
    */
    virtual unsigned long instanceNumber() const = 0;

  };

}

#endif // StorageManager_Notifier_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
