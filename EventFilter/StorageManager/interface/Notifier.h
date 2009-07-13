// -*- c++ -*-                                                                              
// $Id: Notifier.h,v 1.5 2009/07/13 13:27:45 dshpakov Exp $

#ifndef NOTIFIER_H
#define NOTIFIER_H

#include <string>

#include "xdaq/Application.h"
#include "xdaq/exception/Exception.h"

namespace stor
{

  /**
    Interface class for handling RCMS notifier
    
    $Author: $
    $Revision: $
    $Date: $
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

#endif // NOTIFIER_H
