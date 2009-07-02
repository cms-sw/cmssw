// -*- c++ -*-                                                                              
// $Id: Notifier.h,v 1.3 2009/07/01 13:08:17 dshpakov Exp $

#ifndef NOTIFIER_H
#define NOTIFIER_H

// Interface class for handling RCMS notifier

#include <string>

#include "xdaq/Application.h"
#include "xdaq/exception/Exception.h"

namespace stor
{

  class Notifier
  {

  public:

    Notifier() {}

    virtual ~Notifier() {};

    virtual void reportNewState( const std::string& stateName ) = 0;
    virtual Logger& getLogger() = 0;
    virtual void tellSentinel( const std::string& level, xcept::Exception& e ) = 0;

    void localDebug( const std::string& message ) const;

  protected:

    virtual unsigned long instanceNumber() const = 0;

  };

}

#endif // NOTIFIER_H
