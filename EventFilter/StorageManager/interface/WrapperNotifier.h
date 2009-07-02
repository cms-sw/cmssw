// -*- c++ -*-
// $Id: WrapperNotifier.h,v 1.4 2009/07/01 13:08:17 dshpakov Exp $

#ifndef WRAPPERNOTIFIER_H
#define WRAPPERNOTIFIER_H

// Notifier implementation to be used by StorageManager

#include "EventFilter/StorageManager/interface/Notifier.h"

#include "xdaq/Application.h"
#include "xdaq2rc/RcmsStateNotifier.h"


namespace stor
{

  class WrapperNotifier: public Notifier
  {
    
  public:

    WrapperNotifier( xdaq::Application* app );
    
    void reportNewState( const std::string& stateName );
    Logger& getLogger() { return _app->getApplicationLogger(); }
    void tellSentinel( const std::string& level,
                       xcept::Exception& e )
    {
      _app->notifyQualified( level, e );
    }

  private:

    xdaq2rc::RcmsStateNotifier _rcms_notifier;
    xdaq::Application* _app;

    unsigned long instanceNumber() const
    {
      return _app->getApplicationDescriptor()->getInstance();
    }

  };

}

#endif // WRAPPERNOTIFIER_H



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
