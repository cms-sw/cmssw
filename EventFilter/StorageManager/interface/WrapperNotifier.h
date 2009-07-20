// $Id: WrapperNotifier.h,v 1.7 2009/07/14 10:34:44 dshpakov Exp $
/// @file: WrapperNotifier.h 

#ifndef StorageManager_WrapperNotifier_h
#define StorageManager_WrapperNotifier_h

#include "EventFilter/StorageManager/interface/Notifier.h"

#include "xdaq/Application.h"
#include "xdaq2rc/RcmsStateNotifier.h"


namespace stor
{

  /**
     Notifier implementation used by StorageManager

     $Author: dshpakov $
     $Revision: 1.7 $
     $Date: 2009/07/14 10:34:44 $
  */
  class WrapperNotifier: public Notifier
  {
    
  public:

    /**
       Constructor
    */
    WrapperNotifier( xdaq::Application* app );

    /**
       Report new state to RCMS
    */
    void reportNewState( const std::string& stateName );

    /**
       Access logger
    */
    Logger& getLogger() { return _app->getApplicationLogger(); }

    /**
       Send message to sentinel
    */
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

#endif // StorageManager_WrapperNotifier_h



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
