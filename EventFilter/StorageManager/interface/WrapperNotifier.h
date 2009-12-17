// $Id: WrapperNotifier.h,v 1.9 2009/09/29 07:54:01 mommsen Exp $
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

     $Author: mommsen $
     $Revision: 1.9 $
     $Date: 2009/09/29 07:54:01 $
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

  private:

    xdaq2rc::RcmsStateNotifier _rcms_notifier;
    xdaq::Application* _app;

  };

}

#endif // StorageManager_WrapperNotifier_h



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
