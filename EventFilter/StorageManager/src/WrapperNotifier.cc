// $Id: WrapperNotifier.cc,v 1.3 2009/07/01 11:38:38 dshpakov Exp $
/// @file: WrapperNotifier.cc

#include "EventFilter/StorageManager/interface/WrapperNotifier.h"

#include "xdata/InfoSpace.h"


using namespace stor;

WrapperNotifier::WrapperNotifier( xdaq::Application* app ):
  _rcms_notifier( xdaq2rc::RcmsStateNotifier( app->getApplicationLogger(),
                                              app->getApplicationDescriptor(),
                                              app->getApplicationContext() )
                  ),
  _app( app )
{
  xdata::InfoSpace *ispace = app->getApplicationInfoSpace();
  
  ispace->fireItemAvailable( "rcmsStateListener",
    _rcms_notifier.getRcmsStateListenerParameter() );
  ispace->fireItemAvailable( "foundRcmsStateListener",
    _rcms_notifier.getFoundRcmsStateListenerParameter() );
  _rcms_notifier.findRcmsStateListener();
  _rcms_notifier.subscribeToChangesInRcmsStateListener( ispace );
}


void WrapperNotifier::reportNewState( const std::string& stateName )
{
  _rcms_notifier.stateChanged( stateName,
			       std::string( "StorageManager is now " ) + stateName );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
