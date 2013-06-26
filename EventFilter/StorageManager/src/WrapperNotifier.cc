// $Id: WrapperNotifier.cc,v 1.5 2011/03/07 15:31:32 mommsen Exp $
/// @file: WrapperNotifier.cc

#include "EventFilter/StorageManager/interface/WrapperNotifier.h"

#include "xdata/InfoSpace.h"


using namespace stor;

WrapperNotifier::WrapperNotifier( xdaq::Application* app ):
  rcmsNotifier_(
    xdaq2rc::RcmsStateNotifier(
      app->getApplicationLogger(),
      app->getApplicationDescriptor(),
      app->getApplicationContext()
    )
  ),
  app_( app )
{
  xdata::InfoSpace *ispace = app->getApplicationInfoSpace();
  
  ispace->fireItemAvailable( "rcmsStateListener",
    rcmsNotifier_.getRcmsStateListenerParameter() );
  ispace->fireItemAvailable( "foundRcmsStateListener",
    rcmsNotifier_.getFoundRcmsStateListenerParameter() );
  rcmsNotifier_.findRcmsStateListener();
  rcmsNotifier_.subscribeToChangesInRcmsStateListener( ispace );
}


void WrapperNotifier::reportNewState( const std::string& stateName )
{
  rcmsNotifier_.stateChanged(
    stateName,
    std::string( "StorageManager is now " ) + stateName
  );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
