// $Id: Failed.cc,v 1.2 2009/06/10 08:15:26 dshpakov Exp $

#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include <iostream>

using namespace std;
using namespace stor;

Failed::Failed( my_context c ): my_base(c)
{

  string msg = "";

  try
    {
      TransitionRecord tr( stateName(), true );
      outermost_context().updateHistory( tr );
      outermost_context().setExternallyVisibleState( "Failed" );
      outermost_context().getNotifier()->reportNewState( "Failed" );
    }
  catch( xcept::Exception& e )
    {
      msg = e.what();
    }
  catch( std::exception& e )
    {
      msg = e.what();
    }
  catch(...)
    {
      msg = "Unknown exception";
    }

  if( msg != "" )
    {
      try
        {
          LOG4CPLUS_ERROR( outermost_context().getNotifier()->getLogger(), msg );
          XCEPT_DECLARE( stor::exception::Exception, sentinelException, msg );
          outermost_context().getNotifier()->tellSentinel( "error", sentinelException );
        }
      catch(...)
        {
          // Nothing else can be done...
        }
    }

}

Failed::~Failed()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Failed::do_stateName() const
{
  return string( "Failed" );
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
