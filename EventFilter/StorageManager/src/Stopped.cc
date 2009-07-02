// $Id: Stopped.cc,v 1.2 2009/06/10 08:15:28 dshpakov Exp $

#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"

#include <iostream>

#include "xcept/tools.h"

using namespace std;
using namespace stor;

Stopped::Stopped( my_context c ): my_base(c)
{
  const string unknown = "unknown exception";
  string msg = "Error going into Stopped state: ";
  try
    {
      TransitionRecord tr( stateName(), true );
      outermost_context().updateHistory( tr );
      outermost_context().setExternallyVisibleState( "Ready" );
      outermost_context().getNotifier()->reportNewState( "Ready" );
    }
  catch( xcept::Exception& e )
    {
      try
        {
          LOG4CPLUS_FATAL( outermost_context().getNotifier()->getLogger(),
                           msg << xcept::stdformat_exception_history(e) );
          XCEPT_DECLARE_NESTED( stor::exception::StateTransition,
                                sentinelException, msg, e );
          outermost_context().getNotifier()->tellSentinel( "fatal",
                                                           sentinelException );
        }
      catch(...)
        {
          outermost_context().getNotifier()->localDebug( unknown );
        }
    }
  catch( std::exception& e )
    {
      try
        {
          msg += e.what();
          LOG4CPLUS_FATAL( outermost_context().getNotifier()->getLogger(), msg );
          XCEPT_DECLARE( stor::exception::StateTransition,
                         sentinelException, msg );
          outermost_context().getNotifier()->tellSentinel( "fatal",
                                                           sentinelException );
        }
      catch(...)
        {
          outermost_context().getNotifier()->localDebug( unknown );
        }
    }
  catch(...)
    {
      msg += "unknown exception";
      try
        {
          LOG4CPLUS_FATAL( outermost_context().getNotifier()->getLogger(), msg );
          XCEPT_DECLARE( stor::exception::StateTransition,
                         sentinelException, msg );
          outermost_context().getNotifier()->tellSentinel( "fatal",
                                                           sentinelException );
        }
      catch(...)
        {
          outermost_context().getNotifier()->localDebug( unknown );
        }
    }
}

Stopped::~Stopped()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Stopped::do_stateName() const
{
  return string( "Stopped" );
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
