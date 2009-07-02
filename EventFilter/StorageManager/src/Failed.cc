// $Id: Failed.cc,v 1.5 2009/07/02 10:56:42 dshpakov Exp $

#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include <iostream>

#include "xcept/tools.h"

using namespace std;
using namespace stor;

Failed::Failed( my_context c ): my_base(c)
{
  const string unknown = "unknown exception";
  string msg = "Error going into Failed state: ";
  try
    {
      TransitionRecord tr( stateName(), true );
      outermost_context().updateHistory( tr );
      outermost_context().setExternallyVisibleState( "Failed" );
      outermost_context().getNotifier()->reportNewState( "Failed" );
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
