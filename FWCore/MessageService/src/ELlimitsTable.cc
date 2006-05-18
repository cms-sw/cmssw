// ----------------------------------------------------------------------
//
// ELlimitsTable.cc
//
// History:
//   7/6/98     mf      Created
//   6/10/99    jvr     Corrected initialization for severityLimits and
//                      timespans to -1 instead of 0.
//   6/17/99    jvr     setLimit(id,n) works w/ setTimespan(sev||"*",n)
//                      and setTimespan(id,n) w/ setLimit(sev||"*",n)
//   6/15/00    web     using -> USING
//  11/01/01    web     Fixed mixed-type comparisons
//   5/18/06	mf	setInterval
//
// ----------------------------------------------------------------------


#include "FWCore/MessageService/interface/ELlimitsTable.h"

#include <iostream>
using std::cerr;

// Posible traces
// #define ELlimitsTableCONSTRUCTOR_TRACE
// #define ELlimitsTableATRACE


namespace edm {
namespace service {

// ----------------------------------------------------------------------
// Constructor, destructor:
// ----------------------------------------------------------------------


ELlimitsTable::ELlimitsTable()
: wildcardLimit   ( -1 )
, wildcardInterval( -1 )
, wildcardTimespan( -1 )
, tableLimit      ( -1 )
, limits          (    )
, counts          (    )
{

#ifdef ELlimitsTableCONSTRUCTOR_TRACE
  cerr << "Constructor for ELlimitsTable\n";
#endif

  for ( int k = 0;  k < ELseverityLevel::nLevels;  ++k )  {
    severityLimits[k]    = -1;                // JvR 99-06-10
    severityTimespans[k] = -1;
  }

}  // ELlimitsTable()


ELlimitsTable::~ELlimitsTable()  {

#ifdef ELlimitsTableCONSTRUCTOR_TRACE
  cerr << "Destructor for ELlimitsTable\n";
#endif

}  // ~ELlimitsTable()


// ----------------------------------------------------------------------
// Methods invoked by the logger:
// ----------------------------------------------------------------------

void ELlimitsTable::setTableLimit( int n )  { tableLimit = n; }


bool ELlimitsTable::add( const ELextendedID & xid )  {

#ifdef ELlimitsTableATRACE
  cerr << "&&&--- adding to limits table: " << xid.id << '\n';
#endif

  ELmap_counts::iterator c = counts.find( xid );

  if ( c == counts.end() )  {  // no such entry yet

#ifdef ELlimitsTableATRACE
    cerr << "&&&    no such entry yet in counts \n";
#endif
    int lim;
    int ts;
    ELmap_limits::iterator l = limits.find( xid.id );

    if ( l != limits.end() )  { // use limits previously established for this id
      lim = (*l).second.limit;
      ts  = (*l).second.timespan;
      if ( lim < 0 )  {                                    // jvr 6/17/99
        lim = severityLimits[xid.severity.getLevel()];
        if ( lim < 0 )  {
          lim = wildcardLimit;
        }
        limits[xid.id] = LimitAndTimespan( lim, ts );
      }

      if ( ts < 0 )  {
        ts = severityTimespans[xid.severity.getLevel()];
        if (ts < 0 )  {
          ts = wildcardTimespan;
        }
        limits[xid.id] = LimitAndTimespan( lim, ts );
      }
#ifdef ELlimitsTableATRACE
      cerr << "&&&    Entry found in limits: limit = " << lim
           << " timespan = " << ts << '\n';
#endif
    } else  {   // establish and use limits new to this id
      lim = severityLimits[xid.severity.getLevel()];
#ifdef ELlimitsTableATRACE
      cerr << "&&&    Limit taken from severityLimits: " << lim << '\n';
#endif
      if ( lim < 0 )  {
        lim = wildcardLimit;
#ifdef ELlimitsTableATRACE
        cerr << "&&&    Limit reset to wildcard limit: " << lim << '\n';
#endif
      }
      ts = severityTimespans[xid.severity.getLevel()];
#ifdef ELlimitsTableATRACE
      cerr << "&&&    Timespan taken from severityTimespans: " << ts << '\n';
#endif
      if ( ts < 0 )  {
        ts = wildcardTimespan;
#ifdef ELlimitsTableATRACE
        cerr << "&&&    timespan reset to wildcard timespan: " << ts << '\n';
#endif
      }

      // save, if possible, id's future limits:
      if ( tableLimit < 0  || static_cast<int>(limits.size()) < tableLimit )
        limits[xid.id] = LimitAndTimespan( lim, ts );
    }

    // save, if possible, this xid's initial entry:
    if ( tableLimit < 0  || static_cast<int>(counts.size()) < tableLimit )
      counts[xid] = CountAndLimit( lim, ts );
    c = counts.find( xid );
  }

  return  ( c == counts.end() )
        ? true               // no limit filtering can be applied
        : (*c).second.add()  // apply limit filtering
        ;

}  // add()


// ----------------------------------------------------------------------
// Control methods invoked by the framework:
// ----------------------------------------------------------------------

void ELlimitsTable::wipe()  {
  // This clears everything -- counts and aggregate counts for severity levels
  // and for individual ID's, as well as any limits established, the limit
  // for "*" all messages, and the collection of severity defaults.  wipe()
  // does not not affect thresholds.

  limits.erase( limits.begin(), limits.end() );
  ELmap_counts::iterator i;
  for ( i = counts.begin();  i != counts.end();  ++i )  {
    (*i).second.limit = -1;
    (*i).second.n = (*i).second.aggregateN = 0;
  }

  wildcardLimit    = -1;
  wildcardTimespan = -1;
  for ( int lev = 0;  lev < ELseverityLevel::nLevels;  ++lev )  {
    severityLimits   [lev] = -1;
    severityTimespans[lev] = -1;
  }

}


void ELlimitsTable::zero()  {
  // This clears counts but not aggregate counts for severity levels
  // and for individual ID's.

  ELmap_counts::iterator i;
  for ( i = counts.begin();  i != counts.end();  ++i )
    (*i).second.n = 0;
}


void ELlimitsTable::setLimit( const ELstring & id, int n )  {
  if ( id[0] == '*' )  wildcardLimit    = n;
  else                 limits[id].limit = n;
}


void ELlimitsTable::setLimit( const ELseverityLevel & sev, int n )  {
  severityLimits[sev.getLevel()] = n;
}

void ELlimitsTable::setInterval( const ELstring & id, int interval )  {
  if ( id[0] == '*' )  wildcardInterval    = interval;
  else                 limits[id].interval = interval;
}


void ELlimitsTable::setInterval( const ELseverityLevel & sev, int interval )  {
  severityIntervals[sev.getLevel()] = interval;
}


void ELlimitsTable::setTimespan( const ELstring & id, int n )  {
  if ( id[0] == '*' )  wildcardTimespan    = n;
  else                 limits[id].timespan = n;
}


void ELlimitsTable::setTimespan( const ELseverityLevel & sev, int n )  {
  severityTimespans[sev.getLevel()] = n;
}


// ----------------------------------------------------------------------
// Support for internal operations:
// ----------------------------------------------------------------------

ELlimitsTable & ELlimitsTable::operator=( const ELlimitsTable & t )  {

  limits = t.limits;  // The non-trivial operator= for a map!

  for ( int lev = 0;  lev < ELseverityLevel::nLevels;  ++lev )  {
    severityTimespans[lev] = t.severityTimespans[lev];
    severityTimespans[lev] = t.severityTimespans[lev];
  }

  wildcardLimit    = t.wildcardLimit;
  wildcardTimespan = t.wildcardTimespan;

  return  *this;

}  // operator=()


// ----------------------------------------------------------------------


} // end of namespace service  
} // end of namespace edm  
