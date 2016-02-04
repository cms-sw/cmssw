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
//  11/2/07	mf	add:  Changed ivl = wildcardLimit to wildcardInterval.
//			Probably moot and never reached, but a clear correction.
//  9/29/10	mf,ql	Fix savanah bug 65284 where two messages of same
//			category but different severity, if limits were not
//			take from category, first severity sets the limit
//			for both of those xid's.
//
// ----------------------------------------------------------------------


#include "FWCore/MessageService/interface/ELlimitsTable.h"

// Posible traces
//#include <iostream>
//using std::cerr;
//#define ELlimitsTableCONSTRUCTOR_TRACE
//#define ELlimitsTableATRACE


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
  std::cerr << "Constructor for ELlimitsTable\n";
#endif

  for ( int k = 0;  k < ELseverityLevel::nLevels;  ++k )  {
    severityLimits[k]    = -1;                // JvR 99-06-10
    severityIntervals[k]    = -1;             
    severityTimespans[k] = -1;
  }

}  // ELlimitsTable()


ELlimitsTable::~ELlimitsTable()  {

#ifdef ELlimitsTableCONSTRUCTOR_TRACE
  std::cerr << "Destructor for ELlimitsTable\n";
#endif

}  // ~ELlimitsTable()


// ----------------------------------------------------------------------
// Methods invoked by the logger:
// ----------------------------------------------------------------------

void ELlimitsTable::setTableLimit( int n )  { tableLimit = n; }


bool ELlimitsTable::add( const ELextendedID & xid )  {

#ifdef ELlimitsTableATRACE
  std::cerr << "&&&--- adding to limits table: " << xid.id << '\n';
#endif

  ELmap_counts::iterator c = counts.find( xid );

  if ( c == counts.end() )  {  // no such entry yet

    #ifdef ELlimitsTableATRACE
    std::cerr << "&&&    no such entry yet in counts \n";
    #endif

    // if the counts table is "full", then this will never be rejected
    // and info will not be kept so why go through significant work: 
    if ( tableLimit > 0  && static_cast<int>(counts.size()) >= tableLimit ) {
      return true;
    }
    int lim;
    int ivl;
    int ts;
    ELmap_limits::iterator l = limits.find( xid.id );

    if ( l != limits.end() )  { // use limits previously established for this id
      lim = (*l).second.limit;
      ivl = (*l).second.interval;
      ts  = (*l).second.timespan;
      if ( lim < 0 )  {                                    // jvr 6/17/99
        lim = severityLimits[xid.severity.getLevel()];
        if ( lim < 0 )  {
          lim = wildcardLimit;
        }
      }
      if ( ivl < 0 )  {                                    
        ivl = severityIntervals[xid.severity.getLevel()];
        if ( ivl < 0 )  {
          ivl = wildcardInterval;			// mf 11/02/07
        }
      }
      if ( ts < 0 )  {
        ts = severityTimespans[xid.severity.getLevel()];
        if (ts < 0 )  {
          ts = wildcardTimespan;
        }
        limits[xid.id] = LimitAndTimespan( lim, ts );
      }
      #ifdef ELlimitsTableATRACE
      std::cerr << "&&&    Entry found in limits: limit = " << lim
           << " interval = " << ivl
           << " timespan = " << ts << '\n';
      #endif
      // change log 9/29/10:  Do not put this into limits table
   } else  {   // establish and use limits new to this id
      lim = severityLimits   [xid.severity.getLevel()];
      ivl = severityIntervals[xid.severity.getLevel()];
      ts  = severityTimespans[xid.severity.getLevel()];
      #ifdef ELlimitsTableATRACE
      std::cerr << "&&&    Limit taken from severityLimits: " << lim << '\n'
           << "&&&    Interval taken from severityLimits: " << ivl << '\n';
      #endif
      if ( lim < 0 )  {
        lim = wildcardLimit;
        #ifdef ELlimitsTableATRACE
        std::cerr << "&&&    Limit reset to wildcard limit: " << lim << '\n';
        #endif
      }
      if ( ivl < 0 )  {
        ivl = wildcardInterval;
        #ifdef ELlimitsTableATRACE
        std::cerr << "&&&    Interval reset to wildcard interval: " << ivl << '\n';
        #endif
      }
      #ifdef ELlimitsTableATRACE
      std::cerr << "&&&    Timespan taken from severityTimespans: " << ts << '\n';
      #endif
      if ( ts < 0 )  {
        ts = wildcardTimespan;
        #ifdef ELlimitsTableATRACE
        std::cerr << "&&&    timespan reset to wildcard timespan: " << ts << '\n';
        #endif
      }

      // change log 9/29/10 DO not save id's future limits:
    }

    // save, if possible, this xid's initial entry:
    if ( tableLimit < 0  || static_cast<int>(counts.size()) < tableLimit )
      counts[xid] = CountAndLimit( lim, ts, ivl );
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
    severityIntervals[lev] = -1;
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
