// ----------------------------------------------------------------------
//
// ELmap.cc
//
// Change History:
//   99-06-10	mf	correction in sense of comparison between timespan
//			and diff (now, lastTime)
//		mf	ELcountTRACE made available
//   99-06-11	mf	Corrected logic for suppressing output when n > limit
//			but not but a factor of 2**K
//
// ----------------------------------------------------------------------


#include "FWCore/MessageLogger/interface/ELmap.h"

// Possible traces
// #define ELcountTRACE
// #define ELmapDumpTRACE


namespace edm {       


// ----------------------------------------------------------------------
// LimitAndTimespan:
// ----------------------------------------------------------------------

LimitAndTimespan::LimitAndTimespan( int lim, int ts )
: limit   ( lim )
, timespan( ts )
{ }


// ----------------------------------------------------------------------
// CountAndLimit:
// ----------------------------------------------------------------------

CountAndLimit::CountAndLimit( int lim, int ts )
: n         ( 0 )
, aggregateN( 0 )
, lastTime  ( time(0) )
, limit     ( lim )
, timespan  ( ts )
{ }


bool  CountAndLimit::add()  {

  time_t  now = time(0);

#ifdef ELcountTRACE
  std::cout << "&&&--- CountAndLimit::add \n";
  std::cout << "&&&    Time now  is " << now << "\n";
  std::cout << "&&&    Last time is " << lastTime << "\n";
  std::cout << "&&&    timespan  is " << timespan << "\n";
  std::cout << "&&&    difftime  is " << difftime( now, lastTime ) << "\n"
				<< std::flush;
#endif

  // Has it been so long that we should restart counting toward the limit?
  if ( (timespan >= 0)
	    &&
	(difftime(now, lastTime) >= timespan) )  {
     n = 0;
  }

  lastTime = now;

  ++n;  ++aggregateN;

#ifdef ELcountTRACE
  std::cout << "&&&    n is " << n << "-- limit is " << limit << "\n";
#endif

  if ( limit == 0 ) return false; 	// Zero limit - never react to this	
  if ( limit < 0  ) return true;   	// No limit - always react
  if ( n <= limit ) return true;  	// Under the limit so react

  // Now we are over the limit - have we exceeded limit by 2^N * limit?
  long  diff = n - limit;
  long  r = diff/limit;
  if ( r*limit != diff ) return false; 	// Not a multiple of limit - don't react
  if ( r == 1 )          return true;  	// Exactly twice limit - react
  while ( r > 1 )  {
    if ( (r & 1) != 0 )  return false;	// Not 2**n times limit - don't react
    r >>= 1;
  }
  			 return true; 	// If you never get an odd number till
					// one, r is 2**n so react

}  // add()


// ----------------------------------------------------------------------
// StatsCount:
// ----------------------------------------------------------------------

StatsCount::StatsCount()
: n          ( 0 )
, aggregateN ( 0 )
, ignoredFlag( false )
, context1   ( "" )
, context2   ( "" )
, contextLast( "" )
{ }


void  StatsCount::add( const ELstring & context, bool reactedTo )  {

  ++n;  ++aggregateN;

  ( (1 == n) ? context1
  : (2 == n) ? context2
  :            contextLast
  )                        = ELstring( context, 0, 16 );

  if ( ! reactedTo )
    ignoredFlag = true;

}  // add()


// ----------------------------------------------------------------------

#ifdef ELmapDumpTRACE
// ----------------------------------------------------------------------
// Global Dump methods (useful for debugging)
// ----------------------------------------------------------------------

#include <sstream>
#include <string.h>

char *  ELmapDump ( ELmap_limits m )  {

  std::ostringstream s;
  s << "**** ELmap_limits Dump **** \n";

  ELmap_limits::const_iterator i;
  for ( i = m.begin();  i != m.end();  ++i )  {
    LimitAndTimespan lt = (*i).second;
    s << "     " << (*i).first << ":  " << lt.limit << ", " <<
		lt.timespan << "\n";
  }
  s << "--------------------------------------------\n";

  char *  dump = new char[s.str().size()+1];
  strcpy( dump, s.str().c_str() );

  return dump;

}
#endif

} // end of namespace edm  */

