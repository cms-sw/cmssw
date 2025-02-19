#ifndef MessageLogger_ELmap_h
#define MessageLogger_ELmap_h


// ----------------------------------------------------------------------
//
// ELmap.h      Provides a map class with the semantics of std::map.
//              Customizers may substitute for this class to provide either
//              a map with a different allocator, or whatever else.
//
//	We typedef an individual type for each of these maps since
//	the syntax
//		typedef map ELmap;
//		ELlist<ELextededID, Counts, ELextendedID::less> table;
//	may or may not be valid C++, and if valid probably won't work
//	everywhere.
//
//	The key types are common enough types (strings and extended id's);
//	the data types are peculiar to each type of map.  We have made the
//	design choice of grouping all maps in this one file, so that if
//	a customizer needs to do something else with maps, all the slop is
//	in one place.
//
//	The drawback is that all the classes that depend on these maps must
//	include ELmap.h, which pulls in not only the maps and structs they
// 	need but those anybody else needs.  Fortunately, only two classes
//	use maps at all:  ELlimitsTable and the ELstatistics destination.
//	So this drawback is very slight.
//
// The elements of map semantics which ErrorLogger code rely upon are
// listed in ELmap.semantics.
//
// ----------------------------------------------------------------------


#include <map>

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"


namespace edm {       


// ----------------------------------------------------------------------


class LimitAndTimespan  {

public:

  int limit;
  int timespan;
  int interval;

  LimitAndTimespan( int lim = -1, int ts = -1, int ivl = -1 );

};  // LimitAndTimespan


class CountAndLimit  {

public:

  int    n;
  int    aggregateN;
  time_t lastTime;
  int    limit;
  int    timespan;
  int    interval;
  int    skipped;

  CountAndLimit( int lim = -1, int ts = -1, int ivl = -1 );
  bool add();

};  // CountAndLimit


class StatsCount  {

public:

  int      n;
  int      aggregateN;
  bool     ignoredFlag;
  ELstring context1;
  ELstring context2;
  ELstring contextLast;

  StatsCount();
  void add( const ELstring & context, bool reactedTo );

};  // StatsCount


// ----------------------------------------------------------------------


typedef std::map< ELstring     , LimitAndTimespan > ELmap_limits;

typedef std::map< ELextendedID , CountAndLimit    > ELmap_counts;

typedef std::map< ELextendedID , StatsCount       > ELmap_stats;

// See ELseverityLevel.cc for another map:  ELmap_sevTran

// ----------------------------------------------------------------------


}        // end of namespace edm


#endif // MessageLogger_ELmap_h
