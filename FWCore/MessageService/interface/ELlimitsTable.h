#ifndef FWCore_MessageService_ELlimitsTable_h
#define FWCore_MessageService_ELlimitsTable_h


// ----------------------------------------------------------------------
//
// ELlimitsTable is a class holding two maps:  One listing various
//		 limits associated with each id, and one counting occurrences
//		 and tracking latest times of each type of extended-id
//		 (id + severity + module + subroutine + process).
//		 In addition, there is a table by severity of defaults,
//		 and a single wildcard default limit.
//
// The fundamental operation is
//
//	bool add( const ELextendedID & )
//
// which checks if the extended id is in the main map.  If it is not, it
// looks for the specified limit (by id, then severity, then wildcard) and
// cretes an entry in the main map for this extended id.  The count is
// incremented, (perhaps zero-ed first if the timespan was exceeded) and
// checked against its limit.  If the limit is not exceeded, OR is exceeded
// by 2**N times the limit, this returns true.
//
// Limits of -1 mean react to this error always.
// Limits of 0 in the auxilliary defaults indicate that the corresponding
// limit was not specified.
//
// 7/6/98 mf	Created file.
// 6/7/00 web	Reflect consolidation of ELdestination/X
// 6/14/00 web	Declare classes before granting friendship.
// 6/15/01 mf	Grant friendship to ELoutput so that faithful copying of
//		ELoutput destinations becomes possible.
// 5/16/06 mf   Added wildcardInterval member, and severityIntervals
//
// ----------------------------------------------------------------------


#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"
#include "FWCore/MessageLogger/interface/ELmap.h"

namespace edm {       
namespace service {       


// ----------------------------------------------------------------------
// Prerequisite class:
// ----------------------------------------------------------------------

class ELdestination;
class ELoutput;


// ----------------------------------------------------------------------
// ELlimitsTable:
// ----------------------------------------------------------------------

class ELlimitsTable  {

  friend class ELdestination;
  friend class ELoutput;

public:

  ELlimitsTable();
  ~ELlimitsTable();

// -----  Methods invoked by the destination under impetus from the logger:
//
public:
  bool add( const ELextendedID & xid );
  void setTableLimit( int n );

// -----  Control methods invoked by the framework:
//
public:

  void wipe();  // Clears everything -- counts and limits established.
  void zero();  // Clears only counts.

  void setLimit   ( const ELstring        & id,  int n        );
  void setLimit   ( const ELseverityLevel & sev, int n        );
  void setInterval( const ELstring        & id,  int interval );
  void setInterval( const ELseverityLevel & sev, int interval );
  void setTimespan( const ELstring        & id,  int n        );
  void setTimespan( const ELseverityLevel & sev, int n        );

  ELlimitsTable & operator= (const ELlimitsTable & t);

// -----  Tables and auxilliary private data:
//
protected:

  int severityLimits   [ELseverityLevel::nLevels];
  int severityTimespans[ELseverityLevel::nLevels];
  int severityIntervals[ELseverityLevel::nLevels];
  int wildcardLimit;
  int wildcardInterval;
  int wildcardTimespan;

  int tableLimit;
  ELmap_limits limits;
  ELmap_counts counts;

};  // ELlimitsTable


// ----------------------------------------------------------------------


}        // end of namespace service
}        // end of namespace edm


#endif // FWCore_MessageService_ELlimitsTable_h
