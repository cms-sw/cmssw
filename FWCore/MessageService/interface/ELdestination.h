#ifndef FWCore_MessageService_ELdestination_h
#define FWCore_MessageService_ELdestination_h


// ----------------------------------------------------------------------
//
// ELdestination   is a virtual class defining the interface to a
//		   destination.  Concrete classes derived from this include
//		   ELoutput and ELstatistics.  The ELadministrator owns
//		   a list of ELdestination* as well as the objects those
//		   list elements point to.
//
// 7/5/98 mf	Created file.
// 6/16/99 jvr  Allows suppress/include options on destinations
// 7/1/99 mf	Forward-declared ELdestControl for strict C++ (thanks cg).
// 7/2/99  jvr  Added separate/attachTime, Epilogue, and Serial options
// 12/20/99 mf  Added virtual destructor.
// 6/7/00  web	Consolidated ELdestination/X; add filterModule()
// 6/14/00 web	Declare classes before granting friendship.
// 10/4/00 mf   Add excludeModule
// 1/15/01 mf	setLineLength()
// 2/13/01 mf	fix written by pc to accomodate NT problem with 
//		static init { $001$ }.  Corresponding fix is in .cc file.
// 3/13/01 mf	statisticsMap()
// 04/04/01 mf  add ignoreMOdule and respondToModule
// 6/23/03 mf   changeFile() and flush() 
// 1/10/06 mf	finish
// 6/19/08 mf   summaryForJobReport()
//
// ----------------------------------------------------------------------

#include "FWCore/MessageService/interface/ELlimitsTable.h"

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"

#include <unordered_set>
#include <string>

namespace edm {       
namespace service {       

// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ELadministrator;


// ----------------------------------------------------------------------
// ELdestination:
// ----------------------------------------------------------------------

class ELdestination  {

  friend class ELadministrator;

public:

  ELdestination();
  virtual ~ELdestination();

  // -----  Methods invoked by the ELadministrator:
  //
public:
  virtual bool log( const edm::ErrorObj & msg );

  virtual ELstring getNewline() const;

  virtual void finish();

  // -----  Behavior control methods invoked by the framework:
  //
  void setThreshold( const ELseverityLevel & sv );
  void setTraceThreshold( const ELseverityLevel & sv );
  void setLimit( const ELstring & s, int n );
  void setLimit( const ELseverityLevel & sv, int n );
  void setInterval( const ELstring & s, int interval );
  void setInterval( const ELseverityLevel& sv, int interval);
  void setTimespan( const ELstring& s, int n );
  void setTimespan( const ELseverityLevel & sv, int n );

  // -----  Select output format options:
  //
  virtual void suppressText();           virtual void includeText(); // $$ jvr
  virtual void suppressModule();         virtual void includeModule();
  virtual void suppressSubroutine();     virtual void includeSubroutine();
  virtual void suppressTime();           virtual void includeTime();
  virtual void suppressContext();        virtual void includeContext();
  virtual void suppressSerial();         virtual void includeSerial();
  virtual void useFullContext();         virtual void useContext();
  virtual void separateTime();           virtual void attachTime();
  virtual void separateEpilogue();       virtual void attachEpilogue();
  virtual int  setLineLength(int len);	 virtual int  getLineLength() const;

  virtual void wipe();
  virtual void zero();
  virtual void filterModule( ELstring const & moduleName );
  virtual void excludeModule( ELstring const & moduleName );
  virtual void ignoreModule( ELstring const & moduleName );
  virtual void respondToModule( ELstring const & moduleName );
  virtual bool thisShouldBeIgnored(const ELstring & s) const;

  virtual void setTableLimit( int n );

  virtual void changeFile (std::ostream & os);
  virtual void changeFile (const ELstring & filename);
  virtual void flush(); 				       


protected:
  ELseverityLevel threshold;
  ELseverityLevel traceThreshold;
  ELlimitsTable   limits;
  ELstring        preamble;
  ELstring        newline;
  ELstring        indent;
  int		  lineLength;
  bool            ignoreMostModules;
  std::unordered_set<std::string>  respondToThese;
  bool            respondToMostModules;
  std::unordered_set<std::string> ignoreThese;
					// Fix $001 2/13/01 mf
#ifndef DEFECT_NO_STATIC_CONST_INIT
  static const int defaultLineLength = 80;
#else
  static const int defaultLineLength;	
#endif

  // -----  Verboten methods:
  //
private:
  ELdestination( const ELdestination & orig ) = delete;
  ELdestination& operator= ( const ELdestination & orig ) = delete;

};  // ELdestination

struct close_and_delete {
  void operator()(std::ostream* os) const;
};

}        // end of namespace service
}        // end of namespace edm


#endif  // FWCore_MessageService_ELdestination_h
