#ifndef FWCore_MessageService_ELdestControl_h
#define FWCore_MessageService_ELdestControl_h


// ----------------------------------------------------------------------
//
// ELdestControl  is a handle class whose purpose is to dispatch orders
//                from the framework to an ELdestination, without
//                allowing the framework to destruct that instance of the
//                destination (which would be a disasterous way to foul up).
//                The ELadministrator creates an ELdestControl handle
//                to its own copy whenever an ELdestination is attached.
//
// 7/5/98  mf	Created file.
// 6/16/99 jvr  Allows suppress/include options on destinations
// 7/2/99  jvr	Added separate/attachTime, Epilogue, and Serial options
// 6/7/00  web	Reflect consolidation of ELdestination/X; add
//		filterModule()
// 10/04/00 mf  add excludeModule()
// 01/15/00 mf  mods to give user control of line length
// 03/13/01 mf  mod to give user control of hex trigger and 
//		statisticsMap() method
// 04/04/01 mf  add ignoreMOdule and respondToModule
// 10/17/01 mf  add setTableLimit which had been omitted
// 10/18/01 mf  Corrected default in summary title =0 to empty string
//  6/23/03 mf  changeFile() and flush() 
//  6/19/08 mf  summaryForJobReport()
//
// ----------------------------------------------------------------------


#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELmap.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include "boost/shared_ptr.hpp"


namespace edm {       
namespace service {       

// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ELdestination;

// ----------------------------------------------------------------------
// ELdestControl:
// ----------------------------------------------------------------------

class ELdestControl  {

public:
  ELdestControl( boost::shared_ptr<ELdestination> dest );
  ELdestControl();
  ~ELdestControl();

  // -----  Behavior control methods invoked by the framework:
  //
  ELdestControl & setThreshold( const ELseverityLevel & sv );
  ELdestControl & setTraceThreshold( const ELseverityLevel & sv );
  ELdestControl & setLimit( const ELstring & s, int n );
  ELdestControl & setLimit( const ELseverityLevel & sv, int n );
  ELdestControl & setInterval( const ELstring & s, int interval );
  ELdestControl & setInterval( const ELseverityLevel& sv, int interval);
  ELdestControl & setTimespan( const ELstring& s, int n );
  ELdestControl & setTimespan( const ELseverityLevel & sv, int n );

  ELdestControl & setTableLimit( int n );

  // -----  Select output format options:
  //
  void suppressText();           void includeText();  // $$ jvr
  void suppressModule();         void includeModule();
  void suppressSubroutine();     void includeSubroutine();
  void suppressTime();           void includeTime();
  void suppressContext();        void includeContext();
  void suppressSerial();         void includeSerial();
  void useFullContext();         void useContext();
  void separateTime();           void attachTime();
  void separateEpilogue();       void attachEpilogue();
  void noTerminationSummary();
  int  setLineLength(int len);	 int  getLineLength() const;

  void filterModule    ( ELstring const & moduleName );
  void excludeModule   ( ELstring const & moduleName );
  void respondToModule ( ELstring const & moduleName );
  void ignoreModule    ( ELstring const & moduleName );

  ELdestControl & clearSummary();
  ELdestControl & wipe();
  ELdestControl & zero();

  ELdestControl & setPreamble( const ELstring & preamble );
  ELdestControl & setNewline( const ELstring & newline );

  // -----  Active methods invoked by the framework:
  //
  void summary( ELdestControl & dest, const char * title="" );
  void summary( std::ostream  & os  , const char * title="" );
  void summary( ELstring      & s   , const char * title="" );
  void summary( );
  void summaryForJobReport( std::map<std::string, double> & sm);

  std::map<ELextendedID , StatsCount> statisticsMap() const;

  bool log( edm::ErrorObj & msg );  // Backdoor to log a formed message
                                            // to only this destination.
				       
  void changeFile (std::ostream & os);
  void changeFile (const ELstring & filename);
  void flush(); 				       
  
  // -----  Helper methods invoked by other ErrorLogger classes

  void summarization( const ELstring & title
                            , const ELstring & sumLines
                            );

  ELstring getNewline() const;

  // -----  Data implementing the trivial handle pattern:
  //
private:
  boost::shared_ptr<ELdestination> d;

};  // ELdestControl


// ----------------------------------------------------------------------


}        // end of namespace service
}        // end of namespace edm


#endif // FWCore_MessageService_ELdestControl_h
