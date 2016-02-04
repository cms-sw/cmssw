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
#include "FWCore/MessageService/interface/ELset.h"

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"

namespace edm {       
namespace service {       

// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ELdestControl;
class ELadministrator;


// ----------------------------------------------------------------------
// ELdestination:
// ----------------------------------------------------------------------

class ELdestination  {

  friend class ELadministrator;
  friend class ELdestControl;

public:

  ELdestination();
  virtual ~ELdestination();

  // -----  Methods invoked by the ELadministrator:
  //
public:
  virtual ELdestination * clone() const = 0;
  virtual bool log( const edm::ErrorObj & msg );

  virtual void summarization(
    		const edm::ELstring & title,
   		const edm::ELstring & sumLines );

  virtual ELstring getNewline() const;

  virtual void finish();

  // -----  Methods invoked through the ELdestControl handle:
  //
protected:
  virtual void clearSummary();
  virtual void wipe();
  virtual void zero();
  virtual void filterModule( ELstring const & moduleName );
  virtual void excludeModule( ELstring const & moduleName );
  virtual void ignoreModule( ELstring const & moduleName );
  virtual void respondToModule( ELstring const & moduleName );
  virtual bool thisShouldBeIgnored(const ELstring & s) const;

  virtual void summary( ELdestControl & dest, const ELstring & title="" );
  virtual void summary( std::ostream  & os  , const ELstring & title="" );
  virtual void summary( ELstring      & s   , const ELstring & title="" );
  virtual void summary( );
  virtual void summaryForJobReport(std::map<std::string, double> & sm);

  virtual void setTableLimit( int n );

  virtual std::map<ELextendedID,StatsCount> statisticsMap() const;

  virtual void changeFile (std::ostream & os);
  virtual void changeFile (const ELstring & filename);
  virtual void flush(); 				       

  // -----  Select output format options:
  //
private:
  virtual void suppressText();           virtual void includeText(); // $$ jvr
  virtual void suppressModule();         virtual void includeModule();
  virtual void suppressSubroutine();     virtual void includeSubroutine();
  virtual void suppressTime();           virtual void includeTime();
  virtual void suppressContext();        virtual void includeContext();
  virtual void suppressSerial();         virtual void includeSerial();
  virtual void useFullContext();         virtual void useContext();
  virtual void separateTime();           virtual void attachTime();
  virtual void separateEpilogue();       virtual void attachEpilogue();
  virtual void noTerminationSummary();
  virtual int  setLineLength(int len);	 virtual int  getLineLength() const;

  // -----  Data affected by methods of the ELdestControl handle:
  //
protected:
  ELseverityLevel threshold;
  ELseverityLevel traceThreshold;
  ELlimitsTable   limits;
  ELstring        preamble;
  ELstring        newline;
  ELstring        indent;
  int		  lineLength;
  bool            ignoreMostModules;
  ELset_string    respondToThese;
  bool            respondToMostModules;
  ELset_string    ignoreThese;
					// Fix $001 2/13/01 mf
#ifndef DEFECT_NO_STATIC_CONST_INIT
  static const int defaultLineLength = 80;
#else
  static const int defaultLineLength;	
#endif

  // -----  Verboten methods:
  //
private:
  ELdestination( const ELdestination & orig );
  ELdestination& operator= ( const ELdestination & orig );

};  // ELdestination

struct close_and_delete {
  void operator()(std::ostream* os) const;
};

}        // end of namespace service
}        // end of namespace edm


#endif  // FWCore_MessageService_ELdestination_h
