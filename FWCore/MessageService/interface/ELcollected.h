#ifndef FWCore_MessageService_ELcollected_h
#define FWCore_MessageService_ELcollected_h


// ----------------------------------------------------------------------
//
// ELcollected	is a subclass of ELdestination representing a
//		destination designed for collection of messages
//		from multiple processes.
//
// Usage:
//
// (0) 	Define "clients" and "server" such that multiple clients each run
//	user code which will issue ErrorLogger messages, and a single server
//	is to amalgamated those messages into a single flow with unified
//	statistics output.
//
// (1) 	Each client connects (one or more) ELcollected destinations to the
//	ELadministrator.  These can be controlled in terms of filtering,
//	throttling and so forth.  The semantics are like those for ELoutput.
//
// (2)  When a ELcollected destination is instantiated, the constructor
//	takes as an argument an object of a type derived from ELsender.
//	ELsender is an abstract class containing a pure virtual method
//	send (int n, const char* data).  The derived class provides a definition
//	of send(n, data) which moves n bytes of data to the server.
//			ELsender also has a pure virtual clone() method;
//			provision of this follows a boiler-plate we provide.
//	One advantage of setting up an ELsender object, rather than making
//	the user provide a global-scope function ELsend, is flexibility:
//	It makes it trivial to set up two ELcollecteds that send their data to
//	two different places or even use two distinct transport mechanisms.
//
// (3)	The server process must set up its own ELadministrator, and attach
//	whatever destinations it wants to have, such as ELoutput and
//	ELstatistics destinations, in the usual manner.
//
// (4a)	The server can be set up such that it has an ErrorLog errlog and
//	whenever one of these chunks of data is moved to it, it does
//	errlog (n, data).
//		This is a natural syntax for a framework used to using
//		errlog to log messages.  If the server process is also doing
//		other things that might issue error messages, this syntax
//		has the advantageous feature that whatever module name is
//		assinged to this errlog, will be prepended to the module name
//		sent as part of the message.
//
// (4b)	Alternatively, the server cam be set up such that whenever one of these
//	chunks of data is moved to it, it calls ELrecv (n, data).
//		This may be marginally more convenient.
//
// (*)	The effect of an error message on the client is that when the data
//	is moved to the server, and that causes ELrecv (or errlog(nbytes,data))
//	to be called, the data is unraveled into the individual items that were
//	supplied, and an error message is issued which is identical to the
//	error message that was issued on the client.  Thus all these messages
//	end up amalgamated on the server destination(s).
//
// 2/22/00 mf	Created file, based mainly on ELoutput.h
// 3/16/00 mf	Mods to support ELsender operation.
// 6/7/00 web	Reflect consolidation of ELdestination/X and ELoutput/X
//		and consolidate ELcollected/X.
// 6/14/00 web	Declare classes before granting friendship.
//
// ----------------------------------------------------------------------

#include "FWCore/MessageService/interface/ELoutput.h"
#include "FWCore/MessageService/interface/ELsender.h"

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELextendedID.h"

#include "boost/shared_ptr.hpp"

namespace edm {       
namespace service {       


// ----------------------------------------------------------------------
// prerequisite classes:
// ----------------------------------------------------------------------

class ELoutput;
class ELdestControl;


// ----------------------------------------------------------------------
// ELcollected:
// ----------------------------------------------------------------------

class ELcollected : public ELoutput  {

  friend class ELdestControl;

public:

  // -----  Birth/death:
  //
  ELcollected( const ELsender & sender );
  ELcollected( const ELcollected & orig );
  ELcollected( ELoutput * d);
  virtual ~ELcollected();

  // -----  Methods invoked by the ELadministrator:
  //
public:
  virtual
  ELcollected *
  clone() const;
  // Used by attach() to put the destination on the ELadministrators list
		//-| There is a note in Design Notes about semantics
		//-| of copying a destination onto the list:  ofstream
		//-| ownership is passed to the new copy.

  virtual bool log( const edm::ErrorObj & msg );

  // -----  Methods invoked through the ELdestControl handle:
  //
protected:
    // trivial clearSummary(), wipe(), zero() from base class
    // trivial three summary(..) from base class

  // -----  Data fundamental to the ELcollected destination:
  //

  // -----  Data affected by methods of specific ELdestControl handle:
  //
protected:
    // ELcollected uses the generic ELdestControl handle

  // -----  Internal Methods -- Users should not invoke these:
  //
protected:
  void emitToken( const ELstring & s, bool nl=false );

  boost::shared_ptr<ELsender> sender;

  void intoBuf ( const ELstring & s );
  void emitXid ( const ELextendedID & xid  );

  ELstring buf;

private:
  ELcollected & operator=( const ELcollected & orig );  // verboten

};  // ELcollected


// ----------------------------------------------------------------------


}        // end of namespace service
}        // end of namespace edm


#endif // MessageService_ELcollected_h
