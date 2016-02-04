#ifndef FWCore_MessageService_ELrecv_h
#define FWCore_MessageService_ELrecv_h


// ----------------------------------------------------------------------
//
// ELrecv	is a METHOD (not a class!) which the receiver of packets on
// 		the client side is to call when it gets an error message packet
//		sent by ELsend.
//
//		The choice of making this a global METHOD rather than a functor
//		class may in some cases make it easier to link to this from
//		non-C++ code.
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
// (2)	The client process must have a method ELsend (int n, char* data) which
//	moves n bytes of data to the server.
//		If namepaces are enabled, ELsend is in namdspace zmel.
//
// (3)	The server must be set up such that whenever one of these chunks of
//	data is moved to it, it calls ELrecv (n, data).
//
// (4)	The server process sets up its own ELadministrator, and attaches
//	whatever destinations it wants to have.
//
// (*)	The effect of an error message on the client is that ELrecv unravels
//	the data into the individual items that were supplied, and issues an
//	error message which is identical to the error message that was issued
//	on the client.  Thus all these messages end up amalgamated.
//
// 2/29/00 mf	Created file
// 3/16/00 mf	Added signature with localModule name.
//
// ----------------------------------------------------------------------

#include "FWCore/MessageLogger/interface/ELstring.h"

namespace edm {       
namespace service {       

void ELrecv ( int nbytes, const char * data );
void ELrecv ( int nbytes, const char * data, ELstring localModule );

}        // end of namespace service
}        // end of namespace edm


#endif // FWCore_MessageService_ELrecv_h
