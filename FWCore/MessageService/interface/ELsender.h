#ifndef FWCore_MessageService_ELsender_h
#define FWCore_MessageService_ELsender_h

// ----------------------------------------------------------------------
//
// ELsender 	is a pure virtual (abstract) class to hold the object
//		telling how a message is transported from the client to the
//		server.  That is, an ELcollected destination constructed
//		from ELcollected ( const ELsender & mySender ) will call
//		mysender.send(nbytes,data).
//
//		The user-defined transport mechanism (at the client end and
//		in the server process) must be responsible receiving that data
//		and calling ELrecv(nbytes, data) or doing errlog(nbytes,data).
//
//		A closely related global method is ELsend, defined in
//		ELcollected.h.  That is used for ELcollecteds which were
//		not instatiated with an ELsender supplied.
//
// Usage:
//
//	class MySender : public ELsender  {
//	  public:
//	  void send (int nbytes, const char * data)  {
//		// ... code that moves the nbytes of data to the server
//	  }
//	};
//
//   // MySender could be more complex if necessary.  For example it could be
//   // constructed with some sort of address specifying where the server is.
//
//	MySender ms;
//	ELcollected remoteLog (ms);
//
// 3/16/00 mf	FIle created.
// 8/30/02 web	Add virtual destructor; fix comment on code guard #endif
//
// ----------------------------------------------------------------------

namespace edm {       
namespace service {       

// ----------------------------------------------------------------------
// ELsender:
// ----------------------------------------------------------------------

class ELsender  {

public:

  // -----  Destructor:

  virtual ~ELsender()  {}


  // -----  Transmit a message to a server:

  virtual void send (int nbytes, const char * data) = 0;

  // -----  Capture of copy of this object:
  //		This is here to avoid any chance of tough-to-decipher
  //		misbehavior due to a user providing the sender but letting
  //		it go out of scope before it is used.

  virtual ELsender * clone()            const = 0;

};  // ELsender


}        // end of namespace service
}        // end of namespace edm


#endif // FWCore_MessageService_ELsender_h
