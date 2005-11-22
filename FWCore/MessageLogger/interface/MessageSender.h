#ifndef FWCore_MessageLogger_MessageSender_h
#define FWCore_MessageLogger_MessageSender_h

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"


namespace edm
{

class MessageSender
{
public:
  // ---  birth/death:
  MessageSender( ELseverityLevel const & sev, ELstring const & id );
  ~MessageSender();

  // ---  stream out the next part of a message:
  template< class T >
    MessageSender &
    operator<< ( T const & t )
  {
    (*errorobj_p) << t;
    return *this;
  }

private:
  // no copying:
  MessageSender( MessageSender const & );
  void operator = ( MessageSender const & );

  // data:
  ErrorObj *  errorobj_p;

};  // MessageSender


}  // namespace edm


#endif  // FWCore_MessageLogger_MessageSender_h
