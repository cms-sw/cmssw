#ifndef _CondTools_Ecal_EcalCondHandler_
#define _CondTools_Ecal_EcalCondHandler_

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>


template <typename Payload, typename XMLTranslator>
class EcalCondHandler: 
   public popcon::PopConSourceHandler<Payload> {

 public:

  EcalCondHandler(const edm::ParameterSet & ps):
    xmlFileSource_(ps.getUntrackedParameter<std::string>("xmlFile")),
    since_(ps.getUntrackedParameter<long long>("since")){}

  virtual void getNewObjects(){

    EcalCondHeader   header;
    Payload * payload = new Payload;
    XMLTranslator::readXML(xmlFileSource_,header,*payload);
    popcon::PopConSourceHandler<Payload>::m_to_transfer.push_back(
		          	    std::make_pair(payload,since_));
  }

  virtual std::string id() const {return std::string("dummy");}

 private:
  
  std::string xmlFileSource_;
  long long since_; 
 

};

#endif

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b"
// End
