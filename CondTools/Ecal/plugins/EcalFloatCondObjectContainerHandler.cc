/**
   \file
   Implementation of class EcalFloatCondObjectContainerHandler

   \author Stefano ARGIRO
   \version $Id: EcalFloatCondObjectContainerHandler.cc,v 1.2 2009/11/06 11:32:53 fra Exp $
   \date 09 Sep 2008
*/

static const char CVSId[] = "$Id: EcalFloatCondObjectContainerHandler.cc,v 1.2 2009/11/06 11:32:53 fra Exp $";

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondTools/Ecal/interface/EcalFloatCondObjectContainerHandler.h"
#include "CondTools/Ecal/interface/EcalFloatCondObjectContainerXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include <utility>

EcalFloatCondObjectContainerHandler::
~EcalFloatCondObjectContainerHandler(){}

EcalFloatCondObjectContainerHandler::
EcalFloatCondObjectContainerHandler(const edm::ParameterSet & ps):
  xmlFileSource_(ps.getUntrackedParameter<std::string>("xmlFile")),
  since_(ps.getUntrackedParameter<long long>("since"))
{
}

void EcalFloatCondObjectContainerHandler::getNewObjects(){
  
  EcalCondHeader          header;

  // we allocate on the heap here, knowing that popcon will
  // take care of deleting the payload
  EcalFloatCondObjectContainer *payload = new EcalFloatCondObjectContainer ;

  EcalFloatCondObjectContainerXMLTranslator::readXML(xmlFileSource_,header,*payload);
  
  
  //cond::Time_t snc = header.since_;
  //for now we don't make use of the xml header to read the since
  //but rely on the one passed from parameter set
  
  m_to_transfer.push_back(std::make_pair(payload,since_));

}

std::string EcalFloatCondObjectContainerHandler::id() const{
  
  // We have to think if this is the right thing to do ...
  
  EcalCondHeader          header;
  xuti::readHeader(xmlFileSource_, header);
  return header.tag_;
}


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b"
// End:
