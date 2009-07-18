/**
   \file
   Implementation of class EcalIntercalibErrorsHandler

   \author Stefano ARGIRO
   \version $Id: EcalIntercalibErrorsHandler.cc,v 1.3 2008/11/06 15:23:23 argiro Exp $
   \date 09 Sep 2008
*/

static const char CVSId[] = "$Id: EcalIntercalibErrorsHandler.cc,v 1.3 2008/11/06 15:23:23 argiro Exp $";

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondTools/Ecal/interface/EcalIntercalibErrorsHandler.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include <utility>

EcalIntercalibErrorsHandler::
~EcalIntercalibErrorsHandler(){}

EcalIntercalibErrorsHandler::
EcalIntercalibErrorsHandler(const edm::ParameterSet & ps):
  xmlFileSource_(ps.getUntrackedParameter<std::string>("xmlFile")),
  since_(ps.getUntrackedParameter<boost::int64_t>("since"))
{
}

void EcalIntercalibErrorsHandler::getNewObjects(){
  
  EcalCondHeader          header;

  // we allocate on the heap here, knowing that popcon will
  // take care of deleting the payload
  EcalIntercalibConstants *dummy     = new EcalIntercalibConstants ;
  EcalIntercalibErrors    *payload   = new EcalIntercalibErrors;

  // our file contains errors and constants, but popcon does one
  // payload type at the time, so we have to use a dummy
  translator_.readXML(xmlFileSource_,header,*dummy,*payload);
  
  delete dummy;
  
  //cond::Time_t snc = header.since_;
  //for now we don't make use of the xml header to read the since
  //but rely on the one passed from parameter set
  m_to_transfer.push_back(std::make_pair(payload,since_));

}

std::string EcalIntercalibErrorsHandler::id() const{
  
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
