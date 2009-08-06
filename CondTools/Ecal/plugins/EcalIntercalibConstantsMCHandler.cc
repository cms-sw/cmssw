/**
   \file
   Implementation of class EcalIntercalibConstantsMCHandler

   \author Stefano ARGIRO
   \version $Id: EcalIntercalibConstantsMCHandler.cc,v 1.1 2008/11/14 15:46:04 argiro Exp $
   \date 09 Sep 2008
*/

static const char CVSId[] = "$Id: EcalIntercalibConstantsMCHandler.cc,v 1.1 2008/11/14 15:46:04 argiro Exp $";

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsMCHandler.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include <utility>

EcalIntercalibConstantsMCHandler::
~EcalIntercalibConstantsMCHandler(){}

EcalIntercalibConstantsMCHandler::
EcalIntercalibConstantsMCHandler(const edm::ParameterSet & ps):
  xmlFileSource_(ps.getUntrackedParameter<std::string>("xmlFile")),
  since_(ps.getUntrackedParameter<boost::int64_t>("since"))
{
}

void EcalIntercalibConstantsMCHandler::getNewObjects(){
  
  EcalCondHeader          header;

  // we allocate on the heap here, knowing that popcon will
  // take care of deleting the payload
  EcalIntercalibConstantsMC *payload = new EcalIntercalibConstants ;
  EcalIntercalibErrors      *dummy   = new EcalIntercalibErrors;

  // our file contains errors as well, but popcon does one
  // payload type at the time, so we have to use a dummy
  translator_.readXML(xmlFileSource_,header,*payload,*dummy);
  
  delete dummy;
  
  //cond::Time_t snc = header.since_;
  //for now we don't make use of the xml header to read the since
  //but rely on the one passed from parameter set
  
  m_to_transfer.push_back(std::make_pair(payload,since_));

}

std::string EcalIntercalibConstantsMCHandler::id() const{
  
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
