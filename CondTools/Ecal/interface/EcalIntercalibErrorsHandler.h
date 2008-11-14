/**
   \file
   Declaration of class IntercalibErrorsHandler

   \author Stefano ARGIRO
   \version $Id: EcalIntercalibErrorsHandler.h,v 1.3 2008/11/06 15:23:51 argiro Exp $
   \date 09 Sep 2008
*/

#ifndef _CondToolsEcal_EcalIntercalibErrorsHandler_h_
#define _CondToolsEcal_EcalIntercalibErrorsHandler_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include <string>

static const char CVSId_CondToolsEcal_EcalIntercalibErrorsHandler[] = 
"$Id: EcalIntercalibErrorsHandler.h,v 1.3 2008/11/06 15:23:51 argiro Exp $";

  /**
     \class EcalIntercalibErrorsHandler EcalIntercalibErrorsHandler.h "/EcalIntercalibErrorsHandler.h"

     \brief popcon application to store IntercalibConstants Records
            using XML tools

     \author Stefano ARGIRO
     \date 09 Sep 2008
  */
  class EcalIntercalibErrorsHandler:
        public popcon::PopConSourceHandler<EcalIntercalibErrors> {

  public:
    EcalIntercalibErrorsHandler(const edm::ParameterSet & ps);
    virtual ~EcalIntercalibErrorsHandler();
    virtual void getNewObjects();
    virtual std::string id() const ;

  private:

    std::string xmlFileSource_;
    boost::int64_t since_;
    EcalIntercalibConstantsXMLTranslator translator_;
    EcalCondHeader header_;
    

  }; // EcalIntercalibErrorsHandler


#endif // _CondToolsEcal_EcalIntercalibErrorsHandler_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b"
// End
