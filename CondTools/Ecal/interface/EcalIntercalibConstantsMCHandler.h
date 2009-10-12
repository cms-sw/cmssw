/**
   \file
   Declaration of class InterlcalibConstantsMCHandler

   \author Stefano ARGIRO
   \version $Id: EcalInterlcalibConstantsMCHandler.h,v 1.1 2008/11/14 15:46:05 argiro Exp $
   \date 09 Sep 2008
*/

#ifndef _CondToolsEcal_EcalIntercalibConstantsMCHandler_h_
#define _CondToolsEcal_EcalIntercalibConstantsMCHandler_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include <string>

static const char CVSId_CondToolsEcal_EcalIntercalibConstantsMCHandler[] = 
"$Id: EcalInterlcalibConstantsMCHandler.h,v 1.1 2008/11/14 15:46:05 argiro Exp $";

  /**
     \class EcalIntercalibConstantsMCHandler EcalIntercalibConstantsMCHandler.h "/EcalIntercalibConstantsMCHandler.h"

     \brief popcon application to store IntercalibConstants Records
            using XML tools

     \author Stefano ARGIRO
     \date 09 Sep 2008
  */
  class EcalIntercalibConstantsMCHandler:
        public popcon::PopConSourceHandler<EcalIntercalibConstantsMC> {

  public:
    EcalIntercalibConstantsMCHandler(const edm::ParameterSet & ps);
    virtual ~EcalIntercalibConstantsMCHandler();
    virtual void getNewObjects();
    virtual std::string id() const ;

  private:

    std::string xmlFileSource_;
    boost::int64_t since_;
    EcalIntercalibConstantsXMLTranslator translator_;
    EcalCondHeader header_;
    

  }; // EcalIntercalibConstantsMCHandler


#endif // _CondToolsEcal_EcalIntercalibConstantsMCHandler_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b"
// End
