/**
   \file
   Declaration of class FloatCondObjectContainerHandler

   \author Stefano ARGIRO
   \version $Id: EcalFloatCondObjectContainerHandler.h,v 1.2 2009/11/09 21:00:40 wmtan Exp $
   \date 09 Sep 2008
*/

#ifndef _CondToolsEcal_EcalFloatCondObjectContainerHandler_h_
#define _CondToolsEcal_EcalFloatCondObjectContainerHandler_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

static const char CVSId_CondToolsEcal_EcalFloatCondObjectContainerHandler[] = 
"$Id: EcalFloatCondObjectContainerHandler.h,v 1.2 2009/11/09 21:00:40 wmtan Exp $";

  /**
     \class EcalFloatCondObjectContainerHandler EcalFloatCondObjectContainerHandler.h "/EcalFloatCondObjectContainerHandler.h"

     \brief popcon application to store FloatCondObjectContainer Records
            using XML tools

     \author Stefano ARGIRO
     \date 09 Sep 2008
  */
  class EcalFloatCondObjectContainerHandler:
        public popcon::PopConSourceHandler<EcalFloatCondObjectContainer> {

  public:
    EcalFloatCondObjectContainerHandler(const edm::ParameterSet & ps);
    virtual ~EcalFloatCondObjectContainerHandler();
    virtual void getNewObjects();
    virtual std::string id() const ;

  private:

    std::string xmlFileSource_;
    long long since_;

    EcalCondHeader header_;
    

  }; // EcalFloatCondObjectContainerHandler


#endif // _CondToolsEcal_EcalFloatCondObjectContainerHandler_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "scram b"
// End
