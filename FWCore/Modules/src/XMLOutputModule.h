#ifndef FWCore_Modules_XMLOutputModule_h
#define FWCore_Modules_XMLOutputModule_h
// -*- C++ -*-
//
// Package:     Modules
// Class  :     XMLOutputModule
// 
/**\class XMLOutputModule XMLOutputModule.h FWCore/Modules/interface/XMLOutputModule.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Aug  4 20:45:42 EDT 2006
// $Id: XMLOutputModule.h,v 1.1 2006/08/25 14:36:07 chrjones Exp $
//

// system include files
#include <fstream>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"

// user include files

// forward declarations
namespace edm {
  class XMLOutputModule : public OutputModule {

   public:
      XMLOutputModule(edm::ParameterSet const&);
      virtual ~XMLOutputModule();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void write(EventPrincipal const& e);
      virtual void endLuminosityBlock(LuminosityBlockPrincipal const&){}
      virtual void endRun(RunPrincipal const&){}

      XMLOutputModule(XMLOutputModule const&); // stop default

      const XMLOutputModule& operator=(XMLOutputModule const&); // stop default

      // ---------- member data --------------------------------
      std::ofstream stream_;
      std::string indentation_;
  };
}

#endif
