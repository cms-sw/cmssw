#ifndef Modules_SimRootErrors_h
#define Modules_SimRootErrors_h
// -*- C++ -*-
//
// Package:     Modules
// Class  :     SimRootErrors
// 
/**\class SimRootErrors SimRootErrors.h FWCore/Modules/src/SimRootErrors.h

 Description: prints out what data is contained within an Event at that point in the path

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep 19 11:49:35 CEST 2005
//

// system include files
#include <string>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

// forward declarations

class SimRootErrors : public edm::EDAnalyzer {
public:
   explicit SimRootErrors(const edm::ParameterSet&);
   ~SimRootErrors();
   
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endJob();

private:
      
   // ----------member data ---------------------------
   std::string indentation_;
   std::string verboseIndentation_;
   std::vector<std::string> moduleLabels_;
   bool        verbose_;  
   int         evno_;
   std::map<std::string, int>  cumulates_;
};



#endif
