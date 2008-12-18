#ifndef Modules_EventContentAnalyzer_h
#define Modules_EventContentAnalyzer_h
// -*- C++ -*-
//
// Package:     Modules
// Class  :     EventContentAnalyzer
// 
/**\class EventContentAnalyzer EventContentAnalyzer.h FWCore/Modules/src/EventContentAnalyzer.h

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
namespace edm {
   class ParameterSetDescription;
}

class EventContentAnalyzer : public edm::EDAnalyzer {
public:
   explicit EventContentAnalyzer(const edm::ParameterSet&);
   ~EventContentAnalyzer();
   
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endJob();

   static void fillDescription(edm::ParameterSetDescription& iDesc,
                               std::string const& moduleLabel);

private:
      
   // ----------member data ---------------------------
   std::string indentation_;
   std::string verboseIndentation_;
   std::vector<std::string> moduleLabels_;
   bool        verbose_; 
   std::vector<std::string> getModuleLabels_;
   bool        getData_;
   int         evno_;
   std::map<std::string, int>  cumulates_;
};



#endif
