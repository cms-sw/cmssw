#ifndef Services_RandomNumberGeneratorService_h
#define Services_RandomNumberGeneratorService_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     RandomNumberGeneratorService
// 
/**\class RandomNumberGeneratorService RandomNumberGeneratorService.h FWCore/Services/interface/RandomNumberGeneratorService.h

 Description: Concrete implementation of a RandomNumberGenerator

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Mar  7 09:43:43 EST 2006
// $Id: RandomNumberGeneratorService.h,v 1.1 2006/03/07 19:46:37 chrjones Exp $
//

// system include files
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <map>
#include <string>
// user include files

// forward declarations
namespace edm {
   class ParameterSet;
   namespace service {

      class RandomNumberGeneratorService : public RandomNumberGenerator
   {
      
public:
      RandomNumberGeneratorService(const ParameterSet&, ActivityRegistry&);
      //virtual ~RandomNumberGeneratorService();
      
      // ---------- const member functions ---------------------
      virtual uint32_t mySeed() const ;
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      void preModuleConstruction(const ModuleDescription&);
      void postModuleConstruction(const ModuleDescription&);

      void preSourceConstruction(const ModuleDescription&);
      void postSourceConstruction(const ModuleDescription&);

      void postBeginJob();
      void postEndJob();

      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const Event&, const EventSetup&);

      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);
      
private:
      RandomNumberGeneratorService(const RandomNumberGeneratorService&); // stop default
      
      const RandomNumberGeneratorService& operator=(const RandomNumberGeneratorService&); // stop default
      
      void push(const std::string&);
      void pop();

      // ---------- member data --------------------------------
      typedef std::map<std::string,uint32_t> LabelToGenMap;
      LabelToGenMap labelToSeed_;
      std::vector<LabelToGenMap::const_iterator> labelStack_;
      LabelToGenMap::const_iterator presentGen_;
      std::vector<std::string> unknownLabelStack_;
      std::string unknownLabel_;
   };
      
      
   }
}


#endif
