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
// $Id$
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

      void postBeginJob();
      void postEndJob();

      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const Event&, const EventSetup&);

      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);
      
private:
      RandomNumberGeneratorService(const RandomNumberGeneratorService&); // stop default
      
      const RandomNumberGeneratorService& operator=(const RandomNumberGeneratorService&); // stop default
      
      // ---------- member data --------------------------------
      typedef std::map<std::string,uint32_t> LabelToGenMap;
      LabelToGenMap labelToSeed_;
      LabelToGenMap::const_iterator presentGen_;
      std::string unknownLabel_;
   };
      
      
   }
}


#endif
