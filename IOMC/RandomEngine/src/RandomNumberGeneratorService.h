#ifndef RandomEngine_RandomNumberGeneratorService_h
#define RandomEngine_RandomNumberGeneratorService_h
// -*- C++ -*-
//
// Package:     RandomEngine
// Class  :     RandomNumberGeneratorService
// 
/**\class RandomNumberGeneratorService RandomNumberGeneratorService.h IOMC/RandomEngine/src/RandomNumberGeneratorService.h

 Description: Manages random number engines for modules and the source

 Usage: See comments in base class, FWCore/Utilities/RandomNumberGenerator.h

*/
//
// Original Authors:  Chris Jones, W. David Dagenhart
//   Created:  Tue Mar  7 09:43:43 EST 2006 (originally in FWCore/Services)
//

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "boost/shared_ptr.hpp"

#include <map>

namespace edm {
  class ParameterSet;
  class ModuleDescription;
  class EventID;
  class Timestamp;
  class Event;
  class EventSetup;
  
  namespace service {

    class RandomNumberGeneratorService : public RandomNumberGenerator
    {
      
    public:

      RandomNumberGeneratorService(const ParameterSet& iPSet, ActivityRegistry& iRegistry);
      virtual ~RandomNumberGeneratorService();

      virtual CLHEP::HepRandomEngine& getEngine() const;    

      virtual uint32_t mySeed() const;

      // The following functions should not be used by general users.  They
      // should only be called by code designed to work with the service while
      // it is saving the engine state to an event or restoring it from an event
      // and also used to keep track of which module is currently active.
      // The first 10 functions are called at various points during he main
      // processing loop.  The next 3 are called by a dedicated producer
      // module (RandomEngineStateProducer).  The other two by the InputSource
      // base class.

      void preModuleConstruction(const ModuleDescription& iDesc);
      void postModuleConstruction(const ModuleDescription& iDesc);

      void preSourceConstruction(const ModuleDescription& iDesc);
      void postSourceConstruction(const ModuleDescription& iDesc);

      void postBeginJob();
      void postEndJob();

      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const Event&, const EventSetup&);

      void preModule(const ModuleDescription& iDesc);
      void postModule(const ModuleDescription& iDesc);

      void preModuleBeginJob(const ModuleDescription& iDesc);
      void postModuleBeginJob(const ModuleDescription& iDesc);

      void preModuleEndJob(const ModuleDescription& iDesc);
      void postModuleEndJob(const ModuleDescription& iDesc);

      virtual const std::vector<std::string>& getCachedLabels() const;
      virtual const std::vector<std::vector<uint32_t> >& getCachedStates() const;
      virtual const std::vector<std::vector<uint32_t> >& getCachedSeeds() const;

      virtual void snapShot();
      virtual void restoreState(const Event& event);

      // For debugging purposes only
      virtual void print();

      void saveEngineState(const std::string& fileName );

      void restoreEngineState(const std::string& fileName);

    private:

      RandomNumberGeneratorService(const RandomNumberGeneratorService&); // disallow default
      
      const RandomNumberGeneratorService& operator=(const RandomNumberGeneratorService&); // disallow default
      
      // These two functions are called internally to keep track
      // of which module is currently active

      void push(const std::string& iLabel);
      void pop();

      void checkEngineType(const std::string& typeFromConfig,
                           const std::string& typeFromEvent,
                           const std::string& engineLabel);

      void dumpVector(const std::vector<uint32_t> &v);

      void stashVector(const std::vector<unsigned long> &v, std::ostream &os);

      std::vector<unsigned long> restoreVector(std::istream &is, const int32_t n);

      bool processStanza(std::istream &is);

      bool isEngineNameValid(std::string const &name);

      int32_t expectedSeedCount(std::string const &name);

      std::string constructSaveFileName();

      void oldStyleConfig(const ParameterSet& iPSet);

      // ---------- member data --------------------------------

      // We store the engines using the corresponding module label
      // as a key into a map
      typedef std::map<std::string, boost::shared_ptr<CLHEP::HepRandomEngine> > EngineMap;
      EngineMap engineMap_;

      // The next four help to keep track of the currently active
      // module (its label and associated engine)

      std::vector<EngineMap::const_iterator> engineStack_;
      EngineMap::const_iterator currentEngine_;

      std::vector<std::string> labelStack_;
      std::string currentLabel_;

      // This holds the module label used in a previous process
      // to store the state of the random number engines.  The
      // empty string is used to signal that we are not trying
      // to restore the random numbers.
      std::string restoreStateLabel_;

      // The state of the engines is cached at the beginning the
      // processing loop for each event.  The producer module
      // gets called later and writes these cached vectors into
      // the event.
      std::vector<std::string> cachedLabels_;
      std::vector<std::vector<uint32_t> > cachedStates_;
      std::vector<std::vector<uint32_t> > cachedSeeds_;

      // Keeps track of the seeds used to initialize the engines.
      // Also uses the module label as a key
      std::map<std::string, std::vector<uint32_t> > seedMap_;

      // Keep the name of the file where we want to save the state
      // of all declared engines at the end of each event. A blank
      // name means don't bother.  Also, keep a record of whether
      // the save file name has been recorded in the job report.
      std::string saveFileName_;
      bool saveFileNameRecorded_;

      // Keep the name of the file from which we restore the state
      // of all declared engines at the befinnig of a run. A
      // blank name means there isn't one.
      std::string restoreFileName_;

     // Remember if we are dealing with a new-style or old-style
     // configuration file.
     bool oldStyle_;
    };
  }
}

#endif
