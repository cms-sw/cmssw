#ifndef FWCore_PythonFramework_PythonEventProcessor_h
#define FWCore_PythonFramework_PythonEventProcessor_h
// -*- C++ -*-
//
// Package:     FWCore/PythonFramework
// Class  :     PythonEventProcessor
//
/**\class PythonEventProcessor PythonEventProcessor.h "PythonEventProcessor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris D Jones
//         Created:  Fri, 20 Jan 2017 16:36:33 GMT
//

// system include files
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"

// user include files

// forward declarations
class PythonProcessDesc;

class PythonEventProcessor {
public:
  PythonEventProcessor(PyBind11ProcessDesc const&);
  PythonEventProcessor(const PythonEventProcessor&) = delete;                   // stop default
  const PythonEventProcessor& operator=(const PythonEventProcessor&) = delete;  // stop default
  ~PythonEventProcessor();
  // ---------- const member functions ---------------------
  /// Return the number of events this EventProcessor has tried to process
  /// (inclues both successes and failures, including failures due
  /// to exceptions during processing).
  int totalEvents() const { return processor_.totalEvents(); }

  /// Return the number of events processed by this EventProcessor
  /// which have been passed by one or more trigger paths.
  int totalEventsPassed() const { return processor_.totalEventsPassed(); }

  /// Return the number of events that have not passed any trigger.
  /// (N.B. totalEventsFailed() + totalEventsPassed() == totalEvents()
  int totalEventsFailed() const { return processor_.totalEventsFailed(); }

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void run();

private:
  // ---------- member data --------------------------------
  int forcePluginSetupFirst_;
  edm::EventProcessor processor_;
};

#endif
