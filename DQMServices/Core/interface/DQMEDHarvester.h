#ifndef CORE_DQMED_HARVESTER_H
#define CORE_DQMED_HARVESTER_H

//<<<<<< INCLUDES                                                       >>>>>>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

class DQMEDHarvester
: public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::WatchLuminosityBlocks,edm::one::SharedResources>
{
public:
  DQMEDHarvester(void);
#ifdef __INTEL_COMPILER
  virtual ~DQMEDHarvester() = default;
#endif
  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) {};
  virtual void analyze(edm::Event const&, edm::EventSetup const&) final {};
  virtual void endRun(edm::Run const&, edm::EventSetup const&) {};
  virtual void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const&) final {};
  virtual void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const&) final;
  virtual void endJob() final;
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) {};
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) = 0;

private:

};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // CORE_DQMED_HARVESTER_H
