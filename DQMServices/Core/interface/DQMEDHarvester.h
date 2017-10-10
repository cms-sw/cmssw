#ifndef CORE_DQMED_HARVESTER_H
#define CORE_DQMED_HARVESTER_H

//<<<<<< INCLUDES                                                       >>>>>>
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "DQMServices/Core/interface/DQMStore.h"

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

class DQMEDHarvester
: public edm::one::EDProducer<edm::one::WatchRuns,edm::one::WatchLuminosityBlocks,edm::one::SharedResources,
edm::EndLuminosityBlockProducer>
{
public:
  DQMEDHarvester(void);
  ~DQMEDHarvester() override = default;

  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
  void beginRun(edm::Run const&, edm::EventSetup const&) override {};
  void produce(edm::Event&, edm::EventSetup const&) final {};
  void endRun(edm::Run const&, edm::EventSetup const&) override {};
  void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const&) final {};
  void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const&) final;
  void endJob() final;
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) {};
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) = 0;

private:
  void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) final;

};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // CORE_DQMED_HARVESTER_H
