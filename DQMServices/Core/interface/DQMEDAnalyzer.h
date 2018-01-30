#ifndef CORE_DQMED_ANALYZER_H
# define CORE_DQMED_ANALYZER_H

//<<<<<< INCLUDES                                                       >>>>>>
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/moduleAbilities.h" 
#include "DQMServices/Core/interface/DQMStore.h"

// TODO: For derived modules that need it
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

namespace edm {class StreamID;}

namespace dqmDetails {struct NoCache {};}


class DQMEDAnalyzer
    : public edm::one::EDProducer<edm::one::WatchRuns,
                                  edm::one::WatchLuminosityBlocks,
                                  edm::Accumulator>
{
public:
  DQMEDAnalyzer();
  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
  virtual void beginJob() {};
  virtual void beginRun(edm::Run const &, edm::EventSetup const&) final;
  virtual void   endRun(edm::Run const &, edm::EventSetup const&) {};
  virtual void analyze(const edm::Event&, const edm::EventSetup&) = 0;
  virtual void accumulate(edm::Event const&, edm::EventSetup const&) override final;
  virtual void endJob() {};
  // TODO: Make sure those get called.
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {};
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {};



  /* (never called, only for stream-modules */ 
  virtual void endRunSummary(edm::Run const&,
                             edm::EventSetup const&,
                             dqmDetails::NoCache*) const final;
  virtual void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         dqmDetails::NoCache*) const final;
  /* ) */

  uint32_t streamId() const {return stream_id_;}
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) = 0;

private:
  uint32_t stream_id_{0};
};

#endif // CORE_DQMED_ANALYZER_H
