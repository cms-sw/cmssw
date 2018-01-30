#ifndef CORE_DQMED_ANALYZER_H
# define CORE_DQMED_ANALYZER_H

//<<<<<< INCLUDES                                                       >>>>>>
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptor.h"
#include "DQMServices/Core/interface/DQMStore.h"
//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

namespace edm {class StreamID;}

namespace dqmDetails {struct NoCache {};}


class DQMEDAnalyzer
    : public edm::stream::EDAnalyzer<edm::RunSummaryCache<dqmDetails::NoCache>,
                                     edm::LuminosityBlockSummaryCache<dqmDetails::NoCache> >
{
public:
  DQMEDAnalyzer();
  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
  void beginStream(edm::StreamID id) final;
  void beginRun(edm::Run const &, edm::EventSetup const&) final;
  static std::shared_ptr<dqmDetails::NoCache> globalBeginRunSummary(edm::Run const&,
                                                        edm::EventSetup const&,
                                                        RunContext const*);
  void endRunSummary(edm::Run const&,
                             edm::EventSetup const&,
                             dqmDetails::NoCache*) const final;
  static void globalEndRunSummary(edm::Run const&,
                                  edm::EventSetup const&,
                                  RunContext const*,
                                  dqmDetails::NoCache*);
  static std::shared_ptr<dqmDetails::NoCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                    edm::EventSetup const&,
                                                                    LuminosityBlockContext const*);
  void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         dqmDetails::NoCache*) const final;
  static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                              edm::EventSetup const&,
                                              LuminosityBlockContext const*,
                                              dqmDetails::NoCache*);
  uint32_t streamId() const {return stream_id_;}
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) = 0;

private:
  uint32_t stream_id_;
};

#endif // CORE_DQMED_ANALYZER_H
