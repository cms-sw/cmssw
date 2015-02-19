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
  DQMEDAnalyzer(void);
  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
  virtual void beginStream(edm::StreamID id) final;
  virtual void beginRun(edm::Run const &, edm::EventSetup const&) final;
  static std::shared_ptr<dqmDetails::NoCache> globalBeginRunSummary(edm::Run const&,
                                                        edm::EventSetup const&,
                                                        RunContext const*);
  virtual void endRunSummary(edm::Run const&,
                             edm::EventSetup const&,
                             dqmDetails::NoCache*) const final;
  static void globalEndRunSummary(edm::Run const&,
                                  edm::EventSetup const&,
                                  RunContext const*,
                                  dqmDetails::NoCache*);
  static std::shared_ptr<dqmDetails::NoCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                    edm::EventSetup const&,
                                                                    LuminosityBlockContext const*);
  virtual void endLuminosityBlockSummary(edm::LuminosityBlock const&,
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

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

//############################## ONLY NEEDED IN THE TRANSITION PERIOD ################################
//here the thread_unsafe (simplified) carbon copy of the DQMEDAnalyzer

#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace thread_unsafe {
  class DQMEDAnalyzer: public edm::EDAnalyzer
    {
    public:
      DQMEDAnalyzer(void);
      virtual void beginRun(edm::Run const &, edm::EventSetup const&) final;
      virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
      virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) = 0;
      
    private:
    };
} //thread_unsafe namespace

#endif // CORE_DQMED_ANALYZER_H
