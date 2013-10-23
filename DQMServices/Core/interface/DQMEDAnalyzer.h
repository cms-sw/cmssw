#ifndef CORE_DQMED_ANALYZER_H
# define CORE_DQMED_ANALYZER_H

//<<<<<< INCLUDES                                                       >>>>>>
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptor.h"
//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

namespace edm {class StreamID;}


class DQMEDAnalyzer
    : public edm::stream::EDAnalyzer<edm::RunSummaryCache<int>,
                                     edm::LuminosityBlockSummaryCache<int> >
{
 public:
  DQMEDAnalyzer(void);
  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
  virtual void beginStream(edm::StreamID id) final;
  static std::shared_ptr<int> globalBeginRunSummary(edm::Run const&,
                                                    edm::EventSetup const&,
                                                    RunContext const*);
  virtual void endRunSummary(edm::Run const&,
                             edm::EventSetup const&,
                             int*) const = 0;
  static void globalEndRunSummary(edm::Run const&,
                                  edm::EventSetup const&,
                                  RunContext const*,
                                  int*);
  static std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                edm::EventSetup const&,
                                                                LuminosityBlockContext const*);
  virtual void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         int*) const = 0;
  static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                              edm::EventSetup const&,
                                              LuminosityBlockContext const*,
                                              int*);
  uint32_t streamId() const {return stream_id_;}
  virtual void bookHistograms(edm::Run const&,
                              uint32_t streamId,
                              uint32_t moduleId) = 0;

private:
  uint32_t stream_id_;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // CORE_DQMED_ANALYZER_H

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
