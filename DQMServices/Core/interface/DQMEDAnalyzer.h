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


class DQMEDAnalyzer
    : public edm::stream::EDAnalyzer<edm::RunSummaryCache<int>,
                                     edm::LuminosityBlockSummaryCache<int> >
{
 public:
  DQMEDAnalyzer(void);
  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
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
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // CORE_DQMED_ANALYZER_H

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
