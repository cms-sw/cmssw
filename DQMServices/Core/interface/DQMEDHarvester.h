#ifndef CORE_DQMED_HARVESTER_H
# define CORE_DQMED_HARVESTER_H

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
: public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:
  DQMEDHarvester(void);
  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
  virtual void beginRun(edm::Run const &, edm::EventSetup const&) final;
  virtual void endRun(edm::Run const &, edm::EventSetup const&);
  virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) = 0;

private:
  uint32_t stream_id_;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // CORE_DQMED_HARVESTER_H
