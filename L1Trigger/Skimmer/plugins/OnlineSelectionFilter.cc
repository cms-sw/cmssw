#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include <iostream>

class OnlineSelectionFilter : public edm::EDFilter
{
public:
  explicit OnlineSelectionFilter(edm::ParameterSet const&);

  virtual ~OnlineSelectionFilter();

  virtual bool filter(edm::Event& e, edm::EventSetup const& c);
  void endJob();
 
private:

  edm::InputTag gtReadoutSource;
  
  int nL1Passed;
  int nL1Failed;
  
};

OnlineSelectionFilter::OnlineSelectionFilter(const edm::ParameterSet& ps) :
  gtReadoutSource(ps.getParameter< edm::InputTag >("gtReadoutSource")),
  nL1Passed(0), nL1Failed(0)
{
}

OnlineSelectionFilter::~OnlineSelectionFilter()
{
}

bool OnlineSelectionFilter::filter(edm::Event& iEvent, edm::EventSetup const&)
{
  edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
  iEvent.getByLabel(gtReadoutSource, gtRecord);
  bool theL1Decision = gtRecord->decision();
  if(theL1Decision)
    {
      nL1Passed++;
    }
  else
    {
      nL1Failed++;
    }
  std::cout << "OnlineSelectionFilter decision is " << theL1Decision << std::endl;
  return theL1Decision;
}

void OnlineSelectionFilter::endJob()
{
  std::cout << "L1 Failed event count = " << nL1Failed << std::endl;
  std::cout << "L1 Passed event count = " << nL1Passed << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(OnlineSelectionFilter);
