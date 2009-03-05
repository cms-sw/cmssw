/** \class HLTrigReport
 *
 * See header file for documentation
 *
 *  $Date: 2009/03/05 15:21:37 $
 *  $Revision: 1.7 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTanalyzers/interface/HLTrigReport.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>

//
// constructors and destructor
//
HLTrigReport::HLTrigReport(const edm::ParameterSet& iConfig) :
  hlTriggerResults_ (iConfig.getParameter<edm::InputTag> ("HLTriggerResults")),
  triggerNames_(),
  nEvents_(0),
  nWasRun_(0),
  nAccept_(0),
  nErrors_(0),
  hlWasRun_(0),
  hlAccept_(0),
  hlErrors_(0),
  hlNames_(0),
  init_(false)
{
  LogDebug("HLTrigReport") << "HL TiggerResults: " + hlTriggerResults_.encode();
}

HLTrigReport::~HLTrigReport()
{ }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTrigReport::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // accumulation of statistics event by event

  using namespace std;
  using namespace edm;

  nEvents_++;

  // get hold of TriggerResults
  Handle<TriggerResults> HLTR;
  iEvent.getByLabel(hlTriggerResults_,HLTR);
  if (HLTR.isValid()) {
    if (HLTR->wasrun()) nWasRun_++;
    const bool accept(HLTR->accept());
    LogDebug("HLTrigReport") << "HL TriggerResults decision: " << accept;
    if (accept) ++nAccept_;
    if (HLTR->error() ) nErrors_++;
  } else {
    LogDebug("HLTrigReport") << "HL TriggerResults with label ["+hlTriggerResults_.encode()+"] not found!";
    nErrors_++;
    return;
  }

  // initialisation (could be made dynamic)
  if (!init_) {
    init_=true;
    triggerNames_.init(*HLTR);
    hlNames_=triggerNames_.triggerNames();
    const unsigned int n(hlNames_.size());
    hlWasRun_.resize(n);
    hlAccept_.resize(n);
    hlErrors_.resize(n);
    for (unsigned int i=0; i!=n; ++i) {
      hlWasRun_[i]=0;
      hlAccept_[i]=0;
      hlErrors_[i]=0;
    }
  }

  // decision for each HL algorithm
  const unsigned int n(hlNames_.size());
  for (unsigned int i=0; i!=n; ++i) {
    if (HLTR->wasrun(i)) hlWasRun_[i]++;
    if (HLTR->accept(i)) hlAccept_[i]++;
    if (HLTR->error(i) ) hlErrors_[i]++;
  }

  return;

}

void
HLTrigReport::endJob()
{
  // final printout of accumulated statistics

  using namespace std;
  using namespace edm;
  const unsigned int n(hlNames_.size());

    LogVerbatim("HLTrigReport") << dec << endl;
    LogVerbatim("HLTrigReport") << "HLT-Report " << "---------- Event  Summary ------------" << endl;
    LogVerbatim("HLTrigReport") << "HLT-Report"
	 << " Events total = " << nEvents_
	 << " wasrun = " << nWasRun_
	 << " passed = " << nAccept_
	 << " errors = " << nErrors_
	 << endl;

    LogVerbatim("HLTrigReport") << endl;
    LogVerbatim("HLTrigReport") << "HLT-Report " << "---------- HLTrig Summary ------------" << endl;
    LogVerbatim("HLTrigReport") << "HLT-Report "
	 << right << setw(10) << "HLT  Bit#" << " "
	 << right << setw(10) << "WasRun" << " "
	 << right << setw(10) << "Passed" << " "
	 << right << setw(10) << "Errors" << " "
	 << "Name" << endl;

  if (init_) {
    for (unsigned int i=0; i!=n; ++i) {
      LogVerbatim("HLTrigReport") << "HLT-Report "
	   << right << setw(10) << i << " "
	   << right << setw(10) << hlWasRun_[i] << " "
	   << right << setw(10) << hlAccept_[i] << " "
	   << right << setw(10) << hlErrors_[i] << " "
	   << hlNames_[i] << endl;
    }
  } else {
    LogVerbatim("HLTrigReport") << "HLT-Report - No HL TriggerResults found!" << endl;
  }

    LogVerbatim("HLTrigReport") << endl;
    LogVerbatim("HLTrigReport") << "HLT-Report end!" << endl;
    LogVerbatim("HLTrigReport") << endl;

    return;
}
