/** \class HLTrigReport
 *
 * See header file for documentation
 *
 *  $Date: 2007/06/19 11:47:52 $
 *  $Revision: 1.2 $
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
  LogDebug("") << "HL TiggerResults: " + hlTriggerResults_.encode();
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
  try {iEvent.getByLabel(hlTriggerResults_,HLTR);} catch (...) {;}
  if (HLTR.isValid()) {
    if (HLTR->wasrun()) nWasRun_++;
    const bool accept(HLTR->accept());
    LogDebug("") << "HL TriggerResults decision: " << accept;
    if (accept) ++nAccept_;
    if (HLTR->error() ) nErrors_++;
  } else {
    LogDebug("") << "HL TriggerResults with label ["+hlTriggerResults_.encode()+"] not found!";
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
  const unsigned int n(hlNames_.size());

    cout << dec << endl;
    cout << "HLT-Report " << "---------- Event  Summary ------------\n";
    cout << "HLT-Report"
	 << " Events total = " << nEvents_
	 << " wasrun = " << nWasRun_
	 << " passed = " << nAccept_
	 << " errors = " << nErrors_
	 << "\n";

    cout << endl;
    cout << "HLT-Report " << "---------- HLTrig Summary ------------\n";
    cout << "HLT-Report "
	 << right << setw(10) << "HLT  Bit#" << " "
	 << right << setw(10) << "WasRun" << " "
	 << right << setw(10) << "Passed" << " "
	 << right << setw(10) << "Errors" << " "
	 << "Name" << "\n";

  if (init_) {
    for (unsigned int i=0; i!=n; ++i) {
      cout << "HLT-Report "
	   << right << setw(10) << i << " "
	   << right << setw(10) << hlWasRun_[i] << " "
	   << right << setw(10) << hlAccept_[i] << " "
	   << right << setw(10) << hlErrors_[i] << " "
	   << hlNames_[i] << "\n";
    }
  } else {
    cout << "HLT-Report - No HL TriggerResults found!" << endl;
  }

    cout << endl;
    cout << "HLT-Report end!" << endl;
    cout << endl;

    return;
}
