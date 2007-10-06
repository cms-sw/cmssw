/** \class L1TrigReport
 *
 * See header file for documentation
 *
 *  $Date: 2007/10/06 09:32:23 $
 *  $Revision: 1.9 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTanalyzers/interface/L1TrigReport.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>

//
// constructors and destructor
//
L1TrigReport::L1TrigReport(const edm::ParameterSet& iConfig) :
  l1GTReadoutRecTag_(iConfig.getParameter<edm::InputTag> ("L1GTReadoutRecord")),
  nEvents_(0),
  nErrors_(0),
  nAccepts_(0),
  l1Accepts_(0),
  l1Names_(0),
  init_(false),
  nSize_(0)
{
  LogDebug("") << "Level-1 Global Trigger Readout Record: " + l1GTReadoutRecTag_.encode();
}

L1TrigReport::~L1TrigReport()
{ }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TrigReport::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // accumulation of statistics event by event

  using namespace std;
  using namespace edm;

  // get hold of L1GlobalReadoutRecord
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  try {iEvent.getByLabel(l1GTReadoutRecTag_,L1GTRR);} catch (...) {;}
  if (L1GTRR.isValid()) {
    const unsigned int n(L1GTRR->decisionWord().size());
    // initialisation
    if ( (!init_) || (nSize_!=n) ) {
      if (!init_) {
	init_=true;
      } else {
	endJob();
	nEvents_=0;
	nErrors_=0;
	nAccepts_=0;
      }
      nSize_=n;
      l1Names_.resize(n);
      l1Accepts_.resize(n);
      for (unsigned int i=0; i!=n; ++i) {
	l1Accepts_[i]=0;
	l1Names_[i]="NameNotAvailable";
      }
    }
    const bool accept(L1GTRR->decision());
    LogDebug("") << "L1GlobalTriggerReadoutRecord decision: " << accept;
    nEvents_++;
    if (accept) ++nAccepts_;
    // decision for each L1 algorithm
    for (unsigned int i=0; i!=n; ++i) {
      if (L1GTRR->decisionWord()[i]) l1Accepts_[i]++;
    }
  } else {
    LogDebug("") << "L1GlobalTriggerReadoutRecord with label ["+l1GTReadoutRecTag_.encode()+"] not found!";
    nEvents_++;
    nErrors_++;
  }

  return;

}

void
L1TrigReport::endJob()
{
  // final printout of accumulated statistics

  using namespace std;

    cout << dec << endl;
    cout << "L1T-Report " << "---------- Event  Summary ------------\n";
    cout << "L1T-Report"
	 << " Events total = " << nEvents_
	 << " passed = " << nAccepts_
	 << " failed = " << nEvents_-nErrors_-nAccepts_
	 << " errors = " << nErrors_
	 << "\n";

    cout << endl;
    cout << "L1T-Report " << "---------- L1Trig Summary ------------\n";
    cout << "L1T-Report "
	 << right << setw(10) << "L1T  Bit#" << " "
	 << right << setw(10) << "Passed" << " "
	 << right << setw(10) << "Failed" << " "
	 << right << setw(10) << "Errors" << " "
	 << "Name" << "\n";

  if (init_) {
    for (unsigned int i=0; i!=nSize_; ++i) {
      cout << "L1T-Report "
	   << right << setw(10) << i << " "
	   << right << setw(10) << l1Accepts_[i] << " "
	   << right << setw(10) << nEvents_-nErrors_-l1Accepts_[i] << " "
	   << right << setw(10) << nErrors_ << " "
	   << l1Names_[i] << "\n";
    }
  } else {
    cout << "L1T-Report - No L1 GTRRs found!" << endl;
  }

    cout << endl;
    cout << "L1T-Report end!" << endl;
    cout << endl;

    return;
}
