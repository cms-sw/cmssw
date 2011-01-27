/** \class HLTrigReport
 *
 * See header file for documentation
 *
 *  $Date: 2010/05/07 15:51:08 $
 *  $Revision: 1.12 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTanalyzers/interface/HLTrigReport.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>

//
// constructors and destructor
//
HLTrigReport::HLTrigReport(const edm::ParameterSet& iConfig) :
  hlTriggerResults_ (iConfig.getParameter<edm::InputTag> ("HLTriggerResults")),
  nEvents_(0),
  nWasRun_(0),
  nAccept_(0),
  nErrors_(0),
  hlWasRun_(0),
  hltL1s_(0),
  hltPre_(0),
  hlAccept_(0),
  hlErrors_(0),
  posL1s_(0),
  posPre_(0),
  hlNames_(0),
  hltConfig_()
{
  LogDebug("HLTrigReport") << "HL TiggerResults: " + hlTriggerResults_.encode();
}

HLTrigReport::~HLTrigReport()
{ }

//
// member functions
//

void
HLTrigReport::beginRun(edm::Run const & iRun, edm::EventSetup const& iSetup)
{
  using namespace std;
  using namespace edm;
  
  bool changed (true);
  if (hltConfig_.init(iRun,iSetup,hlTriggerResults_.process(),changed)) {
    if (changed) {
      // dump previous
      dumpReport();
      nEvents_=0;
      nWasRun_=0;
      nAccept_=0;
      nErrors_=0;
      // const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTR);
      hlNames_=hltConfig_.triggerNames();
      const unsigned int n(hlNames_.size());
      hlWasRun_.resize(n);
      hltL1s_.resize(n);
      hltPre_.resize(n);
      hlAccept_.resize(n);
      hlErrors_.resize(n);
      posL1s_.resize(n);
      posPre_.resize(n);
      for (unsigned int i=0; i!=n; ++i) {
	hlWasRun_[i]=0;
	hltL1s_[i]=0;
	hltPre_[i]=0;
	hlAccept_[i]=0;
	hlErrors_[i]=0;
	posL1s_[i]=-1;
	posPre_[i]=-1;
	const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(i));
	for (unsigned int j=0; j<moduleLabels.size(); ++j) {
	  if (hltConfig_.moduleType(moduleLabels[j])=="HLTLevel1GTSeed") {
	    posL1s_[i]=j;
	  }
	  if (hltConfig_.moduleType(moduleLabels[j])=="HLTPrescaler"   ) {
	    posPre_[i]=j;
	  }
	}
      }
    }
  } else {
    // dump previous
    dumpReport();
    // clear
    nEvents_=0;
    nWasRun_=0;
    nAccept_=0;
    nErrors_=0;
    hlWasRun_.clear();
    hltL1s_.clear();
    hltPre_.clear();
    hlAccept_.clear();
    hlErrors_.clear();
    posL1s_.clear();
    posPre_.clear();
    hlNames_.clear();
  }
  return;
}
      
    
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

  // decision for each HL algorithm
  const unsigned int n(hlNames_.size());
  for (unsigned int i=0; i!=n; ++i) {
    if (HLTR->wasrun(i)) hlWasRun_[i]++;
    if (HLTR->accept(i)) hlAccept_[i]++;
    if (HLTR->error(i) ) hlErrors_[i]++;
    const int index(static_cast<int>(HLTR->index(i)));
    if (HLTR->accept(i)) {
      if (index>=posL1s_[i]) hltL1s_[i]++;
      if (index>=posPre_[i]) hltPre_[i]++;
    } else {
      if (index> posL1s_[i]) hltL1s_[i]++;
      if (index> posPre_[i]) hltPre_[i]++;
    }
  }

  return;

}

void
HLTrigReport::endJob()
{
  dumpReport();
  return;
}

void
HLTrigReport::dumpReport()
{
  // final printout of accumulated statistics

  using namespace std;
  using namespace edm;
  const unsigned int n(hlNames_.size());

  if ((n==0) && (nEvents_==0)) return;

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
	 << right << setw(7) << "HLT #" << " "
	 << right << setw(7) << "WasRun" << " "
	 << right << setw(7) << "L1S" << " "
	 << right << setw(7) << "Pre" << " "
	 << right << setw(7) << "HLT" << " "
	 << right << setw(9) << "%L1sPre" << " "
	 << right << setw(7) << "Errors" << " "
	 << "Name" << endl;

  if (n>0) {
    for (unsigned int i=0; i!=n; ++i) {
      LogVerbatim("HLTrigReport") << "HLT-Report "
	   << right << setw(7) << i << " "
	   << right << setw(7) << hlWasRun_[i] << " "
	   << right << setw(7) << hltL1s_[i] << " "
	   << right << setw(7) << hltPre_[i] << " "
	   << right << setw(7) << hlAccept_[i] << " "
	   << right << setw(9) << fixed << setprecision(5)
	   << static_cast<float>(100*hlAccept_[i])/
	      static_cast<float>(max(hltPre_[i],1u)) << " "
	   << right << setw(7) << hlErrors_[i] << " "
	   << hlNames_[i] << endl;
    }
  } else {
    LogVerbatim("HLTrigReport") << "HLT-Report - No HLT paths found!" << endl;
  }

    LogVerbatim("HLTrigReport") << endl;
    LogVerbatim("HLTrigReport") << "HLT-Report end!" << endl;
    LogVerbatim("HLTrigReport") << endl;

    return;
}
