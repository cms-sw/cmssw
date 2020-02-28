/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Class to print out information on current geometry.
 **/
class CTPPSAlignmentInfo : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSAlignmentInfo(const edm::ParameterSet&);

private:
  std::string alignmentType_;

  edm::ESWatcher<RPRealAlignmentRecord> watcherRealAlignments_;
  edm::ESWatcher<RPMisalignedAlignmentRecord> watcherMisalignedAlignments_;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void printInfo(const CTPPSRPAlignmentCorrectionsData& alignments, const edm::Event& event) const;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSAlignmentInfo::CTPPSAlignmentInfo(const edm::ParameterSet& iConfig)
    : alignmentType_(iConfig.getUntrackedParameter<std::string>("alignmentType", "real")) {}

//----------------------------------------------------------------------------------------------------

void CTPPSAlignmentInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<CTPPSRPAlignmentCorrectionsData> alignments;

  if (alignmentType_ == "real") {
    if (watcherRealAlignments_.check(iSetup)) {
      iSetup.get<RPRealAlignmentRecord>().get(alignments);
      printInfo(*alignments, iEvent);
    }
    return;
  }

  else if (alignmentType_ == "misaligned") {
    if (watcherMisalignedAlignments_.check(iSetup)) {
      iSetup.get<RPMisalignedAlignmentRecord>().get(alignments);
      printInfo(*alignments, iEvent);
    }
    return;
  }

  throw cms::Exception("CTPPSAlignmentInfo") << "Unknown geometry type: `" << alignmentType_ << "'.";
}

//----------------------------------------------------------------------------------------------------

void CTPPSAlignmentInfo::printInfo(const CTPPSRPAlignmentCorrectionsData& alignments, const edm::Event& event) const {
  time_t unixTime = event.time().unixTime();
  char timeStr[50];
  strftime(timeStr, 50, "%F %T", localtime(&unixTime));

  edm::LogInfo("CTPPSAlignmentInfo") << "New " << alignmentType_ << " alignments found in run=" << event.id().run()
                                     << ", event=" << event.id().event() << ", UNIX timestamp=" << unixTime << " ("
                                     << timeStr << "):\n"
                                     << alignments;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSAlignmentInfo);
