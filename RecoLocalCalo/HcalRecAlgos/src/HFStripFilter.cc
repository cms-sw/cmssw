#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HFStripFilter.h"

HFStripFilter::HFStripFilter(const double stripThreshold, const double maxThreshold,
                             const double timeMax, const double maxStripTime,
                             const double wedgeCut, const int gap,
                             const int lstrips, const int acceptSeverityLevel,
                             const int verboseLevel)
    : stripThreshold_(stripThreshold),
      maxThreshold_(maxThreshold),
      timeMax_(timeMax),
      maxStripTime_(maxStripTime),
      wedgeCut_(wedgeCut),
      gap_(gap),
      lstrips_(lstrips),
      acceptSeverityLevel_(acceptSeverityLevel),
      verboseLevel_(verboseLevel)
{
    // For the description of CMSSW message logging, see
    // https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMessageLogger
    if (verboseLevel_ >= 20)
        edm::LogInfo("HFStripFilter") << "constructor called";
}

HFStripFilter::~HFStripFilter()
{
    if (verboseLevel_ >= 20)
        edm::LogInfo("HFStripFilter") << "destructor called";
}

void HFStripFilter::runFilter(HFRecHitCollection& rec,
                              const HcalChannelQuality* myqual,
                              const HcalSeverityLevelComputer* mySeverity) const
{
    if (verboseLevel_ >= 20)
        edm::LogInfo("HFStripFilter") << "runFilter called";

    // Cycle over the rechit collection
    for (HFRecHitCollection::iterator it = rec.begin(); it != rec.end(); ++it)
    {
        // Figure out which rechits need to be tagged

        // To tag a rechit with the anomalous hit flag, do the following
        it->setFlagField(1U, HcalPhase1FlagLabels::HFAnomalousHit);
    }
}

std::unique_ptr<HFStripFilter> HFStripFilter::parseParameterSet(
    const edm::ParameterSet& ps)
{
    return std::make_unique<HFStripFilter>(
        ps.getParameter<double>("stripThreshold"),
        ps.getParameter<double>("maxThreshold"),
        ps.getParameter<double>("timeMax"),
        ps.getParameter<double>("maxStripTime"),
        ps.getParameter<double>("wedgeCut"),
        ps.getParameter<int>("gap"),
        ps.getParameter<int>("lstrips"),
        ps.getParameter<int>("acceptSeverityLevel"),
        ps.getParameter<int>("verboseLevel")
    );
}

edm::ParameterSetDescription HFStripFilter::fillDescription()
{
    edm::ParameterSetDescription desc;

    desc.add<double>("stripThreshold", 40.0);
    desc.add<double>("maxThreshold", 100.0);
    desc.add<double>("timeMax", 6.0);
    desc.add<double>("maxStripTime", 10.0);
    desc.add<double>("wedgeCut", 0.05);
    desc.add<int>("gap", 2);
    desc.add<int>("lstrips", 2);
    desc.add<int>("acceptSeverityLevel", 9);
    desc.add<int>("verboseLevel", 0);

    return desc;
}
