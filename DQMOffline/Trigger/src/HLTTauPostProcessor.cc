#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TEfficiency.h"
#include "TProfile.h"

#include<tuple>

namespace {
  std::tuple<float, float> calcEfficiency(float num, float denom) {
    if(denom == 0.0f)
      return std::make_tuple(0.0f, 0.0f);

    //float eff = num/denom;
    constexpr double cl = 0.683f; // "1-sigma"
    const float eff = num/denom;
    const float errDown = TEfficiency::ClopperPearson(denom, num, cl, false);
    const float errUp = TEfficiency::ClopperPearson(denom, num, cl, true);

    // Because of limitation of TProfile, just take max
    return std::make_tuple(eff, std::max(eff-errDown, errUp-eff));
  }
}

HLTTauPostProcessor::HLTTauPostProcessor(const edm::ParameterSet& ps):
  dqmBaseFolder_(ps.getUntrackedParameter<std::string>("DQMBaseFolder"))
{}

HLTTauPostProcessor::~HLTTauPostProcessor()
{}

void HLTTauPostProcessor::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter)
{
  if(!iGetter.dirExists(dqmBaseFolder_)) {
    LogDebug("HLTTauDQMOffline") << "Folder " << dqmBaseFolder_ << " does not exist";
    return;
  }
  iGetter.setCurrentFolder(dqmBaseFolder_);
  for(const std::string& subfolder: iGetter.getSubdirs()) {
    std::size_t pos = subfolder.rfind("/");
    if(pos == std::string::npos)
      continue;
    ++pos; // position of first letter after /
    if(subfolder.compare(pos, 4, "HLT_") == 0) { // start with HLT_
      LogDebug("HLTTauDQMOffline") << "Processing path " << subfolder.substr(pos);
      plotFilterEfficiencies(iBooker, iGetter, subfolder);
    }
  }
}

void HLTTauPostProcessor::plotFilterEfficiencies(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, const std::string& folder) const {
  // Get the source
  const MonitorElement *eventsPerFilter = iGetter.get(folder+"/EventsPerFilter");
  if(!eventsPerFilter) {
    LogDebug("HLTTauDQMOffline") << "ME " << folder << "/EventsPerFilter not found";
    return;
  }

  // Book efficiency TProfile
  iBooker.setCurrentFolder(folder);
  MonitorElement *efficiency = iBooker.bookProfile("EfficiencyRefPrevious", "Efficiency to previous filter", eventsPerFilter->getNbinsX()-1,0,eventsPerFilter->getNbinsX()-1, 100,0,1);
  efficiency->setAxisTitle("Efficiency", 2);
  const TAxis *xaxis = eventsPerFilter->getTH1F()->GetXaxis();
  for(int bin=1; bin < eventsPerFilter->getNbinsX(); ++bin) {
    efficiency->setBinLabel(bin, xaxis->GetBinLabel(bin+1));
  }

  // Fill efficiency TProfile
  TProfile *prev = efficiency->getTProfile();
  for(int i=2; i <= eventsPerFilter->getNbinsX(); ++i) {
    if(eventsPerFilter->getBinContent(i-1) < eventsPerFilter->getBinContent(i)) {
      LogDebug("HLTTauDQMOffline") << "HLTTauPostProcessor: Encountered denominator < numerator with efficiency plot EfficiencyRefPrevious in folder " << folder << ", bin " << i << " numerator " << eventsPerFilter->getBinContent(i) << " denominator " << eventsPerFilter->getBinContent(i-1);
      continue;
    }
    const std::tuple<float, float> effErr = calcEfficiency(eventsPerFilter->getBinContent(i), eventsPerFilter->getBinContent(i-1));
    const float efficiency = std::get<0>(effErr);
    const float err = std::get<1>(effErr);

    prev->SetBinContent(i-1, efficiency);
    prev->SetBinEntries(i-1, 1);
    prev->SetBinError(i-1, std::sqrt(efficiency*efficiency + err*err));
  }
}

