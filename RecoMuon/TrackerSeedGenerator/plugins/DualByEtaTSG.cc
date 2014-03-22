#include "RecoMuon/TrackerSeedGenerator/plugins/DualByEtaTSG.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DualByEtaTSG::DualByEtaTSG(const edm::ParameterSet &pset,edm::ConsumesCollector& IC) : SeparatingTSG(pset,IC){
  theCategory ="DualByEtaTSG";
  theEtaSeparation = pset.getParameter<double>("etaSeparation");
  if (nTSGs()!=2)
    {edm::LogError(theCategory)<<"not two seed generators provided";}
}

unsigned int DualByEtaTSG::selectTSG(const TrackCand & muonTrackCand, const TrackingRegion& region)
{
  LogDebug(theCategory)<<"|eta|=|"<<muonTrackCand.second->eta()<<"|"
		       <<" compared to: "<<theEtaSeparation;
  return (fabs(muonTrackCand.second->eta()) < theEtaSeparation);
}
    
