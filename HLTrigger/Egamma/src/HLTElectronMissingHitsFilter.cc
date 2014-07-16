/** \class HLTElectronMissingHitsFilter
 *
 *  \author Matteo Sani (UCSD)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronMissingHitsFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

HLTElectronMissingHitsFilter::HLTElectronMissingHitsFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  electronTag_ = iConfig.getParameter< edm::InputTag >("electronProducer");
  electronToken_ = consumes<reco::ElectronCollection>(electronTag_);
  barrelcut_ = iConfig.getParameter<int> ("barrelcut");
  endcapcut_ = iConfig.getParameter<int> ("endcapcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
}

HLTElectronMissingHitsFilter::~HLTElectronMissingHitsFilter()
{}

void HLTElectronMissingHitsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltSingleElectronOneOEMinusOneOPFilter"));
  desc.add<edm::InputTag>("electronProducer", edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15LTIPixelMatchFilte"));
  desc.add<int>("barrelcut", 0);
  desc.add<int>("endcapcut", 0);
  desc.add<int>("ncandcut", 1);
  descriptions.add("hltElectronMissingHitsFilter", desc);
}

bool HLTElectronMissingHitsFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

  using namespace trigger;

  if (saveTags())
    filterproduct.addCollectionTag(electronTag_);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(candToken_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty())
    PrevFilterOutput->getObjects(TriggerPhoton,recoecalcands);

  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByToken(electronToken_,electronHandle);

  int n(0);

  edm::RefToBase<reco::Candidate> candref;
  for (unsigned int i=0; i<recoecalcands.size(); i++) {

    reco::SuperClusterRef scCand = recoecalcands[i]->superCluster();
    for(reco::ElectronCollection::const_iterator iElectron = electronHandle->begin(); iElectron != electronHandle->end(); iElectron++) {
      reco::ElectronRef electronref(reco::ElectronRef(electronHandle, iElectron - electronHandle->begin()));
      const reco::SuperClusterRef scEle = electronref->superCluster();
      if(scCand == scEle) {
	
	int missinghits = 0;
	if (electronref->gsfTrack().isNonnull()){
        missinghits = electronref->gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    } else if (electronref->track().isNonnull()){
        missinghits = electronref->track()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    }else{
        std::cerr << "Electron without track..." << std::endl;
    }

	if(fabs(electronref->eta()) < 1.479) {
	  if (missinghits < barrelcut_) {
	    n++;
	    filterproduct.addObject(TriggerElectron, electronref);
	  }
	}
	
	if(fabs(electronref->eta()) > 1.479) {
	  if (missinghits < endcapcut_) {
	    n++;
	    filterproduct.addObject(TriggerElectron, electronref);
	  }
	}
      }
    }
  }

  bool accept(n >= ncandcut_);

  return accept;
}
