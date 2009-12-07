// -*- C++ -*-
//
// Class:      HLTHcalMETNoiseFilter
// 
/**\class HLTHcalMETNoiseFilter

 Description: HLT filter module for rejecting MET events due to noise in the HCAL

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Leonard Apanasevich
//         Created:  Wed Mar 25 16:01:27 CDT 2009
// $Id: HLTHcalMETNoiseFilter.cc,v 1.4 2009/09/16 20:10:21 johnpaul Exp $
//
//

#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"


HLTHcalMETNoiseFilter::HLTHcalMETNoiseFilter(const edm::ParameterSet& iConfig)
  : HcalNoiseSummaryTag (iConfig.getParameter <edm::InputTag> ("HcalNoiseSummary")),
    severity (iConfig.getParameter <int> ("severity")),
    useLooseFilter (iConfig.getParameter<bool>("useLooseFilter")),
    useTightFilter (iConfig.getParameter<bool>("useTightFilter")),
    useHighLevelFilter (iConfig.getParameter<bool>("useHighLevelFilter")),
    useCustomFilter (iConfig.getParameter<bool>("useCustomFilter")),
    minE2Over10TS (iConfig.getParameter<double>("minE2Over10TS")),
    min25GeVHitTime (iConfig.getParameter<double>("min25GeVHitTime")),
    max25GeVHitTime (iConfig.getParameter<double>("max25GeVHitTime")),
    maxZeros (iConfig.getParameter<int>("maxZeros")),
    maxHPDHits (iConfig.getParameter<int>("maxHPDHits")),
    maxRBXHits (iConfig.getParameter<int>("maxRBXHits")),
    minHPDEMF (iConfig.getParameter<double>("minHPDEMF")),
    minRBXEMF (iConfig.getParameter<double>("minRBXEMF"))
{
}


HLTHcalMETNoiseFilter::~HLTHcalMETNoiseFilter(){}


//
// member functions
//

bool HLTHcalMETNoiseFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace reco;

  bool accept=true;  // assume good event
  if (severity == 0 ) return accept; // do not filter anything

  edm::Handle<HcalNoiseSummary> NoiseSummary;
  iEvent.getByLabel(HcalNoiseSummaryTag,NoiseSummary);
  if (!NoiseSummary.isValid()) {
    LogDebug("") << "HLTHcalMETNoiseFilter: Could not find Hcal NoiseSummary product" << std::endl;
    return accept;
  }

  // apply each filter
  if(useLooseFilter     && !NoiseSummary->passLooseNoiseFilter())     return false;
  if(useTightFilter     && !NoiseSummary->passTightNoiseFilter())     return false;
  if(useHighLevelFilter && !NoiseSummary->passHighLevelNoiseFilter()) return false;
  
  // custom filter
  if(useCustomFilter) {
    if(NoiseSummary->minE2Over10TS()<minE2Over10TS) return false;
    if(NoiseSummary->min25GeVHitTime()<min25GeVHitTime) return false;
    if(NoiseSummary->max25GeVHitTime()>max25GeVHitTime) return false;
    if(NoiseSummary->maxZeros()>maxZeros) return false;
    if(NoiseSummary->maxHPDHits()>maxHPDHits) return false;
    if(NoiseSummary->maxRBXHits()>maxRBXHits) return false;
    if(NoiseSummary->minHPDEMF()<minHPDEMF) return false;
    if(NoiseSummary->minRBXEMF()<minRBXEMF) return false;
  }

  return accept;
}
