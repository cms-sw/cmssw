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
// $Id: HLTHcalMETNoiseFilter.cc,v 1.10 2010/04/13 23:15:29 johnpaul Exp $
//
//

#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/HcalNoiseRBX.h"

#include <iostream>

HLTHcalMETNoiseFilter::HLTHcalMETNoiseFilter(const edm::ParameterSet& iConfig)
  : noisealgo_(iConfig),
    HcalNoiseRBXCollectionTag_(iConfig.getParameter<edm::InputTag>("HcalNoiseRBXCollection")),
    severity_(iConfig.getParameter<int> ("severity")),
    maxNumRBXs_(iConfig.getParameter<int>("maxNumRBXs")),
    numRBXsToConsider_(iConfig.getParameter<int>("numRBXsToConsider")),
    needHighLevelCoincidence_(iConfig.getParameter<bool>("needHighLevelCoincidence")),
    useLooseRatioFilter_(iConfig.getParameter<bool>("useLooseRatioFilter")),
    useLooseHitsFilter_(iConfig.getParameter<bool>("useLooseHitsFilter")),
    useLooseZerosFilter_(iConfig.getParameter<bool>("useLooseZerosFilter")),
    useLooseTimingFilter_(iConfig.getParameter<bool>("useLooseTimingFilter")),
    useTightRatioFilter_(iConfig.getParameter<bool>("useTightRatioFilter")),
    useTightHitsFilter_(iConfig.getParameter<bool>("useTightHitsFilter")),
    useTightZerosFilter_(iConfig.getParameter<bool>("useTightZerosFilter")),
    useTightTimingFilter_(iConfig.getParameter<bool>("useTightTimingFilter")),
    minRecHitE_(iConfig.getParameter<double>("minRecHitE")),
    minLowHitE_(iConfig.getParameter<double>("minLowHitE")),
    minHighHitE_(iConfig.getParameter<double>("minHighHitE"))
{
}


HLTHcalMETNoiseFilter::~HLTHcalMETNoiseFilter(){}


//
// member functions
//

bool HLTHcalMETNoiseFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace reco;

  // in this case, do not filter anything
  if(severity_==0) return true;

  // get the RBXs produced by RecoMET/METProducers/HcalNoiseInfoProducer
  edm::Handle<HcalNoiseRBXCollection> rbxs_h;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag_,rbxs_h);
  if(!rbxs_h.isValid()) {
    edm::LogError("DataNotFound") << "HLTHcalMETNoiseFilter: Could not find HcalNoiseRBXCollection product named "
				  << HcalNoiseRBXCollectionTag_ << "." << std::endl;
    return true;
  }

  // reject events with too many RBXs
  if(static_cast<int>(rbxs_h->size())>maxNumRBXs_) return true;

  // create a sorted set of the RBXs, ordered by energy
  noisedataset_t data;
  for(HcalNoiseRBXCollection::const_iterator it=rbxs_h->begin(); it!=rbxs_h->end(); ++it) {
    const HcalNoiseRBX &rbx=(*it);
    CommonHcalNoiseRBXData d(rbx, minRecHitE_, minLowHitE_, minHighHitE_);
    data.insert(d);
  }

  // data is now sorted by RBX energy
  // only consider top N=numRBXsToConsider_ energy RBXs
  int cntr=0;
  for(noisedataset_t::const_iterator it=data.begin();
      it!=data.end() && cntr<numRBXsToConsider_;
      ++it, ++cntr) {

    bool passFilter=true;
    if(useLooseRatioFilter_ && !noisealgo_.passLooseRatio(*it)) passFilter=false;
    else if(useLooseHitsFilter_ && !noisealgo_.passLooseHits(*it)) passFilter=false;
    else if(useLooseZerosFilter_ && !noisealgo_.passLooseZeros(*it)) passFilter=false;
    else if(useLooseTimingFilter_ && !noisealgo_.passLooseTiming(*it)) passFilter=false;
    else if(useTightRatioFilter_ && !noisealgo_.passTightRatio(*it)) passFilter=false;
    else if(useTightHitsFilter_ && !noisealgo_.passTightHits(*it)) passFilter=false;
    else if(useTightZerosFilter_ && !noisealgo_.passTightZeros(*it)) passFilter=false;
    else if(useTightTimingFilter_ && !noisealgo_.passTightTiming(*it)) passFilter=false;
    
    if((needHighLevelCoincidence_ && !noisealgo_.passHighLevelNoiseFilter(*it) && !passFilter) ||
       (!needHighLevelCoincidence_ && !passFilter)) {
      LogDebug("") << "HLTHcalMETNoiseFilter debug: Found a noisy RBX: "
		   << "energy=" << it->energy() << "; "
		   << "ratio=" << it->ratio() << "; "
		   << "# RBX hits=" << it->numRBXHits() << "; "
		   << "# HPD hits=" << it->numHPDHits() << "; "
		   << "# Zeros=" << it->numZeros() << "; "
		   << "min time=" << it->minHighEHitTime() << "; "
		   << "max time=" << it->maxHighEHitTime() << "; "
		   << "RBX EMF=" << it->RBXEMF()
		   << std::endl;
      return false;
    }
  }

  // no problems found
  return true;
}
