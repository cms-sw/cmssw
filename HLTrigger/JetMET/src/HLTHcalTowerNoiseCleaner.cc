// -*- C++ -*-
//
// Class:      HLTHcalTowerNoiseCleaner
// 
/**\class HLTHcalTowerNoiseCleaner

 Description: HLT filter module for cleaning HCal Noise from MET or MHT

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alexander Mott
//         Created:  Mon Nov 21 11:32:00 CEST 2011
// 
//
//


#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>
#include <fstream>
#include <TVector3.h>
#include <TLorentzVector.h>
#include <set>

#include "HLTrigger/JetMET/interface/HLTHcalTowerNoiseCleaner.h"

//#include <Point.h>

HLTHcalTowerNoiseCleaner::HLTHcalTowerNoiseCleaner(const edm::ParameterSet& iConfig)
  : HcalNoiseRBXCollectionTag_(iConfig.getParameter<edm::InputTag>("HcalNoiseRBXCollection")),
    TowerCollectionTag_(iConfig.getParameter<edm::InputTag>("CaloTowerCollection")),
    severity_(iConfig.getParameter<int> ("severity")),
    maxNumRBXs_(iConfig.getParameter<int>("maxNumRBXs")),
    numRBXsToConsider_(iConfig.getParameter<int>("numRBXsToConsider")),
    needEMFCoincidence_(iConfig.getParameter<bool>("needEMFCoincidence")),
    minRBXEnergy_(iConfig.getParameter<double>("minRBXEnergy")),
    minRatio_(iConfig.getParameter<double>("minRatio")),
    maxRatio_(iConfig.getParameter<double>("maxRatio")),
    minHPDHits_(iConfig.getParameter<int>("minHPDHits")),
    minRBXHits_(iConfig.getParameter<int>("minRBXHits")),
    minHPDNoOtherHits_(iConfig.getParameter<int>("minHPDNoOtherHits")),
    minZeros_(iConfig.getParameter<int>("minZeros")),
    minHighEHitTime_(iConfig.getParameter<double>("minHighEHitTime")),
    maxHighEHitTime_(iConfig.getParameter<double>("maxHighEHitTime")),
    maxRBXEMF_(iConfig.getParameter<double>("maxRBXEMF")),
    minRecHitE_(iConfig.getParameter<double>("minRecHitE")),
    minLowHitE_(iConfig.getParameter<double>("minLowHitE")),
    minHighHitE_(iConfig.getParameter<double>("minHighHitE")),
    TS4TS5EnergyThreshold_(iConfig.getParameter<double>("TS4TS5EnergyThreshold"))
{

  std::vector<double> TS4TS5UpperThresholdTemp = iConfig.getParameter<std::vector<double> >("TS4TS5UpperThreshold");
  std::vector<double> TS4TS5UpperCutTemp = iConfig.getParameter<std::vector<double> >("TS4TS5UpperCut");
  std::vector<double> TS4TS5LowerThresholdTemp = iConfig.getParameter<std::vector<double> >("TS4TS5LowerThreshold");
  std::vector<double> TS4TS5LowerCutTemp = iConfig.getParameter<std::vector<double> >("TS4TS5LowerCut");

  for(int i = 0; i < (int)TS4TS5UpperThresholdTemp.size() && i < (int)TS4TS5UpperCutTemp.size(); i++)
     TS4TS5UpperCut_.push_back(std::pair<double, double>(TS4TS5UpperThresholdTemp[i], TS4TS5UpperCutTemp[i]));
  sort(TS4TS5UpperCut_.begin(), TS4TS5UpperCut_.end());

  for(int i = 0; i < (int)TS4TS5LowerThresholdTemp.size() && i < (int)TS4TS5LowerCutTemp.size(); i++)
     TS4TS5LowerCut_.push_back(std::pair<double, double>(TS4TS5LowerThresholdTemp[i], TS4TS5LowerCutTemp[i]));
  sort(TS4TS5LowerCut_.begin(), TS4TS5LowerCut_.end());

  produces<CaloTowerCollection>();
}


HLTHcalTowerNoiseCleaner::~HLTHcalTowerNoiseCleaner(){}

void
HLTHcalTowerNoiseCleaner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HcalNoiseRBXCollection",edm::InputTag("hltHcalNoiseInfoProducer"));
  desc.add<edm::InputTag>("CaloTowerCollection",edm::InputTag("hltTowerMakerForAll"));
  desc.add<double>("maxTowerNoiseEnergyFraction",0.5);
  desc.add<int>("severity",1);
  desc.add<int>("maxNumRBXs",2);
  desc.add<int>("numRBXsToConsider",2);
  desc.add<bool>("needEMFCoincidence",true);
  desc.add<double>("minRBXEnergy",50.0);
  desc.add<double>("minRatio",-999.);
  desc.add<double>("maxRatio",999.);
  desc.add<int>("minHPDHits",17);
  desc.add<int>("minRBXHits",999);
  desc.add<int>("minHPDNoOtherHits",10);
  desc.add<int>("minZeros",10);
  desc.add<double>("minHighEHitTime",-9999.0);
  desc.add<double>("maxHighEHitTime",9999.0);
  desc.add<double>("maxRBXEMF",0.02);
  desc.add<double>("minRecHitE",1.5);
  desc.add<double>("minLowHitE",10.0);
  desc.add<double>("minHighHitE",25.0);
  desc.add<double>("TS4TS5EnergyThreshold",50.0);

  double TS4TS5UpperThresholdArray[5] = {70, 90, 100, 400, 4000 };
  double TS4TS5UpperCutArray[5] = {1, 0.8, 0.75, 0.72, 0.72};
  double TS4TS5LowerThresholdArray[7] = {100, 120, 150, 200, 300, 400, 500};
  double TS4TS5LowerCutArray[7] = {-1, -0.7, -0.4, -0.2, -0.08, 0, 0.1};
  std::vector<double> TS4TS5UpperThreshold(TS4TS5UpperThresholdArray, TS4TS5UpperThresholdArray+5);
  std::vector<double> TS4TS5UpperCut(TS4TS5UpperCutArray, TS4TS5UpperCutArray+5);
  std::vector<double> TS4TS5LowerThreshold(TS4TS5LowerThresholdArray, TS4TS5LowerThresholdArray+7);
  std::vector<double> TS4TS5LowerCut(TS4TS5LowerCutArray, TS4TS5LowerCutArray+7);

  desc.add<std::vector<double> >("TS4TS5UpperThreshold", TS4TS5UpperThreshold);
  desc.add<std::vector<double> >("TS4TS5UpperCut", TS4TS5UpperCut);
  desc.add<std::vector<double> >("TS4TS5LowerThreshold", TS4TS5LowerThreshold);
  desc.add<std::vector<double> >("TS4TS5LowerCut", TS4TS5LowerCut);
  descriptions.add("hltHcalTowerNoiseCleaner",desc);
}

//
// member functions
//

void HLTHcalTowerNoiseCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace reco;

  //get the calo MET / MHT
  edm::Handle<CaloTowerCollection> tower_h;
  iEvent.getByLabel(TowerCollectionTag_,tower_h);
  
  std::set<unsigned int> noisyTowers;

  if(not tower_h.isValid()){ //No towers MET, don't do anything and accept the event
    edm::LogError("HLTHcalTowerNoiseCleaner") << "Input Tower Collection is not Valid";
    return;
  }

  //get the calotower topology
  edm::ESHandle<CaloTowerTopology> caloTowerTopology;
  iSetup.get<HcalRecNumberingRecord>().get(caloTowerTopology);
  
  // get the RBXs produced by RecoMET/METProducers/HcalNoiseInfoProducer
  edm::Handle<HcalNoiseRBXCollection> rbxs_h;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag_,rbxs_h);
  if(!rbxs_h.isValid()) {
    edm::LogWarning("HLTHcalTowerNoiseCleaner") << "Could not find HcalNoiseRBXCollection product named "
					      << HcalNoiseRBXCollectionTag_ << "." << std::endl;
    severity_=0;
  }
		
  // create a sorted set of the RBXs, ordered by energy
  noisedataset_t data;
  for(HcalNoiseRBXCollection::const_iterator it=rbxs_h->begin(); it!=rbxs_h->end(); ++it) {
    const HcalNoiseRBX &rbx=(*it);
    CommonHcalNoiseRBXData d(rbx, minRecHitE_, minLowHitE_, minHighHitE_, TS4TS5EnergyThreshold_,
			     TS4TS5UpperCut_, TS4TS5LowerCut_);
    data.insert(d);
  }
		
  // data is now sorted by RBX energy
  // only consider top N=numRBXsToConsider_ energy RBXs
  if(severity_>0){
    for(noisedataset_t::const_iterator it=data.begin();
	it!=data.end();
	it++) {
      
      
      bool passFilter=true;
      bool passEMF=true;
      if(it->energy()>minRBXEnergy_) {
	if(it->validRatio() && it->ratio()<minRatio_)        passFilter=false;
	else if(it->validRatio() && it->ratio()>maxRatio_)   passFilter=false;
	else if(it->numHPDHits()>=minHPDHits_)               passFilter=false;
	else if(it->numRBXHits()>=minRBXHits_)               passFilter=false;
	else if(it->numHPDNoOtherHits()>=minHPDNoOtherHits_) passFilter=false;
	else if(it->numZeros()>=minZeros_)                   passFilter=false;
	else if(it->minHighEHitTime()<minHighEHitTime_)      passFilter=false;
	else if(it->maxHighEHitTime()>maxHighEHitTime_)      passFilter=false;
	else if(!it->PassTS4TS5())                           passFilter=false;
	
	if(it->RBXEMF()<maxRBXEMF_){
	  passEMF=false;
	}
      }
      
      if((needEMFCoincidence_ && !passEMF && !passFilter) ||
	 (!needEMFCoincidence_ && !passFilter)) { // check for noise
	LogDebug("") << "HLTHcalTowerNoiseCleaner debug: Found a noisy RBX: "
		     << "energy=" << it->energy() << "; "
		     << "ratio=" << it->ratio() << "; "
		     << "# RBX hits=" << it->numRBXHits() << "; "
		     << "# HPD hits=" << it->numHPDHits() << "; "
		     << "# Zeros=" << it->numZeros() << "; "
		     << "min time=" << it->minHighEHitTime() << "; "
		     << "max time=" << it->maxHighEHitTime() << "; "
		     << "passTS4TS5=" << it->PassTS4TS5() << "; "
		     << "RBX EMF=" << it->RBXEMF()
		     << std::endl;
	// add calotowers associated with this RBX to the noise list
	edm::RefVector<CaloTowerCollection> noiseTowers = it->rbxTowers();
	edm::RefVector<CaloTowerCollection>::const_iterator noiseTowersIt;
	//add these calotowers to the noisy list
	for( noiseTowersIt = noiseTowers.begin(); noiseTowersIt != noiseTowers.end(); noiseTowersIt++){
	  edm::Ref<edm::SortedCollection<CaloTower> > tower_ref = *noiseTowersIt;
	  CaloTowerDetId id = tower_ref->id();
	  noisyTowers.insert( caloTowerTopology->denseIndex(id) );
	}}
    } // done with noise loop
  }//if(severity_>0)
  
  //output collection
  std::auto_ptr<CaloTowerCollection> OutputTowers(new CaloTowerCollection() );

  CaloTowerCollection::const_iterator inTowersIt;
  
  for(inTowersIt = tower_h->begin(); inTowersIt != tower_h->end(); inTowersIt++){
    const CaloTower & tower = (*inTowersIt);
    CaloTowerDetId id = tower.id();
    if(noisyTowers.find( caloTowerTopology->denseIndex(id) ) == noisyTowers.end()){ // the tower is not noisy
      OutputTowers->push_back(*inTowersIt);
    }
  }
  iEvent.put(OutputTowers);

}

 
