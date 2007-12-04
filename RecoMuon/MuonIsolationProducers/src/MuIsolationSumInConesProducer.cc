#include "RecoMuon/MuonIsolationProducers/src/MuIsolationSumInConesProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;


/// constructor with config
MuIsolationSumInConesProducer::MuIsolationSumInConesProducer(const ParameterSet& par) :
  theConfig(par),
  theProduceFloat(par.getParameter<bool>("ProduceSum")),
  theProduceInt(par.getParameter<bool>("ProduceCount")),
  theRemoveOtherVetos(par.getParameter<bool>("RemoveOtherVetos"))
{
  LogDebug("RecoMuon|MuonIsolation")<<" MuIsolationSumInConesProducer CTOR";
  std::vector<edm::ParameterSet> depositInputs = 
    par.getParameter<std::vector<edm::ParameterSet> >("InputMuIsoDeposits");    
  for (uint iDep = 0; iDep < depositInputs.size(); ++iDep){
    DepositConf dConf;
    dConf.tag = depositInputs[iDep].getParameter<edm::InputTag>("DepositTag");
    dConf.weight = depositInputs[iDep].getParameter<double>("DepositWeight");
    dConf.threshold = depositInputs[iDep].getParameter<double>("DepositThreshold");
    
    theDepositConfs.push_back(dConf);
  }

  std::vector<edm::ParameterSet> coneInstances = 
    par.getParameter<std::vector<edm::ParameterSet> >("ConeSizes");
  for (uint iCone = 0; iCone < coneInstances.size(); ++iCone){
    theConeSizes.push_back(std::pair<double, std::string>
			   (coneInstances[iCone].getParameter<double>("ConeSize"),
			    coneInstances[iCone].getParameter<std::string>("ConeSizeName")));
    if (theProduceFloat) produces<MuIsoFloatAssociationMap>(theConeSizes[iCone].second);
    if (theProduceInt) produces<MuIsoIntAssociationMap>(theConeSizes[iCone].second);
  }
  
  if (theRemoveOtherVetos){
    edm::ParameterSet vetoPSet = par.getParameter<edm::ParameterSet>("VetoPSet");
    theVetoCuts.selectAll = vetoPSet.getParameter<bool>("SelectAll");
    if (! theVetoCuts.selectAll){
      theVetoCuts.muAbsEtaMax = vetoPSet.getParameter<double>("MuAbsEtaMax");
      theVetoCuts.muPtMin     = vetoPSet.getParameter<double>("MuPtMin");
      theVetoCuts.muAbsZMax   = vetoPSet.getParameter<double>("MuAbsZMax");
      theVetoCuts.muD0Max      = vetoPSet.getParameter<double>("MuD0Max");    
    }
  }
}

/// destructor
MuIsolationSumInConesProducer::~MuIsolationSumInConesProducer(){
  LogDebug("RecoMuon|MuIsolationSumInConesProducer")<<" MuIsolationSumInConesProducer DTOR";
}

/// build deposits
void MuIsolationSumInConesProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|MuonIsolationProducers";

  LogDebug(metname)<<" Muon Deposit producing..."
		   <<" BEGINING OF EVENT " <<"================================";

  // Take Deposits
  LogTrace(metname)<<" Taking the deposits: ";
  std::vector<Handle<MuIsoDepositAssociationMap> > deposits; deposits.clear();
  for (uint iDep = 0; iDep < theDepositConfs.size(); ++iDep){
    LogTrace(metname)<<" Taking : "<<theDepositConfs[iDep].tag;
    deposits.push_back(Handle<MuIsoDepositAssociationMap>());
    event.getByLabel(theDepositConfs[iDep].tag, deposits[iDep]);
    LogDebug(metname) <<"Got Deposits of size "<<deposits[iDep]->size();
  }


  uint nCones = theConeSizes.size();

  static const uint MAX_CONES=10;
  std::auto_ptr<MuIsoFloatAssociationMap> sumMaps[MAX_CONES];
  std::auto_ptr<MuIsoIntAssociationMap> countMaps[MAX_CONES];
  for (uint i =0;i<nCones; ++i){
    sumMaps[i] =  std::auto_ptr<MuIsoFloatAssociationMap>(new MuIsoFloatAssociationMap());
    countMaps[i] =  std::auto_ptr<MuIsoIntAssociationMap>(new MuIsoIntAssociationMap());
  }


  //if a muon passes theVetoCuts, remove its veto from all other deposits
  std::vector<reco::MuIsoDeposit::Vetos> vetoDeps(deposits.size(), reco::MuIsoDeposit::Vetos());
  if (theRemoveOtherVetos){
    MuIsoDepositAssociationMap::const_iterator muDep0CI = deposits[0]->begin();
    for (; muDep0CI != deposits[0]->end(); ++muDep0CI) {
      TrackRef mu(muDep0CI->key);
      if (theVetoCuts.selectAll 
	  || (fabs(mu->eta()) < theVetoCuts.muAbsEtaMax
	      && mu->pt() > theVetoCuts.muPtMin
	      && fabs(mu->vz()) < theVetoCuts.muAbsZMax
	      && fabs(mu->d0()) < theVetoCuts.muD0Max
	      )
	  ){
	for (uint iDep =0; iDep < deposits.size(); ++iDep){
	  const MuIsoDeposit* dep = &(*deposits[iDep])[mu];
	  vetoDeps[iDep].push_back(dep->veto());
	}
      }
    }
  }

  MuIsoDepositAssociationMap::const_iterator muDep0CI = deposits[0]->begin();
  for (; muDep0CI != deposits[0]->end(); ++muDep0CI) {
    TrackRef mu(muDep0CI->key);
    std::vector<float> sumDep(nCones, 0);
    std::vector<int> countDep(nCones, 0);
    for (uint iDep =0; iDep < deposits.size(); ++iDep){
      const MuIsoDeposit* dep = &(*deposits[iDep])[mu];
      for (uint iCone=0; iCone < nCones; ++iCone){
	std::pair<double, int> sumAndCount = 
	  dep->depositAndCountWithin(theConeSizes[iCone].first, 
				     vetoDeps[iDep], 
				     theDepositConfs[iDep].threshold);
	sumDep[iCone] += sumAndCount.first*theDepositConfs[iDep].weight;
	countDep[iCone] += sumAndCount.second;
      }
    }
    for (uint iCone=0; iCone < nCones; ++iCone){
      sumMaps[iCone]->insert(mu, sumDep[iCone]);
      countMaps[iCone]->insert(mu, countDep[iCone]);
    }
  }


  for (uint iMap = 0; iMap < nCones; ++iMap){
    if (theProduceFloat) event.put(sumMaps[iMap], theConeSizes[iMap].second);
    if (theProduceInt) event.put(countMaps[iMap], theConeSizes[iMap].second);
  }
  LogTrace(metname) <<" END OF EVENT " <<"================================";
}
