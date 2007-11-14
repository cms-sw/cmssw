#include "RecoMuon/MuonIsolationProducers/src/MuIsolatorResultProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"

#include "RecoMuon/MuonIsolation/interface/IsolatorByDeposit.h"
#include "RecoMuon/MuonIsolation/interface/IsolatorByDepositCount.h"
#include "RecoMuon/MuonIsolation/interface/IsolatorByNominalEfficiency.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

template<typename CT>
struct MuIsolatorResultProducer::reader<CT, MuIsolatorResultProducer::map_trait> {
  typedef typename CT::key_type ktype;
  static const ktype& getKey(typename CT::const_iterator it){ return it->key;}
  static const MuIsoDeposit* getValuePtr(typename CT::const_iterator& it){ return &it->val;}
};

template<typename CT>
struct MuIsolatorResultProducer::reader<CT, MuIsolatorResultProducer::vector_trait> {
  typedef typename CT::value_type::first_type ktype;
  static const ktype& getKey(typename CT::const_iterator it){ return it->first;}
  static const MuIsoDeposit* getValuePtr(typename CT::const_iterator& it){ return &it->second;}
};

template<typename KT, typename RT > 
struct MuIsolatorResultProducer::iohelper<AssociationMap<OneToValue<KT, MuIsoDeposit> >, RT> {
  typedef AssociationMap<OneToValue<KT, RT> > otype;
  typedef map_trait trait;
};

template<typename KT, typename RT > 
struct MuIsolatorResultProducer::iohelper<AssociationVector<KT, std::vector<MuIsoDeposit> >, RT >{
  typedef AssociationVector<KT, std::vector<RT> > otype;
  typedef vector_trait trait;
};


//! constructor with config
MuIsolatorResultProducer::MuIsolatorResultProducer(const ParameterSet& par) :
  theConfig(par),
  theRemoveOtherVetos(par.getParameter<bool>("RemoveOtherVetos")),
  theIsolator(0),
  theInputType(par.getParameter<std::string>("InputType"))
{
  LogDebug("RecoMuon|MuonIsolation")<<" MuIsolatorResultProducer CTOR";
  std::vector<edm::ParameterSet> depositInputs = 
    par.getParameter<std::vector<edm::ParameterSet> >("InputMuIsoDeposits");    

  std::vector<double> dWeights( depositInputs.size());
  std::vector<double> dThresholds( depositInputs.size());

  for (uint iDep = 0; iDep < depositInputs.size(); ++iDep){
    DepositConf dConf;
    dConf.tag = depositInputs[iDep].getParameter<edm::InputTag>("DepositTag");
    dConf.weight = depositInputs[iDep].getParameter<double>("DepositWeight");
    dConf.threshold = depositInputs[iDep].getParameter<double>("DepositThreshold");
    
    dWeights[iDep] = dConf.weight;
    dThresholds[iDep] = dConf.threshold;

    theDepositConfs.push_back(dConf);
  }

  edm::ParameterSet isoPset = par.getParameter<edm::ParameterSet>("IsolatorPSet");
  //! will switch to a factory at some point
  std::string isolatorType = isoPset.getParameter<std::string>("ComponentName");
  if ( isolatorType == "IsolatorByDeposit"){    
    std::string coneSizeType = isoPset.getParameter<std::string>("ConeSizeType");
    if (coneSizeType == "FixedConeSize"){
      float coneSize(isoPset.getParameter<double>("coneSize"));

      theIsolator = new IsolatorByDeposit(coneSize, dWeights, dThresholds);

      //      theIsolator = new IsolatorByDeposit(isoPset);
    } else if (coneSizeType == "CutsConeSize"){
//       Cuts cuts(isoPset.getParameter<edm::ParameterSet>("CutsPSet"));
      
//       theIsolator = new IsolatorByDeposit(coneSize, dWeights, dThresholds);
    }
  } else if ( isolatorType == "IsolatorByNominalEfficiency"){
    theIsolator = new IsolatorByNominalEfficiency("noname", std::vector<std::string>(1,"8:0.97"), dWeights);
  } else if ( isolatorType == "IsolatorByDepositCount"){    
    std::string coneSizeType = isoPset.getParameter<std::string>("ConeSizeType");
    if (coneSizeType == "FixedConeSize"){
      float coneSize(isoPset.getParameter<double>("coneSize"));
      
      theIsolator = new IsolatorByDepositCount(coneSize, dThresholds);
      
      //      theIsolator = new IsolatorByDeposit(isoPset);
    } else if (coneSizeType == "CutsConeSize"){
      //       Cuts cuts(isoPset.getParameter<edm::ParameterSet>("CutsPSet"));
      
      //       theIsolator = new IsolatorByDeposit(coneSize, dWeights, dThresholds);
    }
  }
  
  if (theIsolator == 0 ){
    edm::LogError("MuonIsolationProducers")<<"Failed to initialize an isolator";
  }
  theResultType = theIsolator->resultType();

  if (theInputType == "MapToMuons") callWhatProduces<MuIsoDepositAssociationMapToMuon>();
  if (theInputType == "MapToTracks") callWhatProduces<MuIsoDepositAssociationMap>();
  if (theInputType == "VectorToMuons") callWhatProduces<MuIsoDepositAssociationVectorToMuon>();
  if (theInputType == "VectorToTracks") callWhatProduces<MuIsoDepositAssociationVector>();
  if (theInputType == "VectorToCandidateView") callWhatProduces<MuIsoDepositAssociationVectorToCandidateView>();

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

//! destructor
MuIsolatorResultProducer::~MuIsolatorResultProducer(){
  if (theIsolator) delete theIsolator;
  LogDebug("RecoMuon|MuIsolatorResultProducer")<<" MuIsolatorResultProducer DTOR";
}


template<typename CT>
void MuIsolatorResultProducer::produceImpl(Event& event, const EventSetup& eventSetup){
  
  std::string metname = "RecoMuon|MuonIsolationProducers";

  // Take Deposits
  LogTrace(metname)<<" Taking the deposits: ";

  CandMap<CT> candMapT;
  
  typename CT::size_type colSize = initAssociation(event, candMapT);

  Results results(colSize);
  if (colSize != 0){
    if (theRemoveOtherVetos){
      std::vector<reco::MuIsoDeposit::Vetos> vetoDeps(colSize, reco::MuIsoDeposit::Vetos());
      initVetos<CT>(vetoDeps, candMapT);
    }

    for (uint muI=0; muI < colSize; ++muI){
      results[muI] = theIsolator->result(candMapT.get()[muI].second, *(candMapT.get()[muI].first));
      
      if (results[muI].typeF()!= theIsolator->resultType()){
	edm::LogError("MuonIsolationProducers")<<"Failed to get result from the isolator";
      }
    }
    
  }

  LogDebug(metname)<<"Ready to write out results of size "<<results.size();
  writeOut<CT>(event, candMapT, results);

}

//! build deposits
void MuIsolatorResultProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|MuonIsolationProducers";

  LogDebug(metname)<<" Muon Deposit producing..."
		   <<" BEGINING OF EVENT " <<"================================";


  if (theInputType == "MapToMuons") produceImpl<MuIsoDepositAssociationMapToMuon>(event, eventSetup);
  if (theInputType == "MapToTracks") produceImpl<MuIsoDepositAssociationMap>(event, eventSetup);
  if (theInputType == "VectorToMuons") produceImpl<MuIsoDepositAssociationVectorToMuon>(event, eventSetup);
  if (theInputType == "VectorToTracks") produceImpl<MuIsoDepositAssociationVector>(event, eventSetup);
  if (theInputType == "VectorToCandidateView") produceImpl<MuIsoDepositAssociationVectorToCandidateView>(event, eventSetup);
  
  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

template<typename CT>
typename CT::size_type
MuIsolatorResultProducer::initAssociation(Event& event, CandMap<CT>& candMapT) const {
  std::string metname = "RecoMuon|MuonIsolationProducers";
  
  typedef typename iohelper<CT,bool>::trait mytrait;
  for (uint iMap = 0; iMap < theDepositConfs.size(); ++iMap){
    Handle<CT> depH;
    event.getByLabel(theDepositConfs[iMap].tag, depH);
    LogDebug(metname) <<"Got Deposits of size "<<depH->size();
    
    candMapT.setHandle(depH);
    for (typename CT::const_iterator depHCI = depH->begin(); depHCI != depH->end(); ++depHCI){
      typename CandMap<CT>::value_type::first_type muPtr(reader<CT, mytrait>::getKey(depHCI));
      if (iMap == 0) candMapT.get().push_back(typename CandMap<CT>::value_type(muPtr, DepositContainer(theDepositConfs.size())));
      typename CandMap<CT>::Association::iterator muI = candMapT.get().begin();
      for (; muI != candMapT.get().end(); ++muI){
	if (muI->first == muPtr) break;
      }
      if (muI->first != muPtr){
	edm::LogError("MuonIsolationProducers")<<"Failed to align muon map";
      }
      muI->second[iMap].dep = reader<CT,mytrait>::getValuePtr(depHCI);	
    }
  }

  LogDebug(metname)<<"Picked and aligned nDeps = "<<candMapT.get().size();
  return candMapT.get().size();
}

template <typename CT >
void MuIsolatorResultProducer::initVetos(std::vector<reco::MuIsoDeposit::Vetos>& vetos, CandMap<CT>& candMapT) const {
  

  if (theRemoveOtherVetos){

    typename CT::size_type muI = 0;
    for (; muI < candMapT.get().size(); ++muI) {
      typename CandMap<CT>::value_type::first_type mu = candMapT.get()[muI].first;
      double d0 = ( mu->vx() * mu->py() - mu->vy() * mu->px() ) / mu->pt();
      if (theVetoCuts.selectAll 
	  || (fabs(mu->eta()) < theVetoCuts.muAbsEtaMax
	      && mu->pt() > theVetoCuts.muPtMin
	      && fabs(mu->vz()) < theVetoCuts.muAbsZMax
	      && fabs(d0) < theVetoCuts.muD0Max
	      )
	  ){
	for (uint iDep =0; iDep < candMapT.get()[muI].second.size(); ++iDep){
	  vetos[iDep].push_back(candMapT.get()[muI].second[iDep].dep->veto());
	}
      }
    }

    muI = 0;
    for (; muI < candMapT.get().size(); ++muI) {
      for(uint iDep =0; iDep < candMapT.get()[muI].second.size(); ++iDep){
	candMapT.get()[muI].second[iDep].vetos = &vetos[iDep];
      }
    }
  }
}


  
