#ifndef MuonIsolationProducers_MuIsolatorResultProducer_H
#define MuonIsolationProducers_MuIsolatorResultProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"


#include <string>

namespace edm { class EventSetup; }

struct muisorhelper {
  struct map_trait {};
  struct vector_trait {};

  template<typename CT, typename RT = void>
  struct adapter {
    typedef CT inprod_type;
    typedef RT result_type;
    typedef void trait_type;
  };

  template<typename CT, typename TR = typename adapter<CT>::trait_type>
    struct reader { 
    };

  typedef muonisolation::MuIsoBaseIsolator Isolator;
  typedef Isolator::Result Result;
  typedef Isolator::ResultType ResultType;
  typedef std::vector<Result> Results;
  typedef Isolator::DepositContainer DepositContainer;

  template<typename CT>
  class CandMap {
  public:
    typedef typename reader<CT>::key_type key_type;
    typedef DepositContainer value_type;
    typedef std::pair<key_type, value_type> pair_type;
    typedef typename std::vector<pair_type > map_type;
    typedef typename map_type::iterator iterator;

    map_type& get() { return cMap_;}
    const map_type& get() const {return cMap_;}
    const edm::Handle<CT> handle() const { return handle_;}
    void setHandle(const edm::Handle<CT>& rhs) { handle_ = rhs;}
  private:
    map_type cMap_;
    edm::Handle<CT> handle_;
  };
  
  template<typename CT, typename RT, typename TR = typename adapter<CT>::trait_type>
    struct writer { 
      void writeImpl(edm::Event& event, const CandMap<CT>& candMapT, const Results& results);
    };

};

template<typename KT, typename RT>
  struct muisorhelper::adapter<edm::AssociationMap<edm::OneToValue<KT, reco::MuIsoDeposit> >, RT > {
    typedef KT keyprod_type;
    typedef muisorhelper::map_trait trait_type;
    
    typedef edm::AssociationMap<edm::OneToValue<KT, RT> > outprod_type;
  };

template<typename KT, typename RT>
  struct muisorhelper::adapter<edm::AssociationVector<KT, std::vector<reco::MuIsoDeposit> >, RT >{
    typedef KT keyprod_type;
    typedef muisorhelper::vector_trait trait_type;
    
    typedef edm::AssociationVector<KT, std::vector<RT> > outprod_type;
  };

template<typename CT>
struct muisorhelper::reader<CT, muisorhelper::map_trait>{
  typedef typename CT::key_type key_type;
  static const key_type& getKey(typename CT::const_iterator it){ return it->key;}
  static const reco::MuIsoDeposit* getValuePtr(typename CT::const_iterator& it){ return &it->val;}  
};

template<typename CT>
struct muisorhelper::reader<CT, muisorhelper::vector_trait>{
  typedef typename CT::value_type::first_type key_type;
  static const key_type& getKey(typename CT::const_iterator it){ return it->first;}
  static const reco::MuIsoDeposit* getValuePtr(typename CT::const_iterator& it){ return &it->second;}  
};

template<typename CT, typename RT>
  struct muisorhelper::writer<CT, RT, muisorhelper::map_trait> {
  inline void writeImpl(edm::Event& event, 
			const CandMap<CT>& candMapT, 
			const Results& results) {
    std::string metname = "RecoMuon|MuonIsolationProducers";
    typedef typename muisorhelper::adapter<CT,RT>::outprod_type OM;

    std::auto_ptr<OM> outMap(new OM());
    for (uint muI = 0; muI < results.size(); ++muI){
      outMap->insert(typename OM::key_type(candMapT.get()[muI].first), results[muI].val<typename OM::result_type>()); 
      LogDebug(metname)<<"Inserted into a map a value of "<<results[muI].val<typename OM::result_type>();
    }
    LogDebug(metname)<<"Before event.put: the size is "<<outMap->size();
    event.put(outMap);
    
  }
};


template<typename CT, typename RT>
  struct muisorhelper::writer<CT, RT, muisorhelper::vector_trait> {
    inline void writeImpl(edm::Event& event, 
			  const CandMap<CT>& candMapT, 
			  const Results& results) {
      std::string metname = "RecoMuon|MuonIsolationProducers";

      typedef typename muisorhelper::adapter<CT,RT>::outprod_type OV;
      std::auto_ptr<OV> outMap(new OV(candMapT.handle()->keyProduct()));
      for (uint muI = 0; muI < results.size(); ++muI){
	outMap->setValue(candMapT.get()[muI].first.key(), results[muI].val<typename OV::value_type::second_type>()); 
	LogDebug(metname)<<"Inserted into a map a value of "<<results[muI].val<typename OV::value_type::second_type>();
      }
      LogDebug(metname)<<"Before event.put: the size is "<<outMap->size();
      event.put(outMap);
    }
  };

template <typename CT>
class MuIsolatorResultProducer : public edm::EDProducer {

public:
  MuIsolatorResultProducer(const edm::ParameterSet&);

  virtual ~MuIsolatorResultProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  typedef muisorhelper::Isolator Isolator;
  typedef muisorhelper::Result Result;
  typedef muisorhelper::ResultType ResultType;
  typedef muisorhelper::Results Results;
  typedef muisorhelper::DepositContainer DepositContainer;

  typedef muisorhelper::CandMap<CT> CandMap;

  struct DepositConf { 
    edm::InputTag tag; 
    double weight; 
    double threshold;
  };

  struct VetoCuts { 
    bool selectAll; 
    double muAbsEtaMax; 
    double muPtMin;
    double muAbsZMax;
    double muD0Max;
  };
  


  void callWhatProduces();
  
  typename CT::size_type initAssociation(edm::Event& event, CandMap& candMapT) const;

  void initVetos(std::vector<reco::MuIsoDeposit::Vetos*>& vetos, CandMap& candMap) const;
  
  template <typename OV>
    void writeOutImpl(edm::Event& event, const CandMap& candMapT, const Results& results) const;

  void writeOut(edm::Event& event, const CandMap& candMap, const Results& results) const;
  
  edm::ParameterSet theConfig;
  std::vector<DepositConf> theDepositConfs;
  
  //!choose which muon vetos should be removed from all deposits  
  bool theRemoveOtherVetos;
  VetoCuts theVetoCuts;

  //!the isolator
  Isolator* theIsolator;
  ResultType theResultType;

};



template<typename CT>
template<typename RT> inline
void MuIsolatorResultProducer<CT>::writeOutImpl(edm::Event& event, const CandMap& candMapT, 
						const Results& results) const {
  muisorhelper::writer<CT, RT> lWriter;
  lWriter.writeImpl(event, candMapT, results);
}


template<typename CT>
inline void MuIsolatorResultProducer<CT>::writeOut(edm::Event& event, 
						   const CandMap& candMapT, 
						   const Results& results) const {
  
  std::string metname = "RecoMuon|MuonIsolationProducers";
  LogDebug(metname)<<"Before calling writeOutMap  with result type "<<theIsolator->resultType();

  if (theResultType == Isolator::ISOL_INT_TYPE) writeOutImpl<int>(event, candMapT, results);
  if (theResultType == Isolator::ISOL_FLOAT_TYPE) writeOutImpl<float>(event, candMapT, results);
  if (theResultType == Isolator::ISOL_BOOL_TYPE) writeOutImpl<bool>(event, candMapT, results);
}


template<typename CT>
inline void MuIsolatorResultProducer<CT>::callWhatProduces() {
  if (theResultType == Isolator::ISOL_FLOAT_TYPE) produces<typename muisorhelper::adapter<CT,float>::outprod_type>();
  if (theResultType == Isolator::ISOL_INT_TYPE  ) produces<typename muisorhelper::adapter<CT,int>::outprod_type>();
  if (theResultType == Isolator::ISOL_BOOL_TYPE ) produces<typename muisorhelper::adapter<CT,bool>::outprod_type>();      
}

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

//! constructor with config
template<typename CT>
MuIsolatorResultProducer<CT>::MuIsolatorResultProducer(const ParameterSet& par) :
  theConfig(par),
  theRemoveOtherVetos(par.getParameter<bool>("RemoveOtherVetos")),
  theIsolator(0)
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

  callWhatProduces();

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
template<typename CT>
MuIsolatorResultProducer<CT>::~MuIsolatorResultProducer(){
  if (theIsolator) delete theIsolator;
  LogDebug("RecoMuon|MuIsolatorResultProducer")<<" MuIsolatorResultProducer DTOR";
}


template<typename CT>
void MuIsolatorResultProducer<CT>::produce(Event& event, const EventSetup& eventSetup){
  
  std::string metname = "RecoMuon|MuonIsolationProducers";
  LogDebug(metname)<<" Muon Deposit producing..."
		   <<" BEGINING OF EVENT " <<"================================";

  CandMap candMapT;
  
  typename CT::size_type colSize = initAssociation(event, candMapT);

  std::vector<reco::MuIsoDeposit::Vetos*> vetoDeps(theDepositConfs.size(), 0);
  Results results(colSize);

  if (colSize != 0){
    if (theRemoveOtherVetos){

      initVetos(vetoDeps, candMapT);
    }

    for (uint muI=0; muI < colSize; ++muI){
      results[muI] = theIsolator->result(candMapT.get()[muI].second, *(candMapT.get()[muI].first));
      
      if (results[muI].typeF()!= theIsolator->resultType()){
	edm::LogError("MuonIsolationProducers")<<"Failed to get result from the isolator";
      }
    }
    
  }

  LogDebug(metname)<<"Ready to write out results of size "<<results.size();
  writeOut(event, candMapT, results);

  for(uint iDep = 0; iDep< vetoDeps.size(); ++iDep){
    //! do cleanup
    if (vetoDeps[iDep]){
      delete vetoDeps[iDep];
      vetoDeps[iDep] = 0;
    }
  }
}

template<typename CT>
typename CT::size_type
MuIsolatorResultProducer<CT>::initAssociation(Event& event, CandMap& candMapT) const {
  std::string metname = "RecoMuon|MuonIsolationProducers";
  
  typedef typename muisorhelper::reader<CT> myreader;

  for (uint iMap = 0; iMap < theDepositConfs.size(); ++iMap){
    Handle<CT> depH;
    event.getByLabel(theDepositConfs[iMap].tag, depH);
    LogDebug(metname) <<"Got Deposits of size "<<depH->size();
    
    candMapT.setHandle(depH);
    for (typename CT::const_iterator depHCI = depH->begin(); depHCI != depH->end(); ++depHCI){
      typename CandMap::key_type muPtr(myreader::getKey(depHCI));
      if (iMap == 0) candMapT.get().push_back(typename CandMap::pair_type(muPtr, DepositContainer(theDepositConfs.size())));
      typename CandMap::iterator muI = candMapT.get().begin();
      for (; muI != candMapT.get().end(); ++muI){
	if (muI->first == muPtr) break;
      }
      if (muI->first != muPtr){
	edm::LogError("MuonIsolationProducers")<<"Failed to align muon map";
      }
      muI->second[iMap].dep = myreader::getValuePtr(depHCI);	
    }
  }

  LogDebug(metname)<<"Picked and aligned nDeps = "<<candMapT.get().size();
  return candMapT.get().size();
}

template <typename CT >
void MuIsolatorResultProducer<CT>::initVetos(std::vector<reco::MuIsoDeposit::Vetos*>& vetos, CandMap& candMapT) const {
  std::string metname = "RecoMuon|MuonIsolationProducers";
  

  if (theRemoveOtherVetos){
    LogDebug(metname)<<"Start checking for vetos based on input/expected vetos.size of "<<vetos.size()
		     <<" passed at "<<&vetos
		     <<" and an input map.size of "<<candMapT.get().size();

    typename CT::size_type muI = 0;
    for (; muI < candMapT.get().size(); ++muI) {
      typename CandMap::key_type mu = candMapT.get()[muI].first;
      double d0 = ( mu->vx() * mu->py() - mu->vy() * mu->px() ) / mu->pt();
      LogDebug(metname)<<"Muon at index "<<muI;
      if (theVetoCuts.selectAll 
	  || (fabs(mu->eta()) < theVetoCuts.muAbsEtaMax
	      && mu->pt() > theVetoCuts.muPtMin
	      && fabs(mu->vz()) < theVetoCuts.muAbsZMax
	      && fabs(d0) < theVetoCuts.muD0Max
	      )
	  ){
	LogDebug(metname)<<"muon passes the cuts";
	for (uint iDep =0; iDep < candMapT.get()[muI].second.size(); ++iDep){
	  if (vetos[iDep] == 0) vetos[iDep] = new reco::MuIsoDeposit::Vetos();

	  vetos[iDep]->push_back(candMapT.get()[muI].second[iDep].dep->veto());
	}
      }
    }

    LogDebug(metname)<<"Assigning vetos";
    muI = 0;
    for (; muI < candMapT.get().size(); ++muI) {
      for(uint iDep =0; iDep < candMapT.get()[muI].second.size(); ++iDep){
	candMapT.get()[muI].second[iDep].vetos = vetos[iDep];
      }
    }
    LogDebug(metname)<<"Done with vetos";
  }
}


  
#endif
