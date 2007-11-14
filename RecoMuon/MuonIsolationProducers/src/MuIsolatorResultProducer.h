#ifndef MuonIsolationProducers_MuIsolatorResultProducer_H
#define MuonIsolationProducers_MuIsolatorResultProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

namespace edm { class EventSetup; }

class MuIsolatorResultProducer : public edm::EDProducer {

public:
  MuIsolatorResultProducer(const edm::ParameterSet&);

  virtual ~MuIsolatorResultProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  typedef muonisolation::MuIsoBaseIsolator Isolator;
  typedef Isolator::Result Result;
  typedef Isolator::ResultType ResultType;
  typedef std::vector<Result> Results;
  typedef Isolator::DepositContainer DepositContainer;

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


  enum InType { is_map = 0, is_vector };

  template<typename CT, typename OT>
    struct iohelper {
    };
  
  struct vector_trait {};
  struct map_trait {};
  
  template<typename CT, typename TR>
    struct reader {

    };
  

  template <typename CT>
    class CandMap {
  public:
    typedef typename std::vector<std::pair<typename reader<CT, typename iohelper<CT,bool>::trait>::ktype, DepositContainer> > Association;
    typedef typename Association::value_type value_type;
    typedef typename value_type::first_type key_type;
    Association& get() { return cMap_;}
    const Association& get() const {return cMap_;}
    const edm::Handle<CT> handle() const { return handle_;}
    void setHandle(const edm::Handle<CT>& rhs) { handle_ = rhs;}
  private:
    Association cMap_;
    edm::Handle<CT> handle_;
  };
  
  template<typename CT, typename OT, typename TR>
    struct writer {
      void writeImpl(edm::Event& event, const CandMap<CT>& candMapT, const Results& results);
    };

  template<typename CT>
    void callWhatProduces();
  
  template<typename CT>
    void produceImpl(edm::Event& event, const edm::EventSetup& eventSetup);

  template <typename CT>
    typename CT::size_type initAssociation(edm::Event& event, CandMap<CT>& candMapT) const;

  template <typename CT >
    void initVetos(std::vector<reco::MuIsoDeposit::Vetos>& vetos, CandMap<CT>& candMap) const;
  
  template <typename CT, typename OV>
    void writeOutImpl(edm::Event& event, const CandMap<CT>& candMapT, const Results& results) const;

  template <typename CT >
    void writeOut(edm::Event& event, const CandMap<CT>& candMap, const Results& results) const;
  
  edm::ParameterSet theConfig;
  std::vector<DepositConf> theDepositConfs;

  //!choose which muon vetos should be removed from all deposits  
  bool theRemoveOtherVetos;
  VetoCuts theVetoCuts;

  //!the isolator
  Isolator* theIsolator;
  ResultType theResultType;

  //! input config
  std::string theInputType;

};


template<typename CT, typename OM> inline
  void MuIsolatorResultProducer::writeOutImpl(edm::Event& event, const CandMap<CT>& candMapT, 
					     const Results& results) const {
  writer<CT,OM,typename iohelper<CT,bool>::trait> lWriter;
  lWriter.writeImpl(event, candMapT, results);
}

template<typename CT, typename OM>
struct MuIsolatorResultProducer::writer<CT, OM, MuIsolatorResultProducer::map_trait> {
  inline void writeImpl(edm::Event& event, const CandMap<CT>& candMapT, const Results& results) {
    std::string metname = "RecoMuon|MuonIsolationProducers";
    std::auto_ptr<OM> outMap(new OM());
    for (uint muI = 0; muI < results.size(); ++muI){
      outMap->insert(typename OM::key_type(candMapT.get()[muI].first), results[muI].val<typename OM::result_type>()); 
      LogDebug(metname)<<"Inserted into a map a value of "<<results[muI].val<typename OM::result_type>();
    }
    LogDebug(metname)<<"Before event.put: the size is "<<outMap->size();
    event.put(outMap);
    
  }
};


template<typename CT, typename OV>
struct MuIsolatorResultProducer::writer<CT, OV, MuIsolatorResultProducer::vector_trait> {
  inline void writeImpl(edm::Event& event, const CandMap<CT>& candMapT, const Results& results) {
    std::string metname = "RecoMuon|MuonIsolationProducers";
    std::auto_ptr<OV> outMap(new OV(candMapT.handle()->keyProduct()));
    for (uint muI = 0; muI < results.size(); ++muI){
      outMap->setValue(candMapT.get()[muI].first.key(), results[muI].val<typename OV::value_type::second_type>()); 
      LogDebug(metname)<<"Inserted into a map a value of "<<results[muI].val<typename OV::value_type::second_type>();
    }
    LogDebug(metname)<<"Before event.put: the size is "<<outMap->size();
    event.put(outMap);
  }
};

template <typename CT >
inline void MuIsolatorResultProducer::writeOut(edm::Event& event, const CandMap<CT>& candMapT, 
					       const Results& results) const {
  
  std::string metname = "RecoMuon|MuonIsolationProducers";
  typedef typename iohelper<CT, int>::otype OutInt;
  typedef typename iohelper<CT, float>::otype OutFloat;
  typedef typename iohelper<CT, bool>::otype OutBool;

  //! ugly
  LogDebug(metname)<<"Before calling writeOutMap  with result type "<<theIsolator->resultType();
  if (theResultType == Isolator::ISOL_INT_TYPE) writeOutImpl<CT,OutInt>(event, candMapT, results);
  if (theResultType == Isolator::ISOL_FLOAT_TYPE) writeOutImpl<CT,OutFloat>(event, candMapT, results);
  if (theResultType == Isolator::ISOL_BOOL_TYPE) writeOutImpl<CT,OutBool>(event, candMapT, results);
}


template <typename CT>
inline void MuIsolatorResultProducer::callWhatProduces() {
  if (theResultType == Isolator::ISOL_FLOAT_TYPE) produces<typename iohelper<CT, float>::otype>();
  if (theResultType == Isolator::ISOL_INT_TYPE  ) produces<typename iohelper<CT, int>::otype>();
  if (theResultType == Isolator::ISOL_BOOL_TYPE ) produces<typename iohelper<CT, bool>::otype>();      
}

#endif
