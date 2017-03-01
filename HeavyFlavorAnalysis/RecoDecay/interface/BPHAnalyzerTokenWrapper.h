#ifndef HeavyFlavorAnalysis_RecoDecay_BPHAnalyzerTokenWrapper_h
#define HeavyFlavorAnalysis_RecoDecay_BPHAnalyzerTokenWrapper_h
/** \classes BPHModuleWrapper, BPHTokenWrapper and BPHAnalyzerWrapper
 *
 *  Description: 
 *    Common interfaces to define modules and get objects
 *    from "old" and "new" CMSSW version in an uniform way
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHModuleWrapper {
 public:
  typedef edm::   one::EDAnalyzer<>    one_analyzer;
  typedef edm::   one::EDProducer<>    one_producer;
  typedef edm::stream::EDAnalyzer<> stream_analyzer;
  typedef edm::stream::EDProducer<> stream_producer;
};

template<class Obj>
class BPHTokenWrapper {
 public:
  typedef typename edm::EDGetTokenT<Obj> type;
  bool get( const edm::Event& ev,
            edm::Handle<Obj>& obj ) {
    return ev.getByToken( token, obj );
  }
  type token;
};

template<class T>
class BPHAnalyzerWrapper: public T {
 protected:
  template<class Obj>
  void consume( BPHTokenWrapper<Obj>& tw,
                const std::string& label ) {
    edm::InputTag tag( label );
    tw.token = this->template consumes<Obj>( tag );
    return;
  }
  template<class Obj>
  void consume( BPHTokenWrapper<Obj>& tw,
                const edm::InputTag& tag ) {
    tw.token = this->template consumes<Obj>( tag );
    return;
  }
};

#endif

