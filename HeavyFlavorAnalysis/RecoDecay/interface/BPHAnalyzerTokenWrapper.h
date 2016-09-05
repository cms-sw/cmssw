#ifndef BPHAnalyzerTokenWrapper_H
#define BPHAnalyzerTokenWrapper_H
/** \class BPHTokenWrapper
 *
 *  Description: 
 *    common interface to get objects from "old" and "new" CMSSW version
 *    in an uniform way
 *
 *  $Date: 2016-04-15 17:47:56 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
//#include "FWCore/Framework/interface/EDAnalyzer.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

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

