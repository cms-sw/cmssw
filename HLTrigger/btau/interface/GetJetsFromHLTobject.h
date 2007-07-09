#ifndef GetJetsFromHLTobject_h
#define GetJetsFromHLTobject_h

//
// Package:    GetJetsFromHLTobject
// Class:      GetJetsFromHLTobject
// 
/**\class GetJetsFromHLTobject 

 Description: 
   HLT algorithms produced HLTFilterObjects containing the jets etc.
   that caused the trigger to fire. This class gets these jets
   and stores references to them directly in the event in a RefVector.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ian Tomalin
//

//#include <memory>
//#include <string>
//#include <iostream>
//using namespace std;
//#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

//#include "DataFormats/Common/interface/Ref.h"
//#include "DataFormats/JetReco/interface/Jet.h"

//using namespace reco;

//
// class decleration
//

class GetJetsFromHLTobject : public edm::EDProducer {
   public:
      explicit GetJetsFromHLTobject(const edm::ParameterSet&);
      virtual void produce(edm::Event&, const edm::EventSetup&);

  private:
     edm::InputTag m_jets;
};

#endif //GetJetsFromHLTobject_h
