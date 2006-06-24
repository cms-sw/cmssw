#ifndef HLTSimpleJet_h
#define HLTSimpleJet_h

/** \class HLTSimpleJet
 *
 *  
 *  This class is an EDFilter implementing a very basic HLT trigger
 *  for jets, cutting on the number of jets above a pt threshold
 *
 *  $Date: 2006/06/17 00:18:35 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class HLTSimpleJet : public edm::EDFilter {

#include "HLTrigger/HLTcore/interface/HLTadd.h"

   public:
      explicit HLTSimpleJet(const edm::ParameterSet&);
      ~HLTSimpleJet();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product contains jets
      double ptcut_;           // pt threshold in GeV 
      int    njcut_;           // number of jets required
};

#endif //HLTSimpleJet_h
