#ifndef HLTSimpleJet_h
#define HLTSimpleJet_h

/** \class HLTSimpleJet
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger for jets, cutting on the number of jets above a pt
 *  threshold
 *
 *  $Date: 2006/06/24 21:04:46 $
 *  $Revision: 1.7 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTSimpleJet : public HLTFilter {

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
