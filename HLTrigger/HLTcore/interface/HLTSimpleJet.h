#ifndef HLTSimpleJet_h
#define HLTSimpleJet_h

/** \class HLTSimpleJet
 *
 *  
 *  This class is an EDFilter implementing a very basic HLT trigger
 *  for jets, cutting on the number of jets above a pt threshold
 *
 *  $Date: 2006/04/26 09:27:44 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>
//
// class decleration
//

class HLTSimpleJet : public edm::EDFilter {

   public:
      explicit HLTSimpleJet(const edm::ParameterSet&);
      ~HLTSimpleJet();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      std::string module_;  // module label for input jets
      double ptcut_;        // pt threshold in GeV 
      int    njcut_;        // number of jets required
};

#endif //HLTSimpleJet_h
