#ifndef HLTFiltCand_h
#define HLTFiltCand_h

/** \class HLTFiltCand
 *
 *  
 *  This class is an EDFilter implementing a very basic HLT trigger
 *  acting on candidates
 *
 *  $Date: 2006/05/12 18:13:01 $
 *  $Revision: 1.4 $
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

class HLTFiltCand : public edm::EDFilter {

   public:
      explicit HLTFiltCand(const edm::ParameterSet&);
      ~HLTFiltCand();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      std::string module_;  // module label for getting data from the event
};

#endif //HLTFiltCand_h
