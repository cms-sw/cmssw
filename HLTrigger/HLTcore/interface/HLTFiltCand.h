#ifndef HLTFiltCand_h
#define HLTFiltCand_h

/** \class HLTFiltCand
 *
 *  
 *  This class is an EDFilter implementing a very basic HLT trigger
 *  acting on candidates, requiring a gem triplet above pt cuts
 *
 *  $Date: 2006/05/18 16:57:45 $
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

class HLTFiltCand : public edm::EDFilter {

   public:
      explicit HLTFiltCand(const edm::ParameterSet&);
      ~HLTFiltCand();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      std::string srcphot_;  // module label for getting photons from the event
      std::string srcelec_;  // module label for getting electrons from the event
      std::string srcmuon_;  // module label for getting muons from the event
      double pt_phot_;       // pt cut for photon
      double pt_elec_;       // pt cut for electron
      double pt_muon_;       // pt cut for muons
};

#endif //HLTFiltCand_h
