//
// $Id: PATElectronCleaner.h,v 1.1 2008/01/16 01:20:37 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATElectronCleaner_h
#define PhysicsTools_PatAlgos_PATElectronCleaner_h

/**
  \class    PATElectronCleaner PATElectronCleaner.h "PhysicsTools/PatAlgos/interface/PATElectronCleaner.h"
  \brief    Produces pat::Electron's

   The PATElectronCleaner produces analysis-level pat::Electron's starting from
   a collection of objects of ElectronType.

  \author   Steven Lowette, James Lamb
  \version  $Id: PATElectronCleaner.h,v 1.1 2008/01/16 01:20:37 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/interface/CleanerHelper.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include <string>


namespace pat {
  class PATElectronCleaner : public edm::EDProducer {
    public:
      explicit PATElectronCleaner(const edm::ParameterSet & iConfig);
      ~PATElectronCleaner();  

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      void removeDuplicates();

      edm::InputTag electronSrc_;
      bool          removeDuplicates_;
      pat::helper::CleanerHelper< reco::PixelMatchGsfElectron, 
                                  reco::PixelMatchGsfElectron,
                                  reco::PixelMatchGsfElectronCollection, 
                                  GreaterByPt<reco::PixelMatchGsfElectron> > helper_;
  
  };


}

#endif
