//
// $Id: PATElectronProducer.h,v 1.3.2.2 2008/04/14 21:36:12 vadler Exp $
//

#ifndef PhysicsTools_PatAlgos_PATElectronProducer_h
#define PhysicsTools_PatAlgos_PATElectronProducer_h

/**
  \class    pat::PATElectronProducer PATElectronProducer.h "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
  \brief    Produces pat::Electron's

   The PATElectronProducer produces analysis-level pat::Electron's starting from
   a collection of objects of ElectronType.

  \author   Steven Lowette, James Lamb
  \version  $Id: PATElectronProducer.h,v 1.3.2.2 2008/04/14 21:36:12 vadler Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include <string>


namespace pat {


  class ObjectResolutionCalc;
  class TrackerIsolationPt;
  class CaloIsolationEnergy;


  class PATElectronProducer : public edm::EDProducer {

    public:

      explicit PATElectronProducer(const edm::ParameterSet & iConfig);
      ~PATElectronProducer();  

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      double electronID(const edm::Handle<edm::View<ElectronType> > & elecs, 
                        const edm::Handle<reco::ElectronIDAssociationCollection> & elecIDs,
	                unsigned int idx);
    private:

      // configurables
      edm::InputTag electronSrc_;
      bool          embedGsfTrack_;
      bool          embedSuperCluster_;
      bool          embedTrack_;
      bool          addGenMatch_;
      edm::InputTag genMatchSrc_;
      bool          addTrigMatch_;
      std::vector<edm::InputTag> trigPrimSrc_;
      bool          addResolutions_;
      bool          useNNReso_;
      std::string   electronResoFile_;
      bool          addElecID_;
      edm::InputTag elecIDSrc_;
      bool          addElecIDRobust_;
      edm::InputTag elecIDRobustSrc_;

      // tools
      ObjectResolutionCalc * theResoCalc_;
      GreaterByPt<Electron>       pTComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;
  };


}

#endif
