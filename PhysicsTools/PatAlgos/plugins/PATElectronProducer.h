//
// $Id: PATElectronProducer.h,v 1.1 2008/03/06 09:23:10 llista Exp $
//

#ifndef PhysicsTools_PatAlgos_PATElectronProducer_h
#define PhysicsTools_PatAlgos_PATElectronProducer_h

/**
  \class    pat::PATElectronProducer PATElectronProducer.h "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
  \brief    Produces pat::Electron's

   The PATElectronProducer produces analysis-level pat::Electron's starting from
   a collection of objects of ElectronType.

  \author   Steven Lowette, James Lamb
  \version  $Id: PATElectronProducer.h,v 1.1 2008/03/06 09:23:10 llista Exp $
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

#include "DataFormats/PatCandidates/interface/Electron.h"

#include <string>


namespace pat {


  class ObjectResolutionCalc;
  class TrackerIsolationPt;
  class CaloIsolationEnergy;
  class LeptonLRCalc;


  class PATElectronProducer : public edm::EDProducer {

    public:

      explicit PATElectronProducer(const edm::ParameterSet & iConfig);
      ~PATElectronProducer();  

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      double electronID(const edm::Handle<edm::View<ElectronType> > & elecs, 
                        const edm::Handle<reco::ElectronIDAssociationCollection> & elecIDs,
	                unsigned int idx);
      void setEgammaIso(Electron & anElectron,
                        const edm::Handle<edm::View<ElectronType> > & elecs,
                        const edm::Handle<edm::ValueMap<float> > tkIso,
                        const edm::Handle<edm::ValueMap<float> >    tkNumIso,
                        const edm::Handle<edm::ValueMap<float> > ecalIso,
                        const edm::Handle<edm::ValueMap<float> > hcalIso,
                        unsigned int idx);

    private:

      // configurables
      edm::InputTag electronSrc_;
      bool          addGenMatch_;
      edm::InputTag genMatchSrc_;
      bool          addResolutions_;
      bool          useNNReso_;
      std::string   electronResoFile_;
      bool          addTrkIso_;
      edm::InputTag tracksSrc_;
      bool          addCalIso_;
      edm::InputTag towerSrc_;
      bool          addElecID_;
      edm::InputTag elecIDSrc_;
      bool          addElecIDRobust_;
      edm::InputTag elecIDRobustSrc_;
      bool          addLRValues_;
      std::string   electronLRFile_;
      bool          addEgammaIso_;
      edm::InputTag egammaTkIsoSrc_;
      edm::InputTag egammaTkNumIsoSrc_;
      edm::InputTag egammaEcalIsoSrc_;
      edm::InputTag egammaHcalIsoSrc_;

      // tools
      ObjectResolutionCalc * theResoCalc_;
      TrackerIsolationPt   * trkIsolation_;
      CaloIsolationEnergy  * calIsolation_;
      LeptonLRCalc         * theLeptonLRCalc_;
      GreaterByPt<Electron>       pTComparator_;

  };


}

#endif
