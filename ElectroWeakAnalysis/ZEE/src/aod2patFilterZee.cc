#ifndef aod2patFilterZee_H
#define aod2patFilterZee_H

/******************************************************************************
 *
 * Implementation Notes:
 *
 *   this is a filter that creates pat::Electrons without the need of
 *   running the PAT sequence
 *
 *   it is meant to be an interface of Wenu and Zee CandidateFilters
 *   for the October 2009 exercise
 *   it does make sense to implement the trigger requirement here
 *   but it will not be implemented in order to keep compatibolity with the
 *   old code
 *
 *
 * contact:
 * Nikolaos.Rompotis@Cern.ch
 *
 * Nikolaos Rompotis
 * Imperial College London
 *
 * 21 Sept 2009
 *
 *****************************************************************************/



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include <vector>
#include <iostream>
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
//
#include "TString.h"
#include "TMath.h"
#include "DataFormats/PatCandidates/interface/MET.h"


class aod2patFilterZee : public edm::EDFilter {
   public:
      explicit aod2patFilterZee(const edm::ParameterSet&);
      ~aod2patFilterZee();

   private:
      virtual void beginJob() override;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;
  //bool isInFiducial(double eta);

      // ----------member data ---------------------------
  //double ETCut_;
  //double METCut_;
  //double ETCut2ndEle_;
  //edm::InputTag triggerCollectionTag_;
  //edm::InputTag triggerEventTag_;
  //std::string hltpath_;
  //edm::InputTag hltpathFilter_;
  edm::InputTag electronCollectionTag_;
  edm::EDGetTokenT<reco::GsfElectronCollection> electronCollectionToken_;
  edm::InputTag metCollectionTag_;
  edm::EDGetTokenT<reco::CaloMETCollection> metCollectionToken_;

  //double BarrelMaxEta_;
  //double EndCapMaxEta_;
  //double EndCapMinEta_;
  //bool electronMatched2HLT_;
  //double electronMatched2HLT_DR_;
  //bool vetoSecondElectronEvents_;
};
#endif


aod2patFilterZee::aod2patFilterZee(const edm::ParameterSet& iConfig)
{

  electronCollectionTag_=iConfig.getUntrackedParameter<edm::InputTag>("electronCollectionTag");
  electronCollectionToken_=consumes<reco::GsfElectronCollection>(electronCollectionTag_);
  metCollectionTag_=iConfig.getUntrackedParameter<edm::InputTag>("metCollectionTag");
  metCollectionToken_=consumes<reco::CaloMETCollection>(metCollectionTag_);


  produces< pat::ElectronCollection >
    ("patElectrons").setBranchAlias("patElectrons");

  produces< pat::METCollection>("patCaloMets").setBranchAlias("patCaloMets");
  //produces< pat::METCollection>("patPfMets").setBranchAlias("patPfMets");
  //produces< pat::METCollection>("patTcMets").setBranchAlias("patTcMets");
  //produces< pat::METCollection>("patT1cMets").setBranchAlias("patT1cMets");

}

aod2patFilterZee::~aod2patFilterZee()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


bool
aod2patFilterZee::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  using namespace pat;
  // *************************************************************************
  // ELECTRONS
  // *************************************************************************
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByToken(electronCollectionToken_, gsfElectrons);
  if (!gsfElectrons.isValid()) {
    std::cout <<"aod2patFilterZee: Could not get electron collection with label: "
	      <<electronCollectionTag_ << std::endl;
    return false;
  }
  const reco::GsfElectronCollection *pElecs = gsfElectrons.product();
  // calculate your electrons
  auto_ptr<pat::ElectronCollection> patElectrons(new pat::ElectronCollection);
  for (reco::GsfElectronCollection::const_iterator elec = pElecs->begin();
       elec != pElecs->end(); ++elec) {
    reco::GsfElectron mygsfelec = *elec;
    pat::Electron myElectron(mygsfelec);
    // now set the isolations from the Gsf electron
    myElectron.setTrackIso(elec->dr03TkSumPt());
    myElectron.setEcalIso(elec->dr04EcalRecHitSumEt());
    myElectron.setHcalIso(elec->dr04HcalTowerSumEt());

    patElectrons->push_back(myElectron);
  }
  // *************************************************************************
  // METs
  // *************************************************************************
  edm::Handle<reco::CaloMETCollection> calomets;
  iEvent.getByToken(metCollectionToken_, calomets);
  if (! calomets.isValid()) {
    std::cout << "aod2patFilterZee: Could not get met collection with label: "
	      << metCollectionTag_ << std::endl;
    return false;
  }
  const  reco::CaloMETCollection *mycalomets =  calomets.product();
  auto_ptr<pat::METCollection> patCaloMets(new pat::METCollection);
  for (reco::CaloMETCollection::const_iterator met = mycalomets->begin();
       met != mycalomets->end(); ++ met ) {
    pat::MET mymet(*met);
    patCaloMets->push_back(mymet);
  }

  //
  // put everything in the event
  //
  iEvent.put( patElectrons, "patElectrons");
  iEvent.put( patCaloMets, "patCaloMets");
  //

  return true;

}

// ------------ method called once each job just before starting event loop  -
void
aod2patFilterZee::beginJob() {
}

// ------------ method called once each job just after ending the event loop  -
void
aod2patFilterZee::endJob() {
}


//define this as a plug-in
DEFINE_FWK_MODULE(aod2patFilterZee);
