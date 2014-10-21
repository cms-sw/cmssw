// -*- C++ -*-
//
// Package:    Zanalyzer
// Class:      Zanalyzer
// 
/**\class Zanalyzer Zanalyzer.cc Zmonitoring/Zanalyzer/src/Zanalyzer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Vieri Candelise
//         Created:  Wed May 11 14:53:26 CEST 2011
//
//


// system include files
#include <memory>

// user include files

#include "DQM/Physics/src/EwkDQM.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include <string>
#include <cmath>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EcalZmassTask: public DQMEDAnalyzer {

public:
  explicit EcalZmassTask (const edm::ParameterSet &);
  ~EcalZmassTask () {}

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze (const edm::Event &, const edm::EventSetup &) override;

  const edm::EDGetTokenT<reco::GsfElectronCollection> electronCollectionToken_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> trackCollectionToken_;

  const std::string prefixME_;

  MonitorElement *h_ee_invMass_EB;
  MonitorElement *h_ee_invMass_EE;
  MonitorElement *h_ee_invMass_BB;
  MonitorElement *h_ee_invMass;
  MonitorElement *h_e1_et;
  MonitorElement *h_e2_et;
  MonitorElement *h_e1_eta;
  MonitorElement *h_e2_eta;
  MonitorElement *h_e1_phi;
  MonitorElement *h_e2_phi;
  MonitorElement *h_95_ee_invMass_EB;
  MonitorElement *h_95_ee_invMass_EE;
  MonitorElement *h_95_ee_invMass_BB;

};

EcalZmassTask::EcalZmassTask (const edm::ParameterSet & parameters) :
  electronCollectionToken_(consumes<reco::GsfElectronCollection>(parameters.getParameter < edm::InputTag > ("electronCollection"))),
  trackCollectionToken_(consumes<reco::GsfTrackCollection>(parameters.getParameter<edm::InputTag>("trackCollection"))),
  prefixME_(parameters.getUntrackedParameter < std::string > ("prefixME", ""))
{
}

// ------------ method called for each event  ------------
void
EcalZmassTask::analyze (const edm::Event & iEvent,
			const edm::EventSetup & iSetup)
{
  using namespace edm;
  Handle < reco::GsfElectronCollection > electronCollection;
  iEvent.getByToken (electronCollectionToken_, electronCollection);
  if (!electronCollection.isValid ()) return;

  //get GSF Tracks
  Handle < reco::GsfTrackCollection > gsftracks_h;
  iEvent.getByToken (trackCollectionToken_, gsftracks_h);

  bool isIsolatedBarrel;
  bool isIDBarrel;
  bool isConvertedBarrel;
  bool isIsolatedEndcap;
  bool isIDEndcap;
  bool isConvertedEndcap;

  int elIsAccepted=0;
  int elIsAcceptedEB=0;
  int elIsAcceptedEE=0;

  std::vector<TLorentzVector> LV;

  for (reco::GsfElectronCollection::const_iterator recoElectron =
	 electronCollection->begin ();
       recoElectron != electronCollection->end (); recoElectron++)
    {

      if (recoElectron->et () <= 25)
	continue;

      // Define Isolation variables
      double IsoTrk = (recoElectron->dr03TkSumPt () / recoElectron->et ());
      double IsoEcal =
	(recoElectron->dr03EcalRecHitSumEt () / recoElectron->et ());
      double IsoHcal =
	(recoElectron->dr03HcalTowerSumEt () / recoElectron->et ());
      double HE = (recoElectron->hcalOverEcal ());

      //Define ID variables

      float DeltaPhiTkClu = recoElectron->deltaPhiSuperClusterTrackAtVtx ();
      float DeltaEtaTkClu = recoElectron->deltaEtaSuperClusterTrackAtVtx ();
      float sigmaIeIe = recoElectron->sigmaIetaIeta ();

      //Define Conversion Rejection Variables

      float Dcot = recoElectron->convDcot ();
      float Dist = recoElectron->convDist ();
      int NumberOfExpectedInnerHits =
	recoElectron->gsfTrack ()->trackerExpectedHitsInner ().
	numberOfHits ();

      //quality flags

      isIsolatedBarrel = false;
      isIDBarrel = false;
      isConvertedBarrel = false;
      isIsolatedEndcap = false;
      isIDEndcap = false;
      isConvertedEndcap = false;


      /***** Barrel WP80 Cuts *****/

      if (fabs (recoElectron->eta ()) <= 1.4442)
	{

	  /* Isolation */
	  if (IsoTrk < 0.09 && IsoEcal < 0.07 && IsoHcal < 0.10)
	    {
	      isIsolatedBarrel = true;
	    }

	  /* Identification */
	  if (fabs (DeltaEtaTkClu) < 0.004 && fabs (DeltaPhiTkClu) < 0.06
	      && sigmaIeIe < 0.01 && HE < 0.04)
	    {
	      isIDBarrel = true;
	    }

	  /* Conversion Rejection */
	  if ((fabs (Dist) >= 0.02 || fabs (Dcot) >= 0.02)
	      && NumberOfExpectedInnerHits <= 1.0)
	    {
	      isConvertedBarrel = true;
	    }
	}

      if (isIsolatedBarrel && isIDBarrel && isConvertedBarrel) {
	elIsAccepted++;
	elIsAcceptedEB++;
	TLorentzVector b_e2(recoElectron->momentum ().x (),recoElectron->momentum ().y (),recoElectron->momentum ().z (), recoElectron->p ());
	LV.push_back(b_e2);
      }

      /***** Endcap WP80 Cuts *****/

      if (fabs (recoElectron->eta ()) >= 1.5660
	  && fabs (recoElectron->eta ()) <= 2.5000)
	{

	  /* Isolation */
	  if (IsoTrk < 0.04 && IsoEcal < 0.05 && IsoHcal < 0.025)
	    {
	      isIsolatedEndcap = true;
	    }

	  /* Identification */
	  if (fabs (DeltaEtaTkClu) < 0.007 && fabs (DeltaPhiTkClu) < 0.03
	      && sigmaIeIe < 0.031 && HE < 0.15)
	    {
	      isIDEndcap = true;
	    }

	  /* Conversion Rejection */
	  if ((fabs (Dcot) > 0.02 || fabs (Dist) > 0.02)
	      && NumberOfExpectedInnerHits <= 1.0)
	    {
	      isConvertedEndcap = true;
	    }
	}
      if (isIsolatedEndcap && isIDEndcap && isConvertedEndcap) {
	elIsAccepted++;
	elIsAcceptedEE++;
	TLorentzVector e_e2(recoElectron->momentum ().x (),recoElectron->momentum ().y (),recoElectron->momentum ().z (), recoElectron->p ());
	LV.push_back(e_e2);
      }

    }

  // Calculate the Z invariant masses

  if (elIsAccepted>1){
    double e_ee_invMass=0; 
    if (LV.size()==2){
      TLorentzVector e_pair = LV[0] + LV[1];
      e_ee_invMass = e_pair.M ();
    }  
		      
    if (elIsAcceptedEB==2){
      h_ee_invMass_BB->Fill(e_ee_invMass);
    }
    if (elIsAcceptedEE==2){
      h_ee_invMass_EE->Fill(e_ee_invMass);
    }
    if (elIsAcceptedEB==1 && elIsAcceptedEE==1){
      h_ee_invMass_EB->Fill(e_ee_invMass);
    }
		      
    LV.clear();
				  
  }
}

void
EcalZmassTask::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&)
{
  std::string logTraceName("EcalZmassTask");

  h_ee_invMass_EB = 0;
  h_ee_invMass_EE = 0;
  h_ee_invMass_BB = 0;

  LogTrace (logTraceName) << "Parameters initialization";

  iBooker.setCurrentFolder (prefixME_ + "/Zmass");	// Use folder with name of PAG

  h_ee_invMass_EB =
	iBooker.book1D ("Z peak - WP80 EB-EE",
                    "Z peak - WP80 EB-EE;InvMass (GeV)", 60, 60.0, 120.0);
  h_ee_invMass_EE =
	iBooker.book1D ("Z peak - WP80 EE-EE",
                    "Z peak - WP80 EE-EE;InvMass (Gev)", 60, 60.0, 120.0);
  h_ee_invMass_BB =
	iBooker.book1D ("Z peak - WP80 EB-EB",
                    "Z peak - WP80 EB-EB;InvMass (Gev)", 60, 60.0, 120.0);
}

//define this as a plug-in
DEFINE_FWK_MODULE (EcalZmassTask);
