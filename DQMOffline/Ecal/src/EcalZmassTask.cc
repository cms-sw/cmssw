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
// $Id$
//
//


// system include files
#include <memory>

// user include files

#include "DQM/Physics/src/EwkDQM.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include <string>
#include <cmath>
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

using namespace std;
using namespace edm;
using namespace reco;


class DQMStore;
class MonitorElement;

std::string prefixME_;

class
  EcalZmassTask:
  public
  edm::EDAnalyzer
{
public:
  explicit
  EcalZmassTask (const edm::ParameterSet &);
   ~
  EcalZmassTask ();

  static void
  fillDescriptions (edm::ConfigurationDescriptions & descriptions);

private:
  virtual void
  beginJob ();
  virtual void
  analyze (const edm::Event &, const edm::EventSetup &);
  virtual void
  endJob ();

  virtual void
  beginRun (edm::Run const &, edm::EventSetup const &);
  virtual void
  endRun (edm::Run const &, edm::EventSetup const &);
  virtual void
  beginLuminosityBlock (edm::LuminosityBlock const &,
			edm::EventSetup const &);
  virtual void
  endLuminosityBlock (edm::LuminosityBlock const &, edm::EventSetup const &);





//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

  edm::InputTag
    theElectronCollectionLabel;

  MonitorElement *
    h_ee_invMass_EB;
  MonitorElement *
    h_ee_invMass_EE;
  MonitorElement *
    h_ee_invMass_BB;
  MonitorElement *
    h_ee_invMass;
  MonitorElement *
    h_e1_et;
  MonitorElement *
    h_e2_et;
  MonitorElement *
    h_e1_eta;
  MonitorElement *
    h_e2_eta;
  MonitorElement *
    h_e1_phi;
  MonitorElement *
    h_e2_phi;
  MonitorElement *
    h_95_ee_invMass_EB;
  MonitorElement *
    h_95_ee_invMass_EE;
  MonitorElement *
    h_95_ee_invMass_BB;

};

EcalZmassTask::EcalZmassTask (const edm::ParameterSet & parameters)
{
  prefixME_ = parameters.getUntrackedParameter < string > ("prefixME", "");
  theElectronCollectionLabel =
    parameters.getParameter < InputTag > ("electronCollection");

}



EcalZmassTask::~EcalZmassTask ()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}




//
// member functions
//

// ------------ method called for each event  ------------
void
EcalZmassTask::analyze (const edm::Event & iEvent,
			const edm::EventSetup & iSetup)
{

  LogTrace (logTraceName) << "Analysis of event # ";

  using namespace edm;
  Handle < GsfElectronCollection > electronCollection;
  iEvent.getByLabel (theElectronCollectionLabel, electronCollection);
  if (!electronCollection.isValid ())
    return;

  //get GSF Tracks
  Handle < reco::GsfTrackCollection > gsftracks_h;
  iEvent.getByLabel ("electronGsfTracks", gsftracks_h);

  // Find the highest and 2nd highest electron in the barrel/endcap that will be selected with WP80

  float b_electron_et = -8.0;
  float b_electron_eta = -8.0;
  float b_electron_phi = -8.0;
  float b_electron2_et = -9.0;
  float b_electron2_eta = -9.0;
  float b_electron2_phi = -9.0;
  float b_ee_invMass = -9.0;
  TLorentzVector b_e1, b_e2;

  float e_electron_et = -8.0;
  float e_electron_eta = -8.0;
  float e_electron_phi = -8.0;
  float e_electron2_et = -9.0;
  float e_electron2_eta = -9.0;
  float e_electron2_phi = -9.0;
  float e_ee_invMass = -9.0;
  TLorentzVector e_e1, e_e2;


  float eb_electron_et = -8.0;
  float eb_electron_eta = -8.0;
  float eb_electron_phi = -8.0;
  float eb_electron2_et = -9.0;
  float eb_electron2_eta = -9.0;
  float eb_electron2_phi = -9.0;
  float eb_ee_invMass = -9.0;
  TLorentzVector eb_e1, eb_e2;

  float be_electron_et = -8.0;
  float be_electron_eta = -8.0;
  float be_electron_phi = -8.0;
  float be_electron2_et = -9.0;
  float be_electron2_eta = -9.0;
  float be_electron2_phi = -9.0;
  TLorentzVector be_e1, be_e2;

/*
  // Find the highest and 2nd highest electron in the barrel/endcap that will be selected with WP95

  float b_95_electron_et = -8.0;
  float b_95_electron_eta = -8.0;
  float b_95_electron_phi = -8.0;
  float b_95_electron2_et = -9.0;
  float b_95_electron2_eta = -9.0;
  float b_95_electron2_phi = -9.0;
  float b_95_ee_invMass = -9.0;
  TLorentzVector b_95_e1, b_95_e2;

  float e_95_electron_et = -8.0;
  float e_95_electron_eta = -8.0;
  float e_95_electron_phi = -8.0;
  float e_95_electron2_et = -9.0;
  float e_95_electron2_eta = -9.0;
  float e_95_electron2_phi = -9.0;
  float e_95_ee_invMass = -9.0;
  TLorentzVector e_95_e1, e_95_e2;

  float eb_95_electron_et = -8.0;
  float eb_95_electron_eta = -8.0;
  float eb_95_electron_phi = -8.0;
  float eb_95_electron2_et = -9.0;
  float eb_95_electron2_eta = -9.0;
  float eb_95_electron2_phi = -9.0;
  float eb_95_ee_invMass = -9.0;
  TLorentzVector eb_95_e1, eb_95_e2;
  
  float be_95_electron_et = -8.0;	            	
  float be_95_electron_eta = -8.0;
  float be_95_electron_phi = -8.0;
  float be_95_electron2_et = -9.0;
  float be_95_electron2_eta = -9.0;
  float be_95_electron2_phi = -9.0;
  float be_95_ee_invMass = -9.0;
  TLorentzVector be_95_e1, be_95_e2;
*/

  bool isBarrelElectrons;
  bool isEndcapElectrons;
  bool isIsolatedBarrel;
  bool isIDBarrel;
  bool isConvertedBarrel;
  bool isIsolatedEndcap;
  bool isIDEndcap;
  bool isConvertedEndcap;

  bool isBarrelElectrons95;
  bool isEndcapElectrons95;
  bool isIsolatedBarrel95;
  bool isIDBarrel95;
  bool isConvertedBarrel95;
  bool isIsolatedEndcap95;
  bool isIDEndcap95;
  bool isConvertedEndcap95;

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

      isBarrelElectrons = false;
      isEndcapElectrons = false;
      isIsolatedBarrel = false;
      isIDBarrel = false;
      isConvertedBarrel = false;
      isIsolatedEndcap = false;
      isIDEndcap = false;
      isConvertedEndcap = false;

      isBarrelElectrons95 = false;
      isEndcapElectrons95 = false;
      isIsolatedBarrel95 = false;
      isIDBarrel95 = false;
      isConvertedBarrel95 = false;
      isIsolatedEndcap95 = false;
      isIDEndcap95 = false;
      isConvertedEndcap95 = false;


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

  /***** Barrel WP95 Cuts *****/

      if (fabs (recoElectron->eta ()) <= 1.4442)
	{

	  /* Isolation */
	  if (IsoTrk < 0.15 && IsoEcal < 2.0 && IsoHcal < 0.12)
	    {
	      isIsolatedBarrel95 = true;
	    }

	  /* Identification */
	  if (fabs (DeltaEtaTkClu) < 0.007 && fabs (DeltaPhiTkClu) < 0.8
	      && sigmaIeIe < 0.01 && HE < 0.15)
	    {
	      isIDBarrel95 = true;
	    }

	  /* Conversion Rejection */
	  if (NumberOfExpectedInnerHits <= 1.0)
	    {
	      isConvertedBarrel95 = true;
	    }
	}

  /***** Endcap WP95 Cuts *****/

      if (fabs (recoElectron->eta ()) >= 1.5660
	  && fabs (recoElectron->eta ()) <= 2.5000)
	{

	  /* Isolation */
	  if (IsoTrk < 0.08 && IsoEcal < 0.06 && IsoHcal < 0.05)
	    {
	      isIsolatedEndcap95 = true;
	    }

	  /* Identification */
	  if (fabs (DeltaEtaTkClu) < 0.01 && fabs (DeltaPhiTkClu) < 0.7
	      && sigmaIeIe < 0.031 && HE < 0.07)
	    {
	      isIDEndcap95 = true;
	    }

	  /* Conversion Rejection */
	  if (NumberOfExpectedInnerHits <= 1.0)
	    {
	      isConvertedEndcap95 = true;
	    }
	}


      // Search for two WP80 barrel electrons

      if (isIsolatedBarrel && isConvertedBarrel && isIDBarrel)
	{

	  if (recoElectron->et () > b_electron_et)
	    {
	      b_electron2_et = b_electron_et;	// 2nd highest gets values from current highest
	      b_electron2_eta = b_electron_eta;
	      b_electron2_phi = b_electron_phi;
	      b_electron_et = recoElectron->et ();	// 1st highest gets values from new highest
	      b_electron_eta = recoElectron->eta ();
	      b_electron_phi = recoElectron->phi ();
	      b_e1 =
		TLorentzVector (recoElectron->momentum ().x (),
				recoElectron->momentum ().y (),
				recoElectron->momentum ().z (),
				recoElectron->p ());
	    }
	  else if (recoElectron->et () > b_electron2_et)
	    {
	      b_electron2_et = recoElectron->et ();
	      b_electron2_eta = recoElectron->eta ();
	      b_electron2_phi = recoElectron->phi ();
	      b_e2 =
		TLorentzVector (recoElectron->momentum ().x (),
				recoElectron->momentum ().y (),
				recoElectron->momentum ().z (),
				recoElectron->p ());
	    }
	}

      // Search for two WP80 endcap electrons

      if (isIsolatedEndcap && isConvertedEndcap && isIDEndcap)
	{
	  if (recoElectron->et () > e_electron_et)
	    {
	      e_electron2_et = e_electron_et;	// 2nd highest gets values from current highest
	      e_electron2_eta = e_electron_eta;
	      e_electron2_phi = e_electron_phi;
	      e_electron_et = recoElectron->et ();	// 1st highest gets values from new highest
	      e_electron_eta = recoElectron->eta ();
	      e_electron_phi = recoElectron->phi ();
	      e_e1 =
		TLorentzVector (recoElectron->momentum ().x (),
				recoElectron->momentum ().y (),
				recoElectron->momentum ().z (),
				recoElectron->p ());
	    }
	  else if (recoElectron->et () > e_electron2_et)
	    {
	      e_electron2_et = recoElectron->et ();
	      e_electron2_eta = recoElectron->eta ();
	      e_electron2_phi = recoElectron->phi ();
	      e_e2 =
		TLorentzVector (recoElectron->momentum ().x (),
				recoElectron->momentum ().y (),
				recoElectron->momentum ().z (),
				recoElectron->p ());
	    }
	}

      // search for barrel/endcap electrons WP80: find the most energetic barrel one with a less energetic endcap 

      if (isIsolatedBarrel && isConvertedBarrel && isIDBarrel)
	{
	  if (recoElectron->et () > eb_electron_et)
	    {
	      eb_electron2_et = eb_electron_et;	// 2nd highest gets values from current highest
	      eb_electron2_eta = eb_electron_eta;
	      eb_electron2_phi = eb_electron_phi;
	      eb_electron_et = recoElectron->et ();	// 1st highest gets values from new highest
	      eb_electron_eta = recoElectron->eta ();
	      eb_electron_phi = recoElectron->phi ();
	      eb_e1 =
		TLorentzVector (recoElectron->momentum ().x (),
				recoElectron->momentum ().y (),
				recoElectron->momentum ().z (),
				recoElectron->p ());
	    }
	}
      else if (recoElectron->et () > eb_electron2_et)
	{
	  if (isIsolatedEndcap && isConvertedEndcap && isIDEndcap)
	    {
	      eb_electron2_et = recoElectron->et ();
	      eb_electron2_eta = recoElectron->eta ();
	      eb_electron2_phi = recoElectron->phi ();
	      eb_e2 =
		TLorentzVector (recoElectron->momentum ().x (),
				recoElectron->momentum ().y (),
				recoElectron->momentum ().z (),
				recoElectron->p ());
	    }
	}

      // search for endcap/barrel WP80: find the most energetic endcap one with a less energetic barrel 

      if (isIsolatedEndcap && isConvertedEndcap && isIDEndcap)
	{
	  if (recoElectron->et () > eb_electron_et)
	    {
	      be_electron2_et = be_electron_et;	// 2nd highest gets values from current highest
	      be_electron2_eta = be_electron_eta;
	      be_electron2_phi = be_electron_phi;
	      be_electron_et = recoElectron->et ();	// 1st highest gets values from new highest
	      be_electron_eta = recoElectron->eta ();
	      be_electron_phi = recoElectron->phi ();
	      be_e1 =
		TLorentzVector (recoElectron->momentum ().x (),
				recoElectron->momentum ().y (),
				recoElectron->momentum ().z (),
				recoElectron->p ());
	    }
	}
      else if (recoElectron->et () > be_electron2_et)
	{
	  if (isIsolatedBarrel && isConvertedBarrel && isIDBarrel)
	    {
	      be_electron2_et = recoElectron->et ();
	      be_electron2_eta = recoElectron->eta ();
	      be_electron2_phi = recoElectron->phi ();
	      be_e2 =
		TLorentzVector (recoElectron->momentum ().x (),
				recoElectron->momentum ().y (),
				recoElectron->momentum ().z (),
				recoElectron->p ());
	    }
	}



/*

    // Search for two WP95 barrel electrons

    if (isIsolatedBarrel95 && isConvertedBarrel95 && isIDBarrel95) {

      if (recoElectron->et () > b_95_electron_et) {
	b_95_electron2_et = b_95_electron_et;	// 2nd highest gets values from current highest
	b_95_electron2_eta = b_95_electron_eta;
	b_95_electron2_phi = b_95_electron_phi;
	b_95_electron_et = recoElectron->et ();	// 1st highest gets values from new highest
	b_95_electron_eta = recoElectron->eta ();
	b_95_electron_phi = recoElectron->phi ();
	b_95_e1 =
	  TLorentzVector (recoElectron->momentum ().x (),
			  recoElectron->momentum ().y (),
			  recoElectron->momentum ().z (), recoElectron->p ());
      }
      else if (recoElectron->et () > b_95_electron2_et) {
	b_95_electron2_et = recoElectron->et ();
	b_95_electron2_eta = recoElectron->eta ();
	b_95_electron2_phi = recoElectron->phi ();
	b_95_e2 =
	  TLorentzVector (recoElectron->momentum ().x (),
			  recoElectron->momentum ().y (),
			  recoElectron->momentum ().z (), recoElectron->p ());
      }
    }

    // Search for two WP95 endcap electrons

    if (isIsolatedEndcap95 && isConvertedEndcap95 && isIDEndcap95) {
      if (recoElectron->et () > e_electron_et) {
	e_95_electron2_et = e_95_electron_et;	// 2nd highest gets values from current highest
	e_95_electron2_eta = e_95_electron_eta;
	e_95_electron2_phi = e_95_electron_phi;
	e_95_electron_et = recoElectron->et ();	// 1st highest gets values from new highest
	e_95_electron_eta = recoElectron->eta ();
	e_95_electron_phi = recoElectron->phi ();
	e_95_e1 =
	  TLorentzVector (recoElectron->momentum ().x (),
			  recoElectron->momentum ().y (),
			  recoElectron->momentum ().z (), recoElectron->p ());
      }
      else if (recoElectron->et () > e_95_electron2_et) {
	e_95_electron2_et = recoElectron->et ();
	e_95_electron2_eta = recoElectron->eta ();
	e_95_electron2_phi = recoElectron->phi ();
	e_95_e2 =
	  TLorentzVector (recoElectron->momentum ().x (),
			  recoElectron->momentum ().y (),
			  recoElectron->momentum ().z (), recoElectron->p ());
      }
    }
    
    
    // Search for barrel/endcap WP95 electrons; highest barrel and less energetic endcap

    if (isIsolatedBarrel95 && isConvertedBarrel95 && isIDBarrel95) {
      if (recoElectron->et () > eb_electron_et) {
	eb_95_electron2_et = eb_95_electron_et;	// 2nd highest gets values from current highest
	eb_95_electron2_eta = eb_95_electron_eta;
	eb_95_electron2_phi = eb_95_electron_phi;
	eb_95_electron_et = recoElectron->et ();	// 1st highest gets values from new highest
	eb_95_electron_eta = recoElectron->eta ();
	eb_95_electron_phi = recoElectron->phi ();
	eb_95_e1 =
	  TLorentzVector (recoElectron->momentum ().x (),
			  recoElectron->momentum ().y (),
			  recoElectron->momentum ().z (), recoElectron->p ());
      }
    }
    else if (recoElectron->et () > eb_electron2_et) {
      if (isIsolatedEndcap95 && isConvertedEndcap95 && isIDEndcap95) {
	eb_95_electron2_et = recoElectron->et ();
	eb_95_electron2_eta = recoElectron->eta ();
	eb_95_electron2_phi = recoElectron->phi ();
	eb_95_e2 =
	  TLorentzVector (recoElectron->momentum ().x (),
			  recoElectron->momentum ().y (),
			  recoElectron->momentum ().z (), recoElectron->p ());
      }
    }
    
    // Search for endcap/barrel WP95 electrons; highest endcap one and less energetic barrel

    if (isIsolatedBarrel95 && isConvertedBarrel95 && isIDBarrel95) {
      if (recoElectron->et () > eb_electron_et) {
	be_95_electron2_et = be_95_electron_et;	// 2nd highest gets values from current highest
	be_95_electron2_eta = be_95_electron_eta;
	be_95_electron2_phi = be_95_electron_phi;
	be_95_electron_et = recoElectron->et ();	// 1st highest gets values from new highest
	be_95_electron_eta = recoElectron->eta ();
	be_95_electron_phi = recoElectron->phi ();
	be_95_e1 =
	  TLorentzVector (recoElectron->momentum ().x (),
			  recoElectron->momentum ().y (),
			  recoElectron->momentum ().z (), recoElectron->p ());
      }
    }
    else if (recoElectron->et () > be_electron2_et) {
      if (isIsolatedEndcap95 && isConvertedEndcap95 && isIDEndcap95) {
	be_95_electron2_et = recoElectron->et ();
	be_95_electron2_eta = recoElectron->eta ();
	be_95_electron2_phi = recoElectron->phi ();
	be_95_e2 =
	  TLorentzVector (recoElectron->momentum ().x (),
			  recoElectron->momentum ().y (),
			  recoElectron->momentum ().z (), recoElectron->p ());
      }
    }*/
    }

  // Calculate the Z invariant masses

  if (e_electron2_et > 0.0)
    {
      TLorentzVector e_pair = e_e1 + e_e2;
      e_ee_invMass = e_pair.M ();
    }
  if (b_electron2_et > 0.0)
    {
      TLorentzVector b_pair = b_e1 + b_e2;
      b_ee_invMass = b_pair.M ();
    }

  if (eb_electron_et > be_electron_et)
    {
      TLorentzVector eb_pair = eb_e1 + eb_e2;
      eb_ee_invMass = eb_pair.M ();
    }
  else
    {
      TLorentzVector eb_pair = be_e1 + be_e2;
      eb_ee_invMass = eb_pair.M ();
    }

/*
  if (e_95_electron2_et > 0.0) {
    TLorentzVector e_95_pair = e_95_e1 + e_95_e2;
    e_95_ee_invMass = e_95_pair.M ();
  }
  if (b_95_electron2_et > 0.0) {
    TLorentzVector b_95_pair = b_95_e1 + b_95_e2;
    b_95_ee_invMass = b_95_pair.M ();
  }

  if (eb_95_electron_et > be_95_electron_et) {
    TLorentzVector eb_95_pair = eb_95_e1 + eb_95_e2;
    eb_95_ee_invMass = eb_95_pair.M ();
  } else {
	 TLorentzVector eb_95_pair = be_95_e1 + be_95_e2;
	 eb_95_ee_invMass = eb_95_pair.M ();
  }
  */


  if (h_ee_invMass_EE != 0)
    {
      if (e_ee_invMass > 0.0)
	{
	  h_ee_invMass_EE->Fill (e_ee_invMass);
	}
    }

  if (h_ee_invMass_BB != 0)
    {
      if (b_ee_invMass > 0.0)
	{
	  h_ee_invMass_BB->Fill (b_ee_invMass);
	}
    }

  if (h_ee_invMass_EB != 0)
    {
      if (eb_ee_invMass > 0.0)
	{
	  h_ee_invMass_EB->Fill (eb_ee_invMass);
	}
    }
/*
  if (h_95_ee_invMass_EE != 0) {
    if (e_95_ee_invMass > 0.0) {
      h_95_ee_invMass_EE->Fill (e_95_ee_invMass);
    }
  }

  if (h_95_ee_invMass_BB != 0) {
    if (b_95_ee_invMass > 0.0) {
      h_95_ee_invMass_BB->Fill (b_95_ee_invMass);
    }
  }

  if (h_95_ee_invMass_EB != 0) {
    if (eb_95_ee_invMass > 0.0) {
      h_95_ee_invMass_EB->Fill (eb_95_ee_invMass);
    }
  }
*/

}				// end of reco electron loop

#ifdef THIS_IS_AN_EVENT_EXAMPLE
Handle < ExampleData > pIn;
iEvent.getByLabel ("example", pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
ESHandle < SetupData > pSetup;
iSetup.get < SetupRecord > ().get (pSetup);
#endif



// ------------ method called once each job just before starting event loop  ------------
void
EcalZmassTask::beginJob ()
{

  DQMStore *theDbe;
  std::string logTraceName;

  logTraceName = "EwkAnalyzer";

  h_ee_invMass_EB = 0;
  h_ee_invMass_EE = 0;
  h_ee_invMass_BB = 0;

  /*
     h_95_ee_invMass_EB = 0;
     h_95_ee_invMass_EE = 0;
     h_95_ee_invMass_BB = 0;
   */


  LogTrace (logTraceName) << "Parameters initialization";
  theDbe = Service < DQMStore > ().operator-> ();

  if (theDbe != 0)
    {
      theDbe->setCurrentFolder (prefixME_ + "/Zmass");	// Use folder with name of PAG


      h_ee_invMass_EB =
	theDbe->book1D ("Z peak - WP80 EB-EE",
			"Z peak - WP80 EB-EE;InvMass (GeV)", 60, 60.0, 120.0);
      h_ee_invMass_EE =
	theDbe->book1D ("Z peak - WP80 EE-EE",
			"Z peak - WP80 EE-EE;InvMass (Gev)", 60, 60.0, 120.0);
      h_ee_invMass_BB =
	theDbe->book1D ("Z peak - WP80 EB-EB",
			"Z peak - WP80 EB-EB;InvMass (Gev)", 60, 60.0, 120.0);

/*    h_95_ee_invMass_EB =
      theDbe->book1D ("Z peak - WP95 EB-EE",
		      "Z peak - WP95 EB-EE;InvMass (Gev)", 60, 60.0, 120.0);
    h_95_ee_invMass_EE =
      theDbe->book1D ("Z peak - WP95 EE-EE",
		      "Z peak - WP95 EE-EE;InvMass (Gev)", 60, 60.0, 120.0);
    h_95_ee_invMass_BB =
      theDbe->book1D ("Z peak - WP95 EB-EB",
		      "Z peak - WP95 EB-EB;InvMass (Gev)", 60, 60.0, 120.0);
*/

      //h_e1_et             = theDbe->book1D("h_e1_et",  "E_{T} of Leading Electron;E_{T} (GeV)"        , 20,  0.0 , 100.0); 
      //h_e1_et             = theDbe->book1D("h_e1_et",  "E_{T} of Leading Electron;E_{T} (GeV)"        , 20,  0.0 , 100.0);
      //h_e2_et             = theDbe->book1D("h_e2_et",  "E_{T} of Second Electron;E_{T} (GeV)"         , 20,  0.0 , 100.0);
      //h_e1_eta            = theDbe->book1D("h_e1_eta", "#eta of Leading Electron;#eta"                , 20, -4.0 , 4.0);
      //h_e2_eta            = theDbe->book1D("h_e2_eta", "#eta of Second Electron;#eta"                 , 20, -4.0 , 4.0);
      //h_e1_phi            = theDbe->book1D("h_e1_phi", "#phi of Leading Electron;#phi"                , 22, (-1.-1./10.)*pi, (1.+1./10.)*pi );
      //h_e2_phi            = theDbe->book1D("h_e2_phi", "#phi of Second Electron;#phi"                 , 22, (-1.-1./10.)*pi, (1.+1./10.)*pi );
    }
}

// ------------ method called once each job just after ending the event loop  ------------
void
EcalZmassTask::endJob ()
{
}

// ------------ method called when starting to processes a run  ------------
void
EcalZmassTask::beginRun (edm::Run const &, edm::EventSetup const &)
{

}

// ------------ method called when ending the processing of a run  ------------
void
EcalZmassTask::endRun (edm::Run const &, edm::EventSetup const &)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
EcalZmassTask::beginLuminosityBlock (edm::LuminosityBlock const &,
				     edm::EventSetup const &)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
EcalZmassTask::endLuminosityBlock (edm::LuminosityBlock const &,
				   edm::EventSetup const &)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EcalZmassTask::fillDescriptions (edm::
				 ConfigurationDescriptions & descriptions)
{
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown ();
  descriptions.addDefault (desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE (EcalZmassTask);
