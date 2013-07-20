// -*- C++ -*-
//
// Package:    ZfitterAnalyzer
// Class:      ZfitterAnalyzer
// 
/**\class ZfitterAnalyzer ZfitterAnalyzer.cc Zfitter/ZfitterAnalyzer/src/ZfitterAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Vieri Candelise
//         Created:  Mon Jun 13 09:49:08 CEST 2011
// $Id: EcalZmassClient.cc,v 1.3 2013/05/10 22:01:23 yiiyama Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/Physics/src/EwkDQM.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TMath.h"
#include <string>
#include <cmath>
#include "TH1.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

class DQMStore;
class MonitorElement;

//
// class declaration
//

class EcalZmassClient:public
  edm::EDAnalyzer
{
public:
  explicit
  EcalZmassClient (const edm::ParameterSet &);
   ~
  EcalZmassClient ();

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

  // ----------member data ---------------------------

  std::string prefixME_;

  MonitorElement *
    h_ee_invMass_EB;
  MonitorElement *
    h_ee_invMass_EE;
  MonitorElement *
    h_ee_invMass_BB;
  MonitorElement *
    h_95_ee_invMass_EB;
  MonitorElement *
    h_95_ee_invMass_EE;
  MonitorElement *
    h_95_ee_invMass_BB;
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
    h_fitres1;
  MonitorElement *
    h_fitres1bis;
  MonitorElement *
    h_fitres1Chi2;
  MonitorElement *
    h_fitres2;
  MonitorElement *
    h_fitres2bis;
  MonitorElement *
    h_fitres2Chi2;
  MonitorElement *
    h_fitres3;
  MonitorElement *
    h_fitres3bis;
  MonitorElement *
    h_fitres3Chi2;


};

EcalZmassClient::EcalZmassClient (const edm::ParameterSet & iConfig)
{
  prefixME_ = iConfig.getUntrackedParameter < std::string > ("prefixME", "");

}


EcalZmassClient::~EcalZmassClient ()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
EcalZmassClient::analyze (const edm::Event & iEvent,
			  const edm::EventSetup & iSetup)
{

}


// ------------ method called once each job just before starting event loop  ------------
void
EcalZmassClient::beginJob ()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
EcalZmassClient::endJob ()
{
}

// ------------ method called when starting to processes a run  ------------
void
EcalZmassClient::beginRun (edm::Run const &, edm::EventSetup const &)
{
  DQMStore *theDbe = edm::Service < DQMStore > ().operator-> ();

  theDbe->setCurrentFolder (prefixME_ + "/Zmass");
  h_fitres1 =
    theDbe->book1D ("Gaussian mean WP80 EB-EB",
		    "Gaussian mean WP80 EB-EB", 1, 0, 1);
  h_fitres1bis =
    theDbe->book1D ("Gaussian sigma WP80 EB-EB",
		    "Gaussian sigma WP80 EB-EB", 1, 0, 1);
  h_fitres1Chi2 =
    theDbe->book1D ("Gaussian Chi2 result over NDF  WP80 EB-EB",
		    "Gaussian Chi2 result over NDF  WP80 EB-EB", 1, 0, 1);

  h_fitres3 =
    theDbe->book1D ("Gaussian mean WP80 EB-EE",
		    "Gaussian mean result WP80 EB-EE", 1, 0, 1);
  h_fitres3bis =
    theDbe->book1D ("Gaussian sigma WP80 EB-EE",
		    "Gaussian sigma WP80 EB-EE", 1, 0, 1);
  h_fitres3Chi2 =
    theDbe->book1D ("Gaussian Chi2 result over NDF WP80 EB-EE",
		    "Gaussian Chi2 result over NDF WP80 EB-EE", 1, 0, 1);

  h_fitres2 =
    theDbe->book1D ("Gaussian mean WP80 EE-EE",
		    "Gaussian mean WP80 EE-EE", 1, 0, 1);
  h_fitres2bis =
    theDbe->book1D ("Gaussian sigma WP80 EE-EE",
		    "Gaussian sigma WP80 EE-EE", 1, 0, 1);
  h_fitres2Chi2 =
    theDbe->book1D ("Gaussian Chi2 result over NDF WP80 EE-EE",
		    "Gaussian Chi2 result over NDF WP80 EE-EE", 1, 0, 1);
}


//Breit-Wigner function
Double_t
mybw (Double_t * x, Double_t * par)
{
  Double_t arg1 = 14.0 / 22.0;	// 2 over pi
  Double_t arg2 = par[1] * par[1] * par[2] * par[2];	//Gamma=par[2]  M=par[1]
  Double_t arg3 =
    ((x[0] * x[0]) - (par[1] * par[1])) * ((x[0] * x[0]) - (par[1] * par[1]));
  Double_t arg4 =
    x[0] * x[0] * x[0] * x[0] * ((par[2] * par[2]) / (par[1] * par[1]));
  return par[0] * arg1 * arg2 / (arg3 + arg4);
}

//Gaussian
Double_t
mygauss (Double_t * x, Double_t * par)
{
  Double_t arg = 0;
  if (par[2] < 0)
    par[2] = -par[2];		// par[2]: sigma
  if (par[2] != 0)
    arg = (x[0] - par[1]) / par[2];	// par[1]: mean
  return par[0] * TMath::Exp (-0.5 * arg * arg) / (TMath::Sqrt (2 * TMath::Pi ()) * par[2]);	// par[0] is constant

}



// ------------ method called when ending the processing of a run  ------------
void
EcalZmassClient::endRun (edm::Run const &, edm::EventSetup const &)
{
  DQMStore *theDbe = edm::Service < DQMStore > ().operator-> ();

  LogTrace ("EwkAnalyzer") << "Parameters initialization";

  MonitorElement *me1 = theDbe->get (prefixME_ + "/Zmass/Z peak - WP80 EB-EB");
  MonitorElement *me2 = theDbe->get (prefixME_ + "/Zmass/Z peak - WP80 EE-EE");
  MonitorElement *me3 = theDbe->get (prefixME_ + "/Zmass/Z peak - WP80 EB-EE");


  if (me1 != 0)
    {
      TH1F *B = me1->getTH1F ();
      TH1F *R1 = h_fitres1->getTH1F ();
      int division = B->GetNbinsX ();
      float massMIN = B->GetBinLowEdge (1);
      float massMAX = B->GetBinLowEdge (division + 1);
      //float BIN_SIZE = B->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      R1->GetStats (stats);
      float N = 0;
      float mean = 0;
      float sigma = 0;
      N = B->GetEntries ();

      try
      {
	if (N != 0)
	  {

	    B->Fit ("mygauss", "QR");
	    mean = fabs (func->GetParameter (1));
	    sigma = fabs (func->GetParError (1));
	  }

	if (N == 0 || mean < 50 || mean > 100 || sigma <= 0 || sigma > 20)
	  {
	    N = 1;
	    mean = 0;
	    sigma = 0;
	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	mean = 40;
	sigma = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = mean * N;
      stats[3] = sigma * sigma * N + mean * mean * N;

      R1->SetEntries (N);
      R1->PutStats (stats);
    }
/*******************************************************/
  if (me1 != 0)
    {
      TH1F *Bbis = me1->getTH1F ();
      TH1F *R1bis = h_fitres1bis->getTH1F ();
      int division = Bbis->GetNbinsX ();
      float massMIN = Bbis->GetBinLowEdge (1);
      float massMAX = Bbis->GetBinLowEdge (division + 1);
      //float BIN_SIZE = B->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      R1bis->GetStats (stats);
      float N = 0;
      float rms = 0;
      float rmsErr = 0;
      N = Bbis->GetEntries ();

      try
      {
	if (N != 0)
	  {

	    Bbis->Fit ("mygauss", "QR");
	    rms = fabs (func->GetParameter (2));
	    rmsErr = fabs (func->GetParError (2));
	  }

	if (N == 0 || rms < 0 || rms > 50 || rmsErr <= 0 || rmsErr > 50)
	  {
	    N = 1;
	    rms = 0;
	    rmsErr = 0;
	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	rms = 40;
	rmsErr = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = rms * N;
      stats[3] = rmsErr * rmsErr * N + rms * rms * N;

      R1bis->SetEntries (N);
      R1bis->PutStats (stats);
    }
/****************************************/

  if (me2 != 0)
    {
      TH1F *E = me2->getTH1F ();
      TH1F *R2 = h_fitres2->getTH1F ();
      int division = E->GetNbinsX ();
      float massMIN = E->GetBinLowEdge (1);
      float massMAX = E->GetBinLowEdge (division + 1);
      //float BIN_SIZE = E->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      R2->GetStats (stats);
      float N = 0;
      float mean = 0;
      float sigma = 0;
      N = E->GetEntries ();

      try
      {
	if (N != 0)
	  {
	    E->Fit ("mygauss", "QR");
	    mean = fabs (func->GetParameter (1));
	    sigma = fabs (func->GetParError (1));
	  }

	if (N == 0 || mean < 50 || mean > 100 || sigma <= 0 || sigma > 20)
	  {
	    N = 1;
	    mean = 0;
	    sigma = 0;

	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	mean = 40;
	sigma = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = mean * N;
      stats[3] = sigma * sigma * N + mean * mean * N;

      R2->SetEntries (N);
      R2->PutStats (stats);
    }
/**************************************************************************/

  if (me2 != 0)
    {
      TH1F *Ebis = me2->getTH1F ();
      TH1F *R2bis = h_fitres2bis->getTH1F ();
      int division = Ebis->GetNbinsX ();
      float massMIN = Ebis->GetBinLowEdge (1);
      float massMAX = Ebis->GetBinLowEdge (division + 1);
      //float BIN_SIZE = B->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      R2bis->GetStats (stats);
      float N = 0;
      float rms = 0;
      float rmsErr = 0;
      N = Ebis->GetEntries ();

      try
      {
	if (N != 0)
	  {

	    Ebis->Fit ("mygauss", "QR");
	    rms = fabs (func->GetParameter (2));
	    rmsErr = fabs (func->GetParError (2));
	  }

	if (N == 0 || rms < 0 || rms > 50 || rmsErr <= 0 || rmsErr > 50)
	  {
	    N = 1;
	    rms = 0;
	    rmsErr = 0;
	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	rms = 40;
	rmsErr = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = rms * N;
      stats[3] = rmsErr * rmsErr * N + rms * rms * N;

      R2bis->SetEntries (N);
      R2bis->PutStats (stats);
    }
/*********************************************************************************************/

  if (me3 != 0)
    {
      TH1F *R3 = h_fitres3->getTH1F ();
      TH1F *M = me3->getTH1F ();
      int division = M->GetNbinsX ();
      float massMIN = M->GetBinLowEdge (1);
      float massMAX = M->GetBinLowEdge (division + 1);
      //float BIN_SIZE = M->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      R3->GetStats (stats);
      float N = 0;
      float mean = 0;
      float sigma = 0;
      N = M->GetEntries ();


      try
      {
	if (N != 0)
	  {

	    M->Fit ("mygauss", "QR");
	    mean = fabs (func->GetParameter (1));
	    sigma = fabs (func->GetParError (1));
	  }
	if (N == 0 || mean < 50 || mean > 100 || sigma <= 0 || sigma > 20)
	  {
	    N = 1;
	    mean = 0;
	    sigma = 0;
	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	mean = 40;
	sigma = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = mean * N;
      stats[3] = sigma * sigma * N + mean * mean * N;

      R3->SetEntries (N);
      R3->PutStats (stats);
    }
  /********************************************************************************/

  if (me3 != 0)
    {
      TH1F *Mbis = me3->getTH1F ();
      TH1F *R3bis = h_fitres3bis->getTH1F ();
      int division = Mbis->GetNbinsX ();
      float massMIN = Mbis->GetBinLowEdge (1);
      float massMAX = Mbis->GetBinLowEdge (division + 1);
      //float BIN_SIZE = B->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      R3bis->GetStats (stats);
      float N = 0;
      float rms = 0;
      float rmsErr = 0;
      N = Mbis->GetEntries ();

      try
      {
	if (N != 0)
	  {

	    Mbis->Fit ("mygauss", "QR");
	    rms = fabs (func->GetParameter (2));
	    rmsErr = fabs (func->GetParError (2));
	  }

	if (N == 0 || rms < 0 || rms > 50 || rmsErr <= 0 || rmsErr > 50)
	  {
	    N = 1;
	    rms = 0;
	    rmsErr = 0;
	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	rms = 40;
	rmsErr = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = rms * N;
      stats[3] = rmsErr * rmsErr * N + rms * rms * N;

      R3bis->SetEntries (N);
      R3bis->PutStats (stats);
    }

  /*Chi2 */

  if (me1 != 0)
    {
      TH1F *C1 = me1->getTH1F ();
      TH1F *S1 = h_fitres1Chi2->getTH1F ();
      int division = C1->GetNbinsX ();
      float massMIN = C1->GetBinLowEdge (1);
      float massMAX = C1->GetBinLowEdge (division + 1);
      //float BIN_SIZE = B->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      S1->GetStats (stats);
      float N = 0;
      float Chi2 = 0;
      float NDF = 0;
      N = C1->GetEntries ();

      try
      {
	if (N != 0)
	  {

	    C1->Fit ("mygauss", "QR");
	    if ((func->GetNDF () != 0))
	      {
		Chi2 = fabs (func->GetChisquare ()) / fabs (func->GetNDF ());
		NDF = 0.1;
	      }
	  }

	if (N == 0 || Chi2 < 0 || NDF < 0)
	  {
	    N = 1;
	    Chi2 = 0;
	    NDF = 0;
	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	Chi2 = 40;
	NDF = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = Chi2 * N;
      stats[3] = NDF * NDF * N + Chi2 * Chi2 * N;

      S1->SetEntries (N);
      S1->PutStats (stats);
    }
  /**********************************************/

  if (me2 != 0)
    {
      TH1F *C2 = me2->getTH1F ();
      TH1F *S2 = h_fitres2Chi2->getTH1F ();
      int division = C2->GetNbinsX ();
      float massMIN = C2->GetBinLowEdge (1);
      float massMAX = C2->GetBinLowEdge (division + 1);
      //float BIN_SIZE = B->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      S2->GetStats (stats);
      float N = 0;
      float Chi2 = 0;
      float NDF = 0;
      N = C2->GetEntries ();

      try
      {
	if (N != 0)
	  {
	    C2->Fit ("mygauss", "QR");
	    if (func->GetNDF () != 0)
	      {
		Chi2 = fabs (func->GetChisquare ()) / fabs (func->GetNDF ());
		NDF = 0.1;
	      }
	  }

	if (N == 0 || Chi2 < 0 || NDF < 0)
	  {
	    N = 1;
	    Chi2 = 0;
	    NDF = 0;
	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	Chi2 = 40;
	NDF = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = Chi2 * N;
      stats[3] = NDF * NDF * N + Chi2 * Chi2 * N;

      S2->SetEntries (N);
      S2->PutStats (stats);
    }
  /**************************************************************************/
  if (me3 != 0)
    {
      TH1F *C3 = me3->getTH1F ();
      TH1F *S3 = h_fitres3Chi2->getTH1F ();
      int division = C3->GetNbinsX ();
      float massMIN = C3->GetBinLowEdge (1);
      float massMAX = C3->GetBinLowEdge (division + 1);
      //float BIN_SIZE = B->GetBinWidth(1);

      TF1 *func = new TF1 ("mygauss", mygauss, massMIN, massMAX, 3);
      func->SetParameter (0, 1.0);
      func->SetParName (0, "const");
      func->SetParameter (1, 95.0);
      func->SetParName (1, "mean");
      func->SetParameter (2, 5.0);
      func->SetParName (2, "sigma");

      double stats[4];
      S3->GetStats (stats);
      float N = 0;
      float Chi2 = 0;
      float NDF = 0;
      N = C3->GetEntries ();

      try
      {
	if (N != 0)
	  {
	    C3->Fit ("mygauss", "QR");
	    if ((func->GetNDF () != 0))
	      {
		Chi2 = fabs (func->GetChisquare ()) / fabs (func->GetNDF ());
		NDF = 0.1;
	      }
	  }


	if (N == 0 || Chi2 < 0 || NDF < 0)
	  {
	    N = 1;
	    Chi2 = 0;
	    NDF = 0;
	  }

      }
      catch (cms::Exception& e)
      {
        edm::LogError ("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
	N = 1;
	Chi2 = 40;
	NDF = 0;
      }

      stats[0] = N;
      stats[1] = N;
      stats[2] = Chi2 * N;
      stats[3] = NDF * NDF * N + Chi2 * Chi2 * N;

      S3->SetEntries (N);
      S3->PutStats (stats);
    }
}

	// ------------ method called when starting to processes a luminosity block  ------------
void
EcalZmassClient::beginLuminosityBlock (edm::LuminosityBlock const &,
				       edm::EventSetup const &)
{
}

	// ------------ method called when ending the processing of a luminosity block  ------------
void
EcalZmassClient::endLuminosityBlock (edm::LuminosityBlock const &,
				     edm::EventSetup const &)
{
}

	// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EcalZmassClient::fillDescriptions (edm::
				   ConfigurationDescriptions & descriptions)
{
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown ();
  descriptions.addDefault (desc);
}

	//define this as a plug-in
DEFINE_FWK_MODULE (EcalZmassClient);
