// Template to demonstrate accessing to correction service from user analyzer
// plot correction as a function of et:eta
// author: F.Ratnikov UMd Mar. 16, 2007: Initial version
//         R. Harris FNAL Mar. 21, 2007: Added plots of response & correction in the Barrel, Endcap, Forward
//
// Note that this provides an example of accessing jet corrections as a function of
// arbitrary Jet ET and eta, however it is not an example of accessing the jet corrections
// if one actually has a jet.  If you have a jet you would pass the jet to the corrector, and
// get the correction for that jet.  Here we calculate a fake Lorentz Vector for a given value of
// jet Et and eta, settting phi to zero and not caring what we get for the mass, and that is not necessary, 
// or advisable, if one actually has a jet.
// 
#include "JetMETCorrections/Modules/interface/PlotJetCorrections.h"

#include <math.h>
#include <TFile.h>
#include <TH2D.h>
#include <TAxis.h>
#include <TRandom.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/JetReco/interface/Jet.h"

PlotJetCorrections::PlotJetCorrections (const edm::ParameterSet& fConfig) 
  :  mCorrectorNames (fConfig.getParameter <std::vector <std::string> > ("correctors")),
     mFileName (fConfig.getParameter<std::string> ("output")),
     mAllDone (false)
{}

void PlotJetCorrections::analyze (const edm::Event& fEvent, const edm::EventSetup& fSetup) {
  if (!mAllDone) {
    // book histogram
    TFile file (mFileName.c_str(), "RECREATE");
    const double eta_max = 4.8;
    const int neta_bins = 100;
    const double et_min = 3.;  //modified from Fedor's initial cut at 1
    const double et_max = 7e3; //modified from Fedor's initial cut at 1e6
    const double et_max_barrel = 7e3;  //Kinematic limit in Barrel
    const double et_max_endcap = 1e3;  //Roughly Kinematic limit in endcap
    const double et_max_forward = 400; //Roughly Kinematic limit in forward
    const int net_bins = 100;

    double etbins [net_bins+1];
    double etbinsBarrel [net_bins+1];
    double etbinsEndcap [net_bins+1];
    double etbinsForward [net_bins+1];
    for (int i = 0; i < net_bins+1; ++i) {
      etbins[i] =  et_min * pow (et_max/et_min, double(i)/double(net_bins));
      etbinsBarrel[i] =  et_min * pow (et_max_barrel/et_min, double(i)/double(net_bins));
      etbinsEndcap[i] =  et_min * pow (et_max_endcap/et_min, double(i)/double(net_bins));
      etbinsForward[i] =  et_min * pow (et_max_forward/et_min, double(i)/double(net_bins));
    }

    //Fedors plot of corrections, randomly sampled in Et and eta within bins
    TH2D corrections ("CorrectionScale", (mCorrectorNames[0] + std::string (": Jet Correction Function")).c_str(), 
		      net_bins, etbins, neta_bins, 0, eta_max);
    TH2D variations ("CorrVariations", (mCorrectorNames[0] + std::string (": Jet Correction Variations")).c_str(),
		     net_bins, etbins, neta_bins, 0, eta_max);
    corrections.GetXaxis()->SetTitle ("Et");
    corrections.GetYaxis()->SetTitle ("Eta");
    variations.GetXaxis()->SetTitle ("Et");
    variations.GetYaxis()->SetTitle ("Eta");

    //Roberts plots of corrections and response, at specific eta values corresponding to centers of MC Jet eta bins
    TH1F responseBarrel("responseBarrel",(mCorrectorNames[0] + std::string (": Jet Response")).c_str(), net_bins, etbinsBarrel);
    TH1F correctionBarrel("correctionBarrel",(mCorrectorNames[0] + std::string (": Jet Correction")).c_str(), net_bins, etbinsBarrel);
    TH1F responseEndcap("responseEndcap",(mCorrectorNames[0] + std::string (": Jet Response")).c_str(), net_bins, etbinsEndcap);
    TH1F correctionEndcap("correctionEndcap",(mCorrectorNames[0] + std::string (": Jet Correction")).c_str(), net_bins, etbinsEndcap);
    TH1F responseForward("responseForward",(mCorrectorNames[0] + std::string (": Jet Response")).c_str(), net_bins, etbinsForward);
    TH1F correctionForward("correctionForward",(mCorrectorNames[0] + std::string (": Jet Correction")).c_str(), net_bins, etbinsForward);

    // look for correctors.  General approach which doesn't assume only one corrector is applied.
    std::vector <const JetCorrector*> correctors;
    for (unsigned i = 0; i < mCorrectorNames.size(); i++) {
      correctors.push_back (JetCorrector::getJetCorrector (mCorrectorNames [i], fSetup)); 
    }

    // fill plots
    TRandom rndm;

    //Specific eta values for Roberts plots
    double etaBarrel=(0.0+0.226)/2;
    double etaEndcap=(2.295+2.487)/2;
    double etaForward=(4.0+4.4)/2;

    //Loop over et bins for both Robert and Fedors plots
    for (int iet = 1; iet < net_bins+1; iet++) {
      double etmin = corrections.GetXaxis()->GetBinLowEdge (iet);
      double etmax = corrections.GetXaxis()->GetBinUpEdge (iet);
      double etmiddle = sqrt (etmin*etmax);

      //Loop over eta bins for Fedors plots
      for (int ieta = 1; ieta < neta_bins+1; ieta++) {
	double etamin = corrections.GetYaxis()->GetBinLowEdge (ieta);
	double etamax = corrections.GetYaxis()->GetBinUpEdge (ieta);
	double etamiddle = (etamin+etamax)/2.;
//  	std::cout << "iet/etmin/etmax/etmidle ieta/etamin/etamax/etamiddle: " 
//  		  << iet << '/' << etmin << '/' << etmax << '/' << etmiddle << "   "
//  		  << ieta << '/' << etamin << '/' << etamax << '/' << etamiddle << std::endl;
	// get variations
	double sumx = 0;
	double sumx2 = 0;
	const int nprobes = 2;
	for (int irndm = 0; irndm < nprobes; ++irndm) {
	  double et = etmin * pow (etmax/etmin, rndm.Uniform ());
	  double eta = etamin + (etamax-etamin) * rndm.Uniform ();

          //Get a Lorentz vector with correct et and eta.  We are not interested in the phi or mass or pt.
	  math::PtEtaPhiELorentzVector v (et, eta, 0, 0);
	  v.SetE (v.P());
	  reco::Jet::LorentzVector jet (v.Px(), v.Py(), v.Pz(), v.E());

	  double corr = 1;
	  for (unsigned icorr = 0; icorr < correctors.size (); ++icorr){
	    corr = corr * correctors[icorr]->correction (jet);
// 	      std::cout << "pt/eta/correction: " << jet.Et() << '/' << jet.Eta() << '/' << corr << std::endl;
	  }
	  sumx += corr;
	  sumx2 += corr*corr;
	} 
	double mean = sumx / nprobes;
	corrections.Fill (etmiddle, etamiddle, mean);
// 	std::cout << "mean: et/eta/value: " << etmiddle << '/' << etamiddle << '/' << mean << std::endl;
	double sigma = sqrt (sumx2 / nprobes - mean*mean);
	variations.Fill (etmiddle, etamiddle, sigma);
// 	std::cout << "sigma: et/eta/value: " << etmiddle << '/' << etamiddle << '/' << sigma << std::endl;
      }

      //Roberts Barrel Plots
      double etBarrel=correctionBarrel.GetBinCenter(iet);  
      //Get a Lorentz vector with correct et and eta.  We are not interested in the phi or mass or pt.
      math::PtEtaPhiELorentzVector vBarrel (etBarrel, etaBarrel, 0, 0);
      vBarrel.SetE (vBarrel.P());
      reco::Jet::LorentzVector jetBarrel (vBarrel.Px(), vBarrel.Py(), vBarrel.Pz(), vBarrel.E());
      //std::cout << "Et =" << etBarrel << ", jet.Et=" <<jetBarrel.Et() << ", eta=" <<etaBarrel<< "jet.eta=" <<jetBarrel.eta() <<std::endl;
      double corr = correctors[0]->correction(jetBarrel);
      std::cout << "Et=" <<jetBarrel.Et()<<", eta="<<jetBarrel.eta()<<", response="<<1/corr<<", corr="<<corr<<std::endl;
      responseBarrel.Fill (jetBarrel.Et(),1/corr);
      correctionBarrel.Fill (jetBarrel.Et(),corr);      

      //Roberts Endcap Pltos
      double etEndcap=correctionEndcap.GetBinCenter(iet);  
      //Get a Lorentz vector with correct et and eta.  We are not interested in the phi or mass or pt.
      math::PtEtaPhiELorentzVector vEndcap (etEndcap, etaEndcap, 0, 0);
      vEndcap.SetE (vEndcap.P());
      reco::Jet::LorentzVector jetEndcap (vEndcap.Px(), vEndcap.Py(), vEndcap.Pz(), vEndcap.E());
      //std::cout << "Et =" << etEndcap << ", jet.Et=" <<jetEndcap.Et() << ", eta=" <<etaEndcap<< "jet.eta=" <<jetEndcap.eta() <<std::endl;
      corr = correctors[0]->correction(jetEndcap);
      //std::cout << "Et=" <<jetEndcap.Et()<<", eta="<<jetEndcap.eta()<<", response="<<1/corr<<", corr="<<corr<<std::endl;
      responseEndcap.Fill (jetEndcap.Et(),1/corr);
      correctionEndcap.Fill (jetEndcap.Et(),corr);      

      //Roberts Forward Plots
      double etForward=correctionForward.GetBinCenter(iet);  
      //Get a Lorentz vector with correct et and eta.  We are not interested in the phi or mass or pt.
      math::PtEtaPhiELorentzVector vForward (etForward, etaForward, 0, 0);
      vForward.SetE (vForward.P());
      reco::Jet::LorentzVector jetForward (vForward.Px(), vForward.Py(), vForward.Pz(), vForward.E());
      //std::cout << "Et =" << etForward << ", jet.Et=" <<jetForward.Et() << ", eta=" <<etaForward<< "jet.eta=" <<jetForward.eta() <<std::endl;
      corr = correctors[0]->correction(jetForward);
      //std::cout << "Et=" <<jetForward.Et()<<", eta="<<jetForward.eta()<<", response="<<1/corr<<", corr="<<corr<<std::endl;
      responseForward.Fill (jetForward.Et(),1/corr);
      correctionForward.Fill (jetForward.Et(),corr);      
    }
    file.Write ();
    mAllDone = true;
  }
}
