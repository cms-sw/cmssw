// Template to demonstrate accessing to correction service from user analyzer
// plot correction as a function of et:eta
// author: F.Ratnikov UMd Mar. 16, 2007
// 
#include "PlotJetCorrections.h"

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
    const double et_min = 1.;
    const double et_max = 1e6;
    const int net_bins = 100;

    double etbins [net_bins+1];
    for (int i = 0; i < net_bins+1; ++i) {
      etbins[i] =  et_min * pow (et_max/et_min, double(i)/double(net_bins));
    }

    TH2D corrections ("Correction Scale", (mCorrectorNames[0] + std::string (": Jet Correction Function")).c_str(), 
		      net_bins, etbins, neta_bins, 0, eta_max);
    corrections.GetXaxis()->SetTitle ("Et");
    corrections.GetYaxis()->SetTitle ("Eta");
    TH2D variations ("Variation of Correction Scale", (mCorrectorNames[0] + std::string (": Jet Correction Variations")).c_str(),
		     net_bins, etbins, neta_bins, 0, eta_max);
    variations.GetXaxis()->SetTitle ("Et");
    variations.GetYaxis()->SetTitle ("Eta");

    // look for correctors
    std::vector <const JetCorrector*> correctors;
    for (unsigned i = 0; i < mCorrectorNames.size(); i++) {
      correctors.push_back (JetCorrector::getJetCorrector (mCorrectorNames [i], fSetup)); 
    }

    // fill plot
    TRandom rndm;
    for (int iet = 1; iet < net_bins+1; iet++) {
      double etmin = corrections.GetXaxis()->GetBinLowEdge (iet);
      double etmax = corrections.GetXaxis()->GetBinUpEdge (iet);
      double etmiddle = sqrt (etmin*etmax);
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
    }
    file.Write ();
    mAllDone = true;
  }
}
