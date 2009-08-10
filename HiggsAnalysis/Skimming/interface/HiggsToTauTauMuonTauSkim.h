#ifndef HiggsToTauTauMuonTauSkim_h
#define HiggsToTauTauMuonTauSkim_h

/** \class HiggsToTauTauMuonTauSkim
 *
 *  
 *  Filter to select events passing 
 *  single muon HLT
 *  1 calojet (Et>15 GeV, eta<2.6 not overlapping DeltaR=0.5 with
 *  the trigger muon)
 *  \author Monica Vazquez Acosta  -  Imperial College London
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

using namespace edm;
using namespace std;

#include <math.h>

class HiggsToTauTauMuonTauSkim : public edm::EDFilter {

    public:
        explicit HiggsToTauTauMuonTauSkim(const edm::ParameterSet&);
        ~HiggsToTauTauMuonTauSkim();

  	virtual bool filter(edm::Event&, const edm::EventSetup& );

   private:
	double deltaPhi(double phi1, double phi2){
	    const double PI = 3.1415926535;	
	    // in ORCA phi = [0,2pi], in TLorentzVector phi = [-pi,pi].
	    // With the conversion below deltaPhi works ok despite the
	    // 2*pi difference in phi definitions.
	    if(phi1 < 0) phi1 += 2*PI;
	    if(phi2 < 0) phi2 += 2*PI;
	
	    double dphi = fabs(phi1-phi2);
	
	    if(dphi > PI) dphi = 2*PI - dphi;
	    return dphi;
	}

	double deltaR(double eta1, double eta2, double phi1, double phi2){
	    double dphi = deltaPhi(phi1,phi2);
	    double deta = fabs(eta1-eta2);
	    return sqrt(dphi*dphi + deta*deta);
	}

	bool 		debug;
        InputTag        hltResultsLabel;
        InputTag        hltEventLabel;
        std::vector<std::string>  hltMuonBits;
        std::vector<std::string>  hltFilterLabels;
	InputTag	jetLabel;
        int 		minNumberOfjets;
        double 		jetEtMin;
        double 		jetEtaMin;
        double 		jetEtaMax;
	double		minDRFromMuon;

        int nEvents, nSelectedEvents;
};
#endif


   
