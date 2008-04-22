#ifndef RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_
#define RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include <map>
#include <string>

class EgammaSCEnergyCorrectionAlgo
{
  public:
    // the Verbosity levels
    enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 }; 

    // public member functions
    EgammaSCEnergyCorrectionAlgo(double noise, 
				 reco::AlgoId theAlgo,
				 const edm::ParameterSet& pset, 
				 VerbosityLevel verbosity = pERROR
                                 );
    ~EgammaSCEnergyCorrectionAlgo();

    // take a SuperCluster and return a corrected SuperCluster
    reco::SuperCluster applyCorrection(const reco::SuperCluster &cl, const EcalRecHitCollection &rhc, reco::AlgoId theAlgo, const CaloSubdetectorGeometry* geometry);
 
    // function to set the verbosity level
    void setVerbosity(VerbosityLevel verbosity)
    {
        verbosity_ = verbosity;
    }
 
  private:
    // Zhang shower leakage corrections
    double fEta(double e, double eta);
    // F(brem) correction with brem = phiWidth/etaWidth
    double fBrem(double e, double brem);
    // F(et, eta) correction
    double fEtEta(double et, double eta);

    // correction factor as a function of number of crystals,
    // BasicCluster algo and location in the detector    
    float fNCrystals(int nCry, reco::AlgoId theAlgo, EcalSubdetector theBase);

    // Return the number of crystals in a BasicCluster above 
    // 2sigma noise level
    int nCrystalsGT2Sigma(const reco::BasicCluster &seed);
	double sigmaElectronicNoise_;

    //  map to hold the RecHits
    std::map<DetId, EcalRecHit> *recHits_m;

    //  the verbosity level
    VerbosityLevel verbosity_;

    reco::AlgoId theAlgo_;

    //Paramete sets for corrections functions
    std::vector<double> fBrem_; 
    std::vector<double> fEtEta_; 

    double brLinearThr_;
};

#endif /*RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_*/
