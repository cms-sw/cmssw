#include "RecoEgamma/ElectronIdentification/interface/ClassBasedElectronID.h"

// ===========================================================================================================
void ClassBasedElectronID::setup(const edm::ParameterSet& conf) 
// ===========================================================================================================
{


  // Get all the parameters
  //baseSetup(conf);
  
  quality_ =  conf.getParameter<std::string>("electronQuality");
  
	if(quality_=="Eff95Cuts") {
		cuts_ = conf.getParameter<edm::ParameterSet>("Eff95Cuts");
	}
	
	else if(quality_=="Eff90Cuts") {
		cuts_ = conf.getParameter<edm::ParameterSet>("Eff90Cuts");
	}
	
	else {
    edm::LogError("ClassBasedElectronID") << "Invalid electronQuality parameter: must be tight, medium or loose." ;
    exit (1);
  }
  
} // end of setup

double ClassBasedElectronID::result(const reco::GsfElectron* electron,
                                    const edm::Event& e ,
									const edm::EventSetup& es) 
{

  //determine which element of the cut arrays in cfi file to read
  //depending on the electron classification
  int icut=0;
  int elClass = electron->classification() ;
  if (electron->isEB()) //barrel
     {
       if (elClass == reco::GsfElectron::GOLDEN)    icut=0;
       if (elClass == reco::GsfElectron::BIGBREM)   icut=1;
       if (elClass == reco::GsfElectron::SHOWERING) icut=2;
       if (elClass == reco::GsfElectron::GAP)       icut=6;
     }
  if (electron->isEE()) //endcap
     {
       if (elClass == reco::GsfElectron::GOLDEN)    icut=3;
       if (elClass == reco::GsfElectron::BIGBREM)   icut=4;
       if (elClass == reco::GsfElectron::SHOWERING) icut=5;
       if (elClass == reco::GsfElectron::GAP)       icut=7;
     }
  if (elClass == reco::GsfElectron::UNKNOWN) 
     {
       edm::LogError("ClassBasedElectronID") << "Error: unrecognized electron classification ";
       return 1.;
     }

  bool useDeltaEtaIn       = true;
  bool useSigmaIetaIeta    = true;
  bool useHoverE           = true;
  bool useEoverPOut        = true;
  bool useDeltaPhiInCharge = true;

  // DeltaEtaIn
  if (useDeltaEtaIn) { 
    double value = electron->deltaEtaSuperClusterTrackAtVtx();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("deltaEtaIn");
    if (fabs(value)>maxcut[icut]) return 0.;
  } 

  // SigmaIetaIeta
  if(useSigmaIetaIeta) {
    double value = electron->sigmaIetaIeta() ; 
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("sigmaIetaIetaMax");
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("sigmaIetaIetaMin");
    if(value<mincut[icut] || value>maxcut[icut]) return 0.;   
  } 

  // H/E
  if (useHoverE) { //_[variables_]) {
    double value = electron->hadronicOverEm();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("HoverE");
    if (value>maxcut[icut]) return 0.;
  } // if use
 
  // Eseed/Pout
  if (useEoverPOut) { 
	double value = electron->eSeedClusterOverPout();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("EoverPOutMax");
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("EoverPOutMin");
    if (value<mincut[icut] || value>maxcut[icut]) return 0.;
  } 

  // DeltaPhiIn*Charge
  if (useDeltaPhiInCharge) { 
    double value1 = electron->deltaPhiSuperClusterTrackAtVtx();
    double value2 = electron->charge();
    double value  = value1*value2;
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("deltaPhiInChargeMax");
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("deltaPhiInChargeMin");
    if (value<mincut[icut] || value>maxcut[icut]) return 0.;
  } 
  
  return 1.;

}
