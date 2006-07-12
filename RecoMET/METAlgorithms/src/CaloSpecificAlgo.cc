#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"

using namespace reco;
using namespace std;

//-------------------------------------------------------------------------
// This algorithm adds calorimeter specific global event information to 
// the MET object which may be useful/needed for MET Data Quality Monitoring
// and MET cleaning.  This list is not exhaustive and additional 
// information will be added in the future. 
//-------------------------------------
reco::CaloMET CaloSpecificAlgo::addInfo(CommonMETData met)
{ 
  // Instantiate the container to hold the calorimeter specific information
  SpecificCaloMETData specific;
  // Fill the container 
  specific.mMaxEInEmTowers = 1.0;         // Maximum energy in EM towers
  specific.mMaxEInHadTowers = 1.0;        // Maximum energy in HCAL towers
  specific.mHadEnergyInHO = 1.0;          // Hadronic energy fraction in HO
  specific.mHadEnergyInHB = 1.0;          // Hadronic energy in HB
  specific.mHadEnergyInHF = 1.0;          // Hadronic energy in HF
  specific.mHadEnergyInHE = 1.0;          // Hadronic energy in HE
  specific.mEmEnergyInEB = 1.0;           // Em energy in EB
  specific.mEmEnergyInEE = 1.0;           // Em energy in EE
  specific.mEmEnergyInHF = 1.0;           // Em energy in HF
  specific.mEnergyFractionHadronic = 1.0; // Hadronic energy fraction
  specific.mEnergyFractionEm = 1.0;       // Em energy fraction
  // Instantiate containers for the MET candidate and initialise them with
  // the MET information in "met" (of type CommonMETData)
  const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
  const Point vtx( 0.0, 0.0, 0.0 );
  // Create and return an object of type CaloMET, which is a MET object with 
  // the extra calorimeter specfic information added
  CaloMET specificmet( specific, met.sumet, p4, vtx );
  return specificmet;
}
//-------------------------------------------------------------------------
