#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"
#include "CLHEP/HepMC/GenEvent.h"

using namespace reco;
using namespace std;

//-------------------------------------------------------------------------
// This algorithm adds calorimeter specific global event information to 
// the MET object which may be useful/needed for MET Data Quality Monitoring
// and MET cleaning.  This list is not exhaustive and additional 
// information will be added in the future. 
//-------------------------------------
reco::GenMET GenSpecificAlgo::addInfo(const CandidateCollection *towers, CommonMETData met)
{ 
  // Instantiate the container to hold the calorimeter specific information
  SpecificGenMETData specific;
  // Initialise the container 
  specific.m_EmEnergy        = 0.0;        // EM Energy
  specific.m_HadEnergy       = 0.0;        // Hadronic Energy
  specific.m_InvisibleEnergy = 0.0;        // Invisible energy
  specific.m_AuxiliaryEnergy = 0.0;        // Other Energy
  // Instantiate containers for the MET candidate and initialise them with
  // the MET information in "met" (of type CommonMETData)
  const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
  const Point vtx( 0.0, 0.0, 0.0 );
  // Create and return an object of type GenMET, which is a MET object with 
  // the extra calorimeter specfic information added
  GenMET specificmet( specific, met.sumet, p4, vtx );
  return specificmet;
}
//-------------------------------------------------------------------------
