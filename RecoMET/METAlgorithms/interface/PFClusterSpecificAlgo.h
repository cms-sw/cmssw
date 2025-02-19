#ifndef METAlgorithms_PFClusterMETInfo_h
#define METAlgorithms_PFClusterMETInfo_h

// Adds Particle Flow specific information to MET base class
// Author: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
// First Implementation: 10/27/08


#include "DataFormats/METReco/interface/PFClusterMET.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"



class PFClusterSpecificAlgo
{
 public:
  PFClusterSpecificAlgo() {;}
  
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  reco::PFClusterMET addInfo(edm::Handle<edm::View<reco::Candidate> > PFClusterCandidates, CommonMETData met);

};

#endif

