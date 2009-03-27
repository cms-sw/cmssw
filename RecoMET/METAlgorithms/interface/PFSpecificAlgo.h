#ifndef METAlgorithms_PFMETInfo_h
#define METAlgorithms_PFMETInfo_h

// Adds Particle Flow specific information to MET base class
// Author: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
// First Implementation: 10/27/08


#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

class PFSpecificAlgo
{
 public:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  reco::PFMET addInfo(edm::Handle<edm::View<reco::Candidate> > PFCandidates, CommonMETData met);
};

#endif

