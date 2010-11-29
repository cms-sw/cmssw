#ifndef METAlgorithms_PFMETInfo_h
#define METAlgorithms_PFMETInfo_h

// Adds Particle Flow specific information to MET base class
// Author: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
// First Implementation: 10/27/08


#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/SignPFSpecificAlgo.h"
#include "DataFormats/JetReco/interface/PFJet.h"


class PFSpecificAlgo
{
 public:
  PFSpecificAlgo() : alsocalcsig(false), pfsignalgo_() {;}
  
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  void runSignificance(metsig::SignAlgoResolutions & resolutions, edm::Handle<edm::View<reco::PFJet> > jets);
  reco::PFMET addInfo(edm::Handle<edm::View<reco::Candidate> > PFCandidates, CommonMETData met);

 private:
  bool alsocalcsig;
  metsig::SignPFSpecificAlgo pfsignalgo_;
};

#endif

