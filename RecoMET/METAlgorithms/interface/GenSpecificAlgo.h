#ifndef METProducers_GenMETInfo_h
#define METProducers_GenMETInfo_h

/// Adds generator level HEPMC specific information to MET base class
/// Author: R. Cavanaugh (taken from F.Ratnikov, UMd)
/// 6 June, 2006

#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

class GenSpecificAlgo 
{
 public:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  typedef std::vector <const reco::Candidate*> ParticleCollection;
  /// Make GenMET. Assumes MET is made from MCCandidates
  //reco::GenMET addInfo(const reco::CandidateCollection *particles, CommonMETData met);
  reco::GenMET addInfo(edm::Handle<edm::View<reco::Candidate> > particles, CommonMETData *met, double globalThreshold, bool onlyFiducial=false, bool usePt=false);
};

#endif
