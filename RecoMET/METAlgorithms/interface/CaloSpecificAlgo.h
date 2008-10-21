#ifndef METProducers_CaloMETInfo_h
#define METProducers_CaloMETInfo_h

/// Adds Calorimeter specific information to MET base class
/// Author: R. Cavanaugh (taken from F.Ratnikov, UMd)
/// 6 June, 2006

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

class CaloSpecificAlgo 
{
 public:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  typedef std::vector <const reco::Candidate*> TowerCollection;
  /// Make CaloMET. Assumes MET is made from CaloTowerCandidates
  //reco::CaloMET addInfo(const reco::CandidateCollection *towers, CommonMETData met);
  reco::CaloMET addInfo(edm::Handle<edm::View<reco::Candidate> > towers, CommonMETData met, bool noHF, double globalThreshold);
};

#endif
