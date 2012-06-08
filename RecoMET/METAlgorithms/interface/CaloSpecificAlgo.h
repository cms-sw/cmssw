// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      CaloSpecificAlgo
// 
/**\class CaloSpecificAlgo CaloSpecificAlgo.h RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h

 Description: Adds Calorimeter specific information to MET base class

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  R. Cavanaugh (taken from F.Ratnikov, UMd)
//         Created:  June 6, 2006
// $Id: METAlgo.h,v 1.12 2012/06/08 00:51:27 sakuma Exp $
//
//
#ifndef METProducers_CaloMETInfo_h
#define METProducers_CaloMETInfo_h

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
  reco::CaloMET addInfo(edm::Handle<edm::View<reco::Candidate> > towers, CommonMETData met, bool noHF, double globalThreshold);
};

#endif
