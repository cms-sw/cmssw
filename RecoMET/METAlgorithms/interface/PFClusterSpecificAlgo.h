// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      PFClusterSpecificAlgo
// 
/**\class PFClusterSpecificAlgo PFClusterSpecificAlgo.h RecoMET/METAlgorithms/interface/PFClusterSpecificAlgo.h

 Description: Adds Particle Flow specific information to MET

 Implementation:
     [Notes on implementation]
*/
//
// Original Authors:  R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
//          Created:  October 27, 2008
// $Id: PFClusterSpecificAlgo.h,v 1.3 2012/06/11 04:27:43 sakuma Exp $
//
//
#ifndef METAlgorithms_PFClusterMETInfo_h
#define METAlgorithms_PFClusterMETInfo_h

//____________________________________________________________________________||
#include "DataFormats/METReco/interface/PFClusterMET.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"


//____________________________________________________________________________||
class PFClusterSpecificAlgo
{
 public:
  PFClusterSpecificAlgo() {;}
  reco::PFClusterMET addInfo(edm::Handle<edm::View<reco::Candidate> > PFClusterCandidates, CommonMETData met);

private:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;

};

//____________________________________________________________________________||
#endif // METAlgorithms_PFClusterMETInfo_h

