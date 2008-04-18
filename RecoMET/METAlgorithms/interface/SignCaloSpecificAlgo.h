#ifndef METProducers_SignCaloMETAlgo_h
#define METProducers_SignCaloMETAlgo_h
// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SignCaloSpecificAlgo
// 
/**\class METSignificance SignCaloSpecificAlgo.h RecoMET/METAlgorithms/include/SignCaloSpecificAlgo.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id$
//
//
// 
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "TF1.h"

class SignCaloSpecificAlgo 
{
 public:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  typedef std::vector <const reco::Candidate*> TowerCollection;
  reco::CaloMET addInfo(edm::Handle<edm::View<reco::Candidate> > towers, CommonMETData met, const metsig::SignAlgoResolutions & resolutions, bool noHF, double globalthreshold);

    
  
};

#endif
