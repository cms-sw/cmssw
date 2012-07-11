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
// $Id: SignCaloSpecificAlgo.h,v 1.3 2009/10/21 11:27:11 fblekman Exp $
//
//
// 
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/SigInputObj.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "TF1.h"
#include "TMatrixD.h"


class SignCaloSpecificAlgo 
{
 public:

  SignCaloSpecificAlgo();
  ~SignCaloSpecificAlgo();

  
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  typedef std::vector <const reco::Candidate*> TowerCollection;
  void usePreviousSignif(const std::vector<double> &values);
  void usePreviousSignif(const TMatrixD &matrix){matrix_=matrix;}
  double getSignificance(){return significance_;}
  TMatrixD getSignificanceMatrix()const {return matrix_;}

  void calculateBaseCaloMET(edm::Handle<edm::View<reco::Candidate> > towers,  CommonMETData met, const metsig::SignAlgoResolutions & resolutions, bool noHF, double globalthreshold);
  
 private:
  
  std::vector<metsig::SigInputObj> makeVectorOutOfCaloTowers(edm::Handle<edm::View<reco::Candidate> > towers, const metsig::SignAlgoResolutions& resolutions, bool noHF, double globalthreshold);
  
  double significance_;
  TMatrixD matrix_;
};

#endif
