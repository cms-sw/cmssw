#ifndef JetMETCorrections_Type1MET_AddCorrectionsToGenericMET_h
#define JetMETCorrections_Type1MET_AddCorrectionsToGenericMET_h

/** \class AddCorrectionsToGenericMET
*
* generic class for MET corrections 
*
* \authors Matthieu Marionneau, ETHZ
*
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

class AddCorrectionsToGenericMET {
public:
  AddCorrectionsToGenericMET(){};
  ~AddCorrectionsToGenericMET(){};

  void setCorTokens(std::vector<edm::EDGetTokenT<CorrMETData> > const& corrTokens);

  reco::MET getCorrectedMET(const reco::MET& srcMET, edm::Event& evt);
  reco::PFMET getCorrectedPFMET(const reco::PFMET& srcMET, edm::Event& evt);
  reco::CaloMET getCorrectedCaloMET(const reco::CaloMET& srcMET, edm::Event& evt);

private:
  CorrMETData getCorrection(const reco::MET& srcMET, edm::Event& evt);

  std::vector<edm::EDGetTokenT<CorrMETData> > corrTokens_;
  reco::Candidate::LorentzVector constructP4From(const reco::MET& met, const CorrMETData& correction);
};

#endif
