#include "JetMETCorrections/Type1MET/interface/AddCorrectionsToGenericMET.h"

void
AddCorrectionsToGenericMET::setCorTokens(std::vector<edm::EDGetTokenT<CorrMETData> > corrTokens){
  corrTokens_=corrTokens;
}

reco::Candidate::LorentzVector 
AddCorrectionsToGenericMET::constructP4From(const reco::MET& met,
					    const CorrMETData& correction) {
  double px = met.px() + correction.mex;
  double py = met.py() + correction.mey;
  double pt = sqrt(px*px + py*py);
  return reco::Candidate::LorentzVector(px, py, 0., pt);
}


CorrMETData
AddCorrectionsToGenericMET::getCorrection(const reco::MET& srcMET, edm::Event& evt, const edm::EventSetup& es) {
  CorrMETData sumCor;
  edm::Handle<CorrMETData> corr;
  for (std::vector<edm::EDGetTokenT<CorrMETData> >::const_iterator corrToken = corrTokens_.begin(); corrToken != corrTokens_.end(); ++corrToken) {
    evt.getByToken(*corrToken, corr);
    sumCor += (*corr);
  }

  return sumCor;
}
 

reco::MET
AddCorrectionsToGenericMET::getCorrectedMET(const reco::MET& srcMET, edm::Event& evt, const edm::EventSetup& es) {
  
  CorrMETData corr = getCorrection(srcMET, evt, es);
  reco::MET outMET(srcMET.sumEt()+corr.sumet, constructP4From(srcMET, corr), srcMET.vertex() );
  
  return outMET;
}

//specific flavors ================================
reco::PFMET
AddCorrectionsToGenericMET::getCorrectedPFMET(const reco::PFMET& srcMET, edm::Event& evt, const edm::EventSetup& es) {
  
  CorrMETData corr = getCorrection(srcMET, evt, es);
  reco::PFMET outMET(srcMET.getSpecific(),srcMET.sumEt()+corr.sumet, constructP4From(srcMET, corr), srcMET.vertex() );
  
  return outMET;
}


reco::CaloMET
AddCorrectionsToGenericMET::getCorrectedCaloMET(const reco::CaloMET& srcMET, edm::Event& evt, const edm::EventSetup& es) {
  
   CorrMETData corr = getCorrection(srcMET, evt, es);
   reco::CaloMET outMET(srcMET.getSpecific(),srcMET.sumEt()+corr.sumet, constructP4From(srcMET, corr), srcMET.vertex() );
  
  return outMET;
}
