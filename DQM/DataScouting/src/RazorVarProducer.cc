#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQM/DataScouting/interface/RazorVarProducer.h"

#include "TVector3.h"

#include <memory>
#include <vector>

//
// constructors and destructor
//
RazorVarProducer::RazorVarProducer(const edm::ParameterSet &iConfig)
    : inputTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      inputMetTag_(iConfig.getParameter<edm::InputTag>("inputMetTag")) {
  // set Token(-s)
  inputTagToken_ = consumes<std::vector<math::XYZTLorentzVector>>(iConfig.getParameter<edm::InputTag>("inputTag"));
  inputMetTagToken_ = consumes<reco::CaloMETCollection>(iConfig.getParameter<edm::InputTag>("inputMetTag"));

  produces<std::vector<double>>();

  LogDebug("") << "Inputs: " << inputTag_.encode() << " " << inputMetTag_.encode() << ".";
}

// ------------ method called to produce the data  ------------
void RazorVarProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // get hold of collection of objects
  Handle<vector<math::XYZTLorentzVector>> hemispheres;
  iEvent.getByToken(inputTagToken_, hemispheres);

  // get hold of the MET Collection
  Handle<CaloMETCollection> inputMet;
  iEvent.getByToken(inputMetTagToken_, inputMet);

  std::unique_ptr<std::vector<double>> result(new std::vector<double>);
  // check the the input collections are available
  if (hemispheres.isValid() && inputMet.isValid() && hemispheres->size() > 1) {
    TLorentzVector ja(hemispheres->at(0).x(), hemispheres->at(0).y(), hemispheres->at(0).z(), hemispheres->at(0).t());
    TLorentzVector jb(hemispheres->at(1).x(), hemispheres->at(1).y(), hemispheres->at(1).z(), hemispheres->at(1).t());

    std::vector<math::XYZTLorentzVector> muonVec;
    const double MR = CalcMR(ja, jb);
    const double R = CalcR(MR, ja, jb, inputMet, muonVec);
    result->push_back(MR);
    result->push_back(R);
  }
  iEvent.put(std::move(result));
}

double RazorVarProducer::CalcMR(TLorentzVector ja, TLorentzVector jb) const {
  if (ja.Pt() <= 0.1)
    return -1;

  ja.SetPtEtaPhiM(ja.Pt(), ja.Eta(), ja.Phi(), 0.0);
  jb.SetPtEtaPhiM(jb.Pt(), jb.Eta(), jb.Phi(), 0.0);

  if (ja.Pt() > jb.Pt()) {
    TLorentzVector temp = ja;
    ja = jb;
    jb = temp;
  }

  double A = ja.P();
  double B = jb.P();
  double az = ja.Pz();
  double bz = jb.Pz();
  TVector3 jaT, jbT;
  jaT.SetXYZ(ja.Px(), ja.Py(), 0.0);
  jbT.SetXYZ(jb.Px(), jb.Py(), 0.0);
  double ATBT = (jaT + jbT).Mag2();

  double MR = sqrt((A + B) * (A + B) - (az + bz) * (az + bz) -
                   (jbT.Dot(jbT) - jaT.Dot(jaT)) * (jbT.Dot(jbT) - jaT.Dot(jaT)) / (jaT + jbT).Mag2());

  double mybeta = (jbT.Dot(jbT) - jaT.Dot(jaT)) / sqrt(ATBT * ((A + B) * (A + B) - (az + bz) * (az + bz)));

  double mygamma = 1. / sqrt(1. - mybeta * mybeta);

  // use gamma times MRstar
  return MR * mygamma;
}

double RazorVarProducer::CalcR(double MR,
                               const TLorentzVector &ja,
                               const TLorentzVector &jb,
                               edm::Handle<reco::CaloMETCollection> inputMet,
                               const std::vector<math::XYZTLorentzVector> &muons) const {
  // now we can calculate MTR
  TVector3 met;
  met.SetPtEtaPhi((inputMet->front()).pt(), 0.0, (inputMet->front()).phi());

  std::vector<math::XYZTLorentzVector>::const_iterator muonIt;
  for (muonIt = muons.begin(); muonIt != muons.end(); muonIt++) {
    TVector3 tmp;
    tmp.SetPtEtaPhi(muonIt->pt(), 0, muonIt->phi());
    met -= tmp;
  }

  double MTR = sqrt(0.5 * (met.Mag() * (ja.Pt() + jb.Pt()) - met.Dot(ja.Vect() + jb.Vect())));

  // filter events
  return float(MTR) / float(MR);  // R
}

DEFINE_FWK_MODULE(RazorVarProducer);
