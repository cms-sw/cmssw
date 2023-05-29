#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQM/DataScouting/interface/AlphaTVarProducer.h"

#include "TVector3.h"

#include <memory>
#include <vector>

//
// constructors and destructor
//
AlphaTVarProducer::AlphaTVarProducer(const edm::ParameterSet &iConfig)
    : inputJetTag_(iConfig.getParameter<edm::InputTag>("inputJetTag")) {
  produces<std::vector<double>>();

  // set Token(-s)
  inputJetTagToken_ = consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("inputJetTag"));

  LogDebug("") << "Inputs: " << inputJetTag_.encode() << " ";
}

// ------------ method called to produce the data  ------------
void AlphaTVarProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // get hold of collection of objects
  edm::Handle<reco::CaloJetCollection> calojet_handle;
  iEvent.getByToken(inputJetTagToken_, calojet_handle);

  std::unique_ptr<std::vector<double>> result(new std::vector<double>);
  // check the the input collections are available
  if (calojet_handle.isValid()) {
    std::vector<TLorentzVector> myJets;
    reco::CaloJetCollection::const_iterator jetIt;
    for (jetIt = calojet_handle->begin(); jetIt != calojet_handle->end(); jetIt++) {
      TLorentzVector j;
      j.SetPtEtaPhiE(jetIt->pt(), jetIt->eta(), jetIt->phi(), jetIt->energy());
      myJets.push_back(j);
    }

    double alphaT = CalcAlphaT(myJets);
    double HT = CalcHT(myJets);

    result->push_back(alphaT);
    result->push_back(HT);
  }
  iEvent.put(std::move(result));
}

double AlphaTVarProducer::CalcAlphaT(const std::vector<TLorentzVector> &jets) const {
  std::vector<double> ETs;
  TVector3 MHT{CalcMHT(jets), 0.0, 0.0};
  float HT = CalcHT(jets);
  // float HT = 0;
  for (unsigned int i = 0; i < jets.size(); i++) {
    if (jets[i].Et() > 50. && fabs(jets[i].Eta()) < 2.5)
      ETs.push_back(jets[i].Et());
    // HT += jets[i].Et();
  }
  if (ETs.size() < 2.)
    return 0.0;
  if (ETs.size() > 16.)
    return 0.0;
  float DHT = deltaHt(ETs);

  float AlphaT = alphaT(HT, DHT, MHT.Mag());

  return AlphaT;
}

double AlphaTVarProducer::deltaHt(const std::vector<double> &ETs) {
  if (ETs.size() > 16.)
    return 9999999;
  std::vector<double> diff(1 << (ETs.size() - 1), 0.);
  for (unsigned i = 0; i < diff.size(); i++)
    for (unsigned j = 0; j < ETs.size(); j++)
      diff[i] += ETs[j] * (1 - 2 * (int(i >> j) & 1));
  std::vector<double>::const_iterator it;
  double min = 9999999;
  for (it = diff.begin(); it != diff.end(); it++)
    if (*it < min)
      min = *it;
  return min;
}

double AlphaTVarProducer::alphaT(const double HT, const double DHT, const double MHT) {
  return 0.5 * (HT - DHT) / sqrt(HT * HT - MHT * MHT);
}

double AlphaTVarProducer::CalcHT(const std::vector<TLorentzVector> &jets) {
  double HT = 0;
  for (unsigned int i = 0; i < jets.size(); i++) {
    if (jets[i].Et() > 50. && fabs(jets[i].Eta()) < 2.5)
      HT += jets[i].Et();
  }

  return HT;
}

double AlphaTVarProducer::CalcMHT(const std::vector<TLorentzVector> &jets) {
  TVector3 MHT;
  for (unsigned int i = 0; i < jets.size(); i++) {
    if (jets[i].Et() > 50. && fabs(jets[i].Eta()) < 2.5)
      MHT -= jets[i].Vect();
  }
  MHT.SetZ(0.0);
  return MHT.Mag();
}

DEFINE_FWK_MODULE(AlphaTVarProducer);
