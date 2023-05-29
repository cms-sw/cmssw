#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQM/DataScouting/interface/DiJetVarProducer.h"

#include "TLorentzVector.h"
#include "TVector3.h"

#include <memory>
#include <vector>

//
// constructors and destructor
//
DiJetVarProducer::DiJetVarProducer(const edm::ParameterSet &iConfig)
    : inputJetTag_(iConfig.getParameter<edm::InputTag>("inputJetTag")),
      wideJetDeltaR_(iConfig.getParameter<double>("wideJetDeltaR")) {
  // register your products
  // produces<std::vector<double> >("dijetvariables");
  produces<std::vector<math::PtEtaPhiMLorentzVector>>("widejets");

  // set Token(-s)
  inputJetTagToken_ = consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("inputJetTag"));

  LogDebug("") << "Input Jet Tag: " << inputJetTag_.encode() << " ";
  LogDebug("") << "Radius Parameter Wide Jet: " << wideJetDeltaR_ << ".";
}

// ------------ method called to produce the data  ------------
void DiJetVarProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // ## The output collections
  // std::unique_ptr<std::vector<double> > dijetvariables(new
  // std::vector<double>);
  std::unique_ptr<std::vector<math::PtEtaPhiMLorentzVector>> widejets(new std::vector<math::PtEtaPhiMLorentzVector>);

  // ## Get jet collection
  edm::Handle<reco::CaloJetCollection> calojets_handle;
  iEvent.getByToken(inputJetTagToken_, calojets_handle);
  // cout << "size: " << calojets_handle->size() << endl;

  // ## Wide Jet Algorithm
  // At least two jets
  if (calojets_handle->size() >= 2) {
    TLorentzVector wj1_tmp;
    TLorentzVector wj2_tmp;
    TLorentzVector wj1;
    TLorentzVector wj2;
    TLorentzVector wdijet;

    //        // Loop over all the input jets
    //        for(reco::CaloJetCollection::const_iterator it =
    //        calojets_handle->begin(); it != calojets_handle->end(); ++it)
    //        	 {
    // 	   cout << "jet: " << it->pt() << " " << it->eta() << " " << it->phi()
    // << endl;
    //        	 }

    // Find two leading jets
    TLorentzVector jet1, jet2;

    reco::CaloJetCollection::const_iterator j1 = calojets_handle->begin();
    reco::CaloJetCollection::const_iterator j2 = j1;
    ++j2;

    jet1.SetPtEtaPhiM(j1->pt(), j1->eta(), j1->phi(), j1->mass());
    jet2.SetPtEtaPhiM(j2->pt(), j2->eta(), j2->phi(), j2->mass());

    // cout << "j1: " << jet1.Pt() << " " << jet1.Eta() << " " << jet1.Phi() <<
    // endl; cout << "j2: " << jet2.Pt() << " " << jet2.Eta() << " " <<
    // jet2.Phi() << endl;

    // Create wide jets (radiation recovery algorithm)
    for (reco::CaloJetCollection::const_iterator it = calojets_handle->begin(); it != calojets_handle->end(); ++it) {
      TLorentzVector currentJet;
      currentJet.SetPtEtaPhiM(it->pt(), it->eta(), it->phi(), it->mass());

      double DeltaR1 = currentJet.DeltaR(jet1);
      double DeltaR2 = currentJet.DeltaR(jet2);

      if (DeltaR1 < DeltaR2 && DeltaR1 < wideJetDeltaR_) {
        wj1_tmp += currentJet;
      } else if (DeltaR2 < wideJetDeltaR_) {
        wj2_tmp += currentJet;
      }
    }

    // Re-order the wide jets in pT
    if (wj1_tmp.Pt() > wj2_tmp.Pt()) {
      wj1 = wj1_tmp;
      wj2 = wj2_tmp;
    } else {
      wj1 = wj2_tmp;
      wj2 = wj1_tmp;
    }

    // Create dijet system
    wdijet = wj1 + wj2;

    //        cout << "j1 wide: " << wj1.Pt() << " " << wj1.Eta() << " " <<
    //        wj1.Phi() << " " << wj1.M() << endl; cout << "j2 wide: " <<
    //        wj2.Pt() << " " << wj2.Eta() << " " << wj2.Phi() << " " << wj2.M()
    //        << endl; cout << "MJJWide: " << wdijet.M() << endl; cout <<
    //        "DeltaEtaJJWide: " << fabs(wj1.Eta()-wj2.Eta()) << endl; cout <<
    //        "DeltaPhiJJWide: " << fabs(wj1.DeltaPhi(wj2)) << endl;

    //        // Put variables in the container
    //        dijetvariables->push_back( wdijet.M() );                 //0 =
    //        MJJWide dijetvariables->push_back( fabs(wj1.Eta()-wj2.Eta()) );
    //        //1 = DeltaEtaJJWide dijetvariables->push_back(
    //        fabs(wj1.DeltaPhi(wj2)) );    //2 = DeltaPhiJJWide

    // Put widejets in the container
    math::PtEtaPhiMLorentzVector wj1math(wj1.Pt(), wj1.Eta(), wj1.Phi(), wj1.M());
    math::PtEtaPhiMLorentzVector wj2math(wj2.Pt(), wj2.Eta(), wj2.Phi(), wj2.M());
    widejets->push_back(wj1math);
    widejets->push_back(wj2math);
  }
  //    else
  //      {
  //        // Put variables in the container
  //        dijetvariables->push_back( -1 );                //0 = MJJWide
  //        dijetvariables->push_back( -1 );                //1 = DeltaEtaJJWide
  //        dijetvariables->push_back( -1 );                //2 = DeltaPhiJJWide
  //      }

  // ## Put objects in the Event
  // iEvent.put(std::move(dijetvariables), "dijetvariables");
  iEvent.put(std::move(widejets), "widejets");
}

DEFINE_FWK_MODULE(DiJetVarProducer);
