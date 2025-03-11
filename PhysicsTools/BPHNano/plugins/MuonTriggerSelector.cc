/////////////////////////////// MuonTriggerSelector //////////////////////////
/// original authors: G Karathanasis (CERN),  G Melachroinos (NKUA)
// filters muons and produces 3 collections: all muons that pass the selection
// triggering muons and transient tracks of triggering muons

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "DataFormats/PatCandidates/interface/TriggerPath.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/PatCandidates/interface/TriggerAlgorithm.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "helper.h"

using namespace std;

constexpr bool debug = false;

class MuonTriggerSelector : public edm::stream::EDProducer <> {

public:

  explicit MuonTriggerSelector(const edm::ParameterSet &iConfig);
  ~MuonTriggerSelector() override {};

private:

  virtual void produce(edm::Event&, const edm::EventSetup&);
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bFieldToken_;
  edm::EDGetTokenT<std::vector<pat::Muon>> muonSrc_;
  edm::EDGetTokenT<edm::TriggerResults> triggerBits_;
  edm::EDGetTokenT<std::vector<pat::TriggerObjectStandAlone>> triggerObjects_;
  edm::EDGetTokenT<pat::PackedTriggerPrescales> triggerPrescales_;
  const StringCutObjectSelector<pat::Muon> muon_selection_;
  // for trigger match
  const double maxdR_;
  // triggers
  std::vector<std::string> HLTPaths_;

};

MuonTriggerSelector::MuonTriggerSelector(const edm::ParameterSet &iConfig):
  bFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
  muonSrc_( consumes<std::vector<pat::Muon>> ( iConfig.getParameter<edm::InputTag>( "muonCollection" ) ) ),
  triggerBits_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("bits"))),
  triggerObjects_(consumes<std::vector<pat::TriggerObjectStandAlone>>(iConfig.getParameter<edm::InputTag>("objects"))),
  triggerPrescales_(consumes<pat::PackedTriggerPrescales>(iConfig.getParameter<edm::InputTag>("prescales"))),
  muon_selection_{iConfig.getParameter<std::string>("muonSelection")},
  maxdR_(iConfig.getParameter<double>("maxdR_matching")),
  HLTPaths_(iConfig.getParameter<std::vector<std::string>>("HLTPaths"))// multiple paths with comma
{
  // outputs
  produces<pat::MuonCollection>("AllMuons");
  produces<pat::MuonCollection>("SelectedMuons");
  produces<TransientTrackCollection>("SelectedTransientMuons");
}

void MuonTriggerSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  const auto& bField = iSetup.getData(bFieldToken_);

  edm::Handle<edm::TriggerResults> triggerBits;
  iEvent.getByToken(triggerBits_, triggerBits);

  std::vector<pat::TriggerObjectStandAlone> triggeringMuons;

  //taken from https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookMiniAOD2016#Trigger
  edm::Handle<std::vector<pat::TriggerObjectStandAlone>> triggerObjects;
  iEvent.getByToken(triggerObjects_, triggerObjects);

  std::unique_ptr<pat::MuonCollection>      allmuons_out   ( new pat::MuonCollection );
  std::unique_ptr<pat::MuonCollection>      muons_out      ( new pat::MuonCollection );
  std::unique_ptr<TransientTrackCollection> trans_muons_out( new TransientTrackCollection );

  edm::Handle<std::vector<pat::Muon>> muons;
  iEvent.getByToken(muonSrc_, muons);

  // check for reco muons matched to triggering muons
  std::vector<int> muonIsTrigger(muons->size(), 0);
  std::vector<float> muonDR(muons->size(), -1.);
  std::vector<float> muonDPT(muons->size(), 10000.);
  std::vector<int> loose_id(muons->size(), 0);

  std::vector<int> matched_reco_flag(muons->size(), -1);
  std::vector<int> matched_trg_index(muons->size(), -1);
  std::vector<float> matched_dr(muons->size(), -1.);
  std::vector<float> matched_dpt(muons->size(), -10000.);
  std::vector<std::vector<int>> fires;
  std::vector<std::vector<float>> matcher;
  std::vector<std::vector<float>> DR;
  std::vector<std::vector<float>> DPT;


  for (const pat::Muon &muon : *muons) {
    if (debug)std::cout << "Muon Pt=" << muon.pt() << " Eta=" << muon.eta() << " Phi=" << muon.phi()  << endl;


    std::vector<int> frs(HLTPaths_.size(), 0); //path fires for each reco muon
    std::vector<float> temp_matched_to(HLTPaths_.size(), 1000.);
    std::vector<float> temp_DR(HLTPaths_.size(), 1000.);
    std::vector<float> temp_DPT(HLTPaths_.size(), 1000.);
    int ipath = -1;

    for (const std::string &path : HLTPaths_) {
      ipath++;
      // the following vectors are used in order to find the minimum DR between a reco muon and all the HLT objects that is matched with it so as a reco muon will be matched with only one HLT object every time so as there is a one-to-one correspondance between the two collection. DPt_rel is not used to create this one-to-one correspondance but only to create a few plots, debugging and be sure thateverything is working fine.
      std::vector<float> temp_dr(muon.triggerObjectMatches().size(), 1000.);
      std::vector<float> temp_dpt(muon.triggerObjectMatches().size(), 1000.);
      std::vector<float> temp_pt(muon.triggerObjectMatches().size(), 1000.);
      char cstr[ (path + "*").size() + 1];
      strcpy( cstr, (path + "*").c_str() );
      //Here we find all the HLT objects from each HLT path each time that are matched with the reco muon.
      if (muon.triggerObjectMatches().size() != 0) {
        for (size_t i = 0; i < muon.triggerObjectMatches().size(); i++) {
          if (muon.triggerObjectMatch(i) != 0 && muon.triggerObjectMatch(i)->hasPathName(cstr, true, true)) {
            frs[ipath] = 1;
            float dr=deltaR(muon.eta(),muon.phi(),muon.triggerObjectMatch(i)->eta(),muon.triggerObjectMatch(i)->phi());
	    float dpt = (muon.triggerObjectMatch(i)->pt() - muon.pt()) / muon.triggerObjectMatch(i)->pt();
            temp_dr[i] = dr;
            temp_dpt[i] = dpt;
            temp_pt[i] = muon.triggerObjectMatch(i)->pt();
            if (debug)std::cout << "Path=" << cstr << endl;
            if (debug)std::cout << "HLT  Pt=" << muon.triggerObjectMatch(i)->pt() << " Eta=" << muon.triggerObjectMatch(i)->eta() << " Phi=" << muon.triggerObjectMatch(i)->phi() << endl;
            if (debug)std::cout << "Muon Pt=" << muon.pt() << " Eta=" << muon.eta() << " Phi=" << muon.phi()  << endl;
            if (debug)std::cout << "DR = " << temp_dr[i] << endl;
          }
        }
        // and now we find the real minimum between the reco muon and all its matched HLT objects.
        temp_DR[ipath] = *min_element(temp_dr.begin(), temp_dr.end());
        int position = std::min_element(temp_dr.begin(), temp_dr.end()) - temp_dr.begin();
        temp_DPT[ipath] = temp_dpt[position];
        temp_matched_to[ipath] = temp_pt[position];
      }
    }
    //and now since we have found the minimum DR we save a few variables for plots
    fires.push_back(frs);//This is used in order to see if a reco muon fired a Trigger (1) or not (0).
    matcher.push_back(temp_matched_to); //This is used in order to see if a reco muon is matched with a HLT object. PT of the reco muon is saved in this vector.
    DR.push_back(temp_DR);
    DPT.push_back(temp_DPT);

  }
  //now, check for different reco muons that are matched to the same HLTObject.
  for (unsigned int path = 0; path < HLTPaths_.size(); path++) {
    for (unsigned int iMuo = 0; iMuo < muons->size(); iMuo++) {
      for (unsigned int im = (iMuo + 1); im < muons->size(); im++) {
        if (matcher[iMuo][path] != 1000. && matcher[iMuo][path] == matcher[im][path]) {
          if (DR[iMuo][path] < DR[im][path]) { //Keep the one that has the minimum DR with the HLT object
            fires[im][path] = 0;
            matcher[im][path] = 1000.;
            DR[im][path] = 1000.;
            DPT[im][path] = 1000.;
          }
          else {
            fires[iMuo][path] = 0;
            matcher[iMuo][path] = 1000.;
            DR[iMuo][path] = 1000.;
            DPT[iMuo][path] = 1000.;
          }
        }
      }
      if (matcher[iMuo][path] != 1000.) {
        muonIsTrigger[iMuo] = 1;
        muonDR[iMuo] = DR[iMuo][path];
        muonDPT[iMuo] = DPT[iMuo][path];
      }
    }
  }



  //save the reco muon in both collections
  for (const pat::Muon & muon : *muons) {
    unsigned int iMuo(&muon - & (muons->at(0)) );
    if ( !muon_selection_(muon) ) continue; // selection cuts
    const reco::TransientTrack muonTT((*(muon.bestTrack())), &bField);
    if (!muonTT.isValid()) continue;

    allmuons_out->emplace_back(muon);
    if (muonIsTrigger[iMuo] != 1) continue;


    muons_out->emplace_back(muon);
    muons_out->back().addUserInt("isTriggering", muonIsTrigger[iMuo]);
    muons_out->back().addUserFloat("trgDR", muonDR[iMuo]);
    muons_out->back().addUserFloat("trgDPT", muonDPT[iMuo]);
    muons_out->back().addUserInt("looseId", loose_id[iMuo]);

    for (unsigned int i = 0; i < HLTPaths_.size(); i++)
      muons_out->back().addUserInt(HLTPaths_[i], fires[iMuo][i]);

    trans_muons_out->emplace_back(muonTT);
  }

  iEvent.put(std::move(allmuons_out),    "AllMuons");
  iEvent.put(std::move(muons_out),       "SelectedMuons");
  iEvent.put(std::move(trans_muons_out), "SelectedTransientMuons");
}

DEFINE_FWK_MODULE(MuonTriggerSelector);
