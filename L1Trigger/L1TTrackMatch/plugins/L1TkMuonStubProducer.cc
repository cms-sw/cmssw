// input: L1TkTracks and Muon Stubs 
//
// 
//
// Author: Vladimir Rekovic
// Version: 1   Date: 10.Feb.2019
// Descritiption: 
//      Match L1Tracks to EndCap muon stubs.
//      Stubs are taken to be inputs to EMTF, EMTFHitCollection
//      Use DynamicWindow matching algorithm which assumes muon 
//      detector object (here Stubs) have coordinates at 2nd station.  
//      Match the two and produce a collection of L1TkMuonParticle

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkMuCorrDynamicWindows.h"
#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

// system include files
#include <memory>
#include <string>

using namespace l1t;

class L1TkMuonStubProducer : public edm::EDProducer {
public:

  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;

  struct PropState { //something simple, imagine it's hardware emulation
    PropState() :
      pt(-99),  eta(-99), phi(-99),
      sigmaPt(-99),  sigmaEta(-99), sigmaPhi(-99),
      valid(false) {}
    float pt;
    float eta;
    float phi;
    float sigmaPt;
    float sigmaEta;
    float sigmaPhi;
    bool valid;

  };

  enum AlgoType {
    kTP = 1,
    kDynamicWindows = 2
  };

  explicit L1TkMuonStubProducer(const edm::ParameterSet&);
  ~L1TkMuonStubProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  // algo for endcap regions using dynamic windows for making the match trk + muStub
  void runOnMuonHitCollection(const edm::Handle<EMTFHitCollection>&,
                          const edm::Handle<L1TTTrackCollectionType>&,
                          L1TkMuonParticleCollection& tkMuons) const;

  // int emtfMatchAlgoVersion_ ;         
  AlgoType emtfMatchAlgoVersion_ ;         

  std::unique_ptr<L1TkMuCorrDynamicWindows> dwcorr_;
  bool requireBX0_;

  const edm::EDGetTokenT< EMTFTrackCollection >          emtfTCToken; // the track collection, directly from the EMTF and not formatted by GT
  const edm::EDGetTokenT< EMTFHitCollection >            emtfHCToken; // the hit collection, directly from the EMTF which stored the input Hits
  const edm::EDGetTokenT< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
} ;


L1TkMuonStubProducer::L1TkMuonStubProducer(const edm::ParameterSet& iConfig) :
  emtfTCToken(consumes< EMTFTrackCollection >         (iConfig.getParameter<edm::InputTag>("L1EMTFTrackCollectionInputTag"))),
  emtfHCToken(consumes< EMTFHitCollection >           (iConfig.getParameter<edm::InputTag>("L1EMTFHitCollectionInputTag"))),
  trackToken (consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
{
   
   // configuration of the EMTF algorithm type
   std::string emtfMatchAlgoVersionString = iConfig.getParameter<std::string>("emtfMatchAlgoVersion");
   std::transform(emtfMatchAlgoVersionString.begin(), emtfMatchAlgoVersionString.end(), emtfMatchAlgoVersionString.begin(), ::tolower); // make lowercase

   requireBX0_ = iConfig.getParameter<bool>("require_BX0");

   if (emtfMatchAlgoVersionString == "dynamicwindows")
      emtfMatchAlgoVersion_ = kDynamicWindows;
   else
    throw cms::Exception("TkMuAlgoConfig") << "the ID of the EMTF algo matcher passed is invalid\n";
   

   produces<L1TkMuonParticleCollection>();

   // initializations
   if (emtfMatchAlgoVersion_ == kDynamicWindows)
   {
      // FIXME: to merge eventually into an unique file with bith phi and theta boundaries
      std::string fIn_bounds_name = iConfig.getParameter<edm::FileInPath>("emtfcorr_boundaries").fullPath();
      std::string fIn_theta_name  = iConfig.getParameter<edm::FileInPath>("emtfcorr_theta_windows").fullPath();
      std::string fIn_phi_name    = iConfig.getParameter<edm::FileInPath>("emtfcorr_phi_windows").fullPath();
      auto bounds = L1TkMuCorrDynamicWindows::prepare_corr_bounds(fIn_bounds_name.c_str(), "h_dphi_l");
      TFile* fIn_theta = TFile::Open (fIn_theta_name.c_str());
      TFile* fIn_phi   = TFile::Open (fIn_phi_name.c_str());
      dwcorr_ = std::unique_ptr<L1TkMuCorrDynamicWindows> (new L1TkMuCorrDynamicWindows(bounds, fIn_theta, fIn_phi));

      // files can be closed since the correlator code clones the TF1s
      fIn_theta->Close();
      fIn_phi->Close();

      // FIXME: more initialisation using the parameters passed from the cfg
      dwcorr_->set_safety_factor  (iConfig.getParameter<double>("final_window_factor"));
      dwcorr_->set_sf_initialrelax(iConfig.getParameter<double>("initial_window_factor"));
      
      dwcorr_->set_relaxation_pattern(
        iConfig.getParameter<double>("pt_start_relax"),
        iConfig.getParameter<double>("pt_end_relax")
        );
      
      dwcorr_->set_do_relax_factor(iConfig.getParameter<bool>("do_relax_factors"));

      //
      dwcorr_ -> set_n_trk_par       (iConfig.getParameter<int>("n_trk_par"));
      dwcorr_ -> set_min_trk_p       (iConfig.getParameter<double>("min_trk_p"));
      dwcorr_ -> set_max_trk_aeta    (iConfig.getParameter<double>("max_trk_aeta"));
      dwcorr_ -> set_max_trk_chi2    (iConfig.getParameter<double>("max_trk_chi2"));
      dwcorr_ -> set_min_trk_nstubs  (iConfig.getParameter<int>("min_trk_nstubs"));
      dwcorr_ -> set_do_trk_qual_presel(true);
   }
}

L1TkMuonStubProducer::~L1TkMuonStubProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkMuonStubProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // the L1Mu objects
  edm::Handle<EMTFTrackCollection> l1emtfTCH;
  edm::Handle<EMTFHitCollection> l1emtfHCH;

  iEvent.getByToken(emtfTCToken, l1emtfTCH);
  iEvent.getByToken(emtfHCToken, l1emtfHCH);

  // the L1Tracks
  edm::Handle<L1TTTrackCollectionType> l1tksH;
  iEvent.getByToken(trackToken, l1tksH);

  L1TkMuonParticleCollection oc_endcap_tkmuonStub;

  // process each of the MTF collections separately! -- we don't want to filter the muons
  //if (emtfMatchAlgoVersion_ == kDynamicWindows) 
    runOnMuonHitCollection(l1emtfHCH, l1tksH, oc_endcap_tkmuonStub);
  //else
    //throw cms::Exception("TkMuAlgoConfig") << "trying to run an invalid algorithm (this should never happen)\n";

  // now combine all trk muons into a single output collection!
  std::unique_ptr<L1TkMuonParticleCollection> oc_tkmuon(new L1TkMuonParticleCollection());
  for (const auto& p : {oc_endcap_tkmuonStub}){
    oc_tkmuon->insert(oc_tkmuon->end(), p.begin(), p.end());
  }

  // put the new track+muon objects in the event!
  iEvent.put( std::move(oc_tkmuon));
};


void
L1TkMuonStubProducer::runOnMuonHitCollection(const edm::Handle<EMTFHitCollection>& muonStubH,
                                     const edm::Handle<L1TTTrackCollectionType>& l1tksH,
                                     L1TkMuonParticleCollection& tkMuons) const

{
  const EMTFHitCollection& l1muStubs = (*muonStubH.product());
  const L1TTTrackCollectionType& l1trks = (*l1tksH.product());
  const int myStation = 2;
  auto corr_muStub_idxs = dwcorr_->find_match_stub(l1muStubs, l1trks, myStation, requireBX0_);
  // it's a vector with as many entries as the L1TT vector.
  // >= 0 : the idx in the muStub vector of matched muStubs
  // < 0: no match
  
  // sanity check
  if (corr_muStub_idxs.size() != l1trks.size())
    throw cms::Exception("TkMuAlgoOutput") << "the size of tkmu indices does not match the size of input trk collection\n";
  
  for (uint il1ttrack = 0; il1ttrack < corr_muStub_idxs.size(); ++il1ttrack)
  {
    int muStub_idx = corr_muStub_idxs.at(il1ttrack);
    if (muStub_idx < 0)
      continue;

    const L1TTTrackType& matchTk = l1trks[il1ttrack];
    const auto& p3 = matchTk.getMomentum(dwcorr_->get_n_trk_par());
    const auto& tkv3 = matchTk.getPOCA(dwcorr_->get_n_trk_par());
    float p4e = sqrt(0.105658369*0.105658369 + p3.mag2() );
    math::XYZTLorentzVector l1tkp4(p3.x(), p3.y(), p3.z(), p4e);

    edm::Ref< RegionalMuonCandBxCollection > l1muRef; // FIXME! The reference to the muon is null 
    edm::Ptr< L1TTTrackType > l1tkPtr(l1tksH, il1ttrack);
    float trkisol = -999; // FIXME: now doing as in the TP algo
    L1TkMuonParticle l1tkmu(l1tkp4, l1muRef, l1tkPtr, trkisol);
    l1tkmu.setTrkzVtx( (float)tkv3.z() );
    
    tkMuons.push_back(l1tkmu);
  }


  return;
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkMuonStubProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


//define this as a plug-in
DEFINE_FWK_MODULE(L1TkMuonStubProducer);



