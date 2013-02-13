/**\class GEMCSCAnalyzer

 Description:

 Analyzer of correlations of signals in CSC & GEM using SimTracks
 Needed for the GEM-CSC triggering algorithm development.

 Original Author:  "Vadim Khotilovich"
 $Id: GEMCSCAnalyzer.cc,v 1.1 2013/02/11 07:34:30 khotilov Exp $
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "RPCGEM/GEMValidation/src/SimTrackMatchManager.h"

#include "TTree.h"

#include <iomanip>
#include <memory>

using namespace std;
using namespace matching;


// "signed" LCT bend pattern
const int LCT_BEND_PATTERN[11] = { -99,  -5,  4, -4,  3, -3,  2, -2,  1, -1,  0};


struct MyTrackDelta
{
  Bool_t odd;
  Char_t charge;
  Char_t chamber;
  Char_t endcap;
  Char_t roll;
  Char_t bend;
  Float_t pt, eta, phi;
  Float_t csc_sh_phi;
  Float_t csc_dg_phi;
  Float_t gem_sh_phi;
  Float_t gem_dg_phi;
  Float_t gem_pad_phi;
  Float_t dphi_sh;
  Float_t dphi_dg;
  Float_t dphi_pad;
  Float_t csc_sh_eta;
  Float_t csc_dg_eta;
  Float_t gem_sh_eta;
  Float_t gem_dg_eta;
  Float_t gem_pad_eta;
  Float_t deta_sh;
  Float_t deta_dg;
  Float_t deta_pad;
  Float_t csc_lct_phi;
  Float_t dphi_lct_pad;
  Float_t csc_lct_eta;
  Float_t deta_lct_pad;
};

struct MySimTrack
{
  Float_t pt, eta, phi;
  Char_t charge;
  Bool_t has_gem_sh;
  Bool_t has_gem_sh_copad;
  Bool_t has_gem_digi;
  Bool_t has_gem_pad;
  Bool_t has_gem_copad;
  Bool_t has_csc_sh_4l;
  Bool_t has_csc_strip_digi_4l;
  Bool_t has_csc_wire_digi_4l;

  Char_t n_layers_gem_simhit;
  Char_t n_layers_csc_simhit;
};


class GEMCSCAnalyzer : public edm::EDAnalyzer
{
public:

  explicit GEMCSCAnalyzer(const edm::ParameterSet&);

  ~GEMCSCAnalyzer() {}
  
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  
  void bookSimTracksTrees();
    
  edm::ParameterSet cfg_;
  std::string simInputLabel_;
  float minPt_;
  int verbose_;

  TTree* tree_trk_;
  TTree* tree_delta_;
  
  MySimTrack  trk_;
  MyTrackDelta trk_d_;
};


GEMCSCAnalyzer::GEMCSCAnalyzer(const edm::ParameterSet& ps)
: cfg_(ps.getParameterSet("simTrackMatching"))
, simInputLabel_(ps.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits"))
, minPt_(ps.getUntrackedParameter<double>("minPt", 4.5))
, verbose_(ps.getUntrackedParameter<int>("verbose", 0))
{
  bookSimTracksTrees();
}



void GEMCSCAnalyzer::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup)
{
  //
}


void GEMCSCAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;
  
  ev.getByLabel(simInputLabel_, sim_tracks);
  ev.getByLabel(simInputLabel_, sim_vertices);
  const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
  
  /*
  {  // print out 1st strip coordinates for rolls in GE1/1 chamber
    edm::ESHandle<GEMGeometry> gem_g;
    es.get<MuonGeometryRecord>().get(gem_g);
    const GEMGeometry * gem_geo = &*gem_g;
    for (int r=1; r<7; ++r)
    {
      GEMDetId p(1, 1, 1, 1, 1, r);
      auto roll = gem_geo->etaPartition(p);
      auto lp = roll->centreOfStrip(1);
      GlobalPoint gp = gem_geo->idToDet(p())->surface().toGlobal(lp);
      cout<<setprecision(9)<<"rollp "<<r<<" "<<gp.phi()<<" "<<gp.perp()<<" "<<roll->localPitch(lp)<<" "<<roll->localPitch(lp)/gp.perp()<<endl;
    }
  }
  */

  /*
  // print out 1st strip coordinates for ME1/b chamber
  edm::ESHandle<CSCGeometry> csc_g;
  es.get<MuonGeometryRecord>().get(csc_g);
  const CSCGeometry * csc_geo = &*csc_g;
  for (int nmb=0;nmb<4;++nmb) for (int la=1; la<7; ++la){
    CSCDetId id(1, 1, 1, 1, la);
    if (nmb==1) id = CSCDetId(1,1,4,1,la);
    if (nmb==2) id = CSCDetId(1,1,2,1,la);
    if (nmb==3) id = CSCDetId(1,2,1,1,la);
    auto strip_topo = csc_geo->layer(id)->geometry()->topology();
    MeasurementPoint mp_top(0.25, 0.5);
    MeasurementPoint mp_bot(0.25, -0.5);
    LocalPoint lp = strip_topo->localPosition(0.25);
    LocalPoint lp_top = strip_topo->localPosition(mp_top);
    LocalPoint lp_bot = strip_topo->localPosition(mp_bot);
    GlobalPoint gp = csc_geo->idToDet(id)->surface().toGlobal(lp);
    GlobalPoint gp_top = csc_geo->idToDet(id)->surface().toGlobal(lp_top);
    GlobalPoint gp_bot = csc_geo->idToDet(id)->surface().toGlobal(lp_bot);
    cout<<id<<endl;
    cout<<setprecision(6)<<"glayer "<<la<<" "<<gp.phi()<<" "<<gp.perp()<<" "<<strip_topo->localPitch(lp)<<" "<<strip_topo->localPitch(lp)/gp.perp()
        <<"  "<<gp_top.phi()<<" "<<gp_top.perp()<<" "<<strip_topo->localPitch(lp_top)<<" "<<strip_topo->localPitch(lp_top)/gp_top.perp()
        <<"  "<<gp_bot.phi()<<" "<<gp_bot.perp()<<" "<<strip_topo->localPitch(lp_bot)<<" "<<strip_topo->localPitch(lp_bot)/gp_bot.perp()
        <<endl;
  }
  */

  int trk_no=0;
  for (auto& t: *sim_tracks.product())
  {
    // SimTrack selection
    if (t.noVertex()) continue;
    if (t.noGenpart()) continue;
    if (std::abs(t.type()) != 13) continue; // only interested in direct muon simtracks
    if (t.momentum().pt() < minPt_) continue;
    float eta = std::abs(t.momentum().eta());
    if (eta > 2.2 || eta < 1.5) continue; // no GEMs could be in such eta

    // match hits and digis to this SimTrack
    SimTrackMatchManager match(&t, &sim_vert[t.vertIndex()], &cfg_, &ev, &es);

    const SimHitMatcher& match_sh = match.simhits();
    const GEMDigiMatcher& match_gd = match.gemDigis();
    const CSCDigiMatcher& match_cd = match.cscDigis();
    const CSCStubMatcher& match_lct = match.cscStubs();

    if (verbose_ > 1) // ---- SimHitMatcher debug printouts
    {
      cout<<"** GEM SimHits **"<<endl;
      cout<<"n_sh_ids "<<match_sh.detIdsGEM().size()<<endl;
      cout<<"n_sh_ids_copad "<<match_sh.detIdsGEMCoincidences().size()<<endl;
      auto gem_sh_sch_ids = match_sh.superChamberIdsGEM();
      cout<<"n_sh_ids_sch "<<gem_sh_sch_ids.size()<<endl;
      cout<<"n_sh_ids_cosch "<<match_sh.superChamberIdsGEMCoincidences().size()<<endl;
      cout<<"n_sh_pad "<<match_sh.nPadsWithHits()<<endl;
      cout<<"n_sh_copad "<<match_sh.nCoincidencePadsWithHits()<<endl;
      for (auto id: gem_sh_sch_ids)
      {
        auto gem_simhits = match_sh.hitsInSuperChamber(id);
        auto gem_simhits_gp = match_sh.simHitsMeanPosition(gem_simhits);
        cout<<"shtrk "<<trk_no<<": "<<eta<<" "<<t.momentum().phi()<<" "<<t.vertIndex()
              <<" | "<<gem_simhits.size()<<" "<<gem_simhits_gp.phi()<<endl;
      }

      int nsch = match_sh.superChamberIdsGEM().size();
      auto gem_sh_ids = match_sh.detIdsGEM();
      for(auto d: gem_sh_ids)
      {
        GEMDetId id(d);
        auto strips = match_sh.hitStripsInDetId(d);
        for(auto s: strips)
        {
          cout<<"sch_strip "<<nsch<<" "<<s<<" "<<id.roll()<<" "<<id.chamber()<<" "<<strips.size()<<endl;
          //if (nsch > 1)cout<<"many_sch_strip "<<s<<" "<<id.roll()<<" "<<id.chamber()<<endl;
          //if (nsch == 1)cout<<"1_sch_strip "<<s<<" "<<id.roll()<<endl;
        }
      }

      cout<<"** CSC SimHits **"<<endl;
      cout<<"n_csh_ids "<<match_sh.detIdsCSC().size()<<endl;
      auto csc_csh_ch_ids = match_sh.chamberIdsCSC();
      cout<<"n_csh_ids_ch "<<csc_csh_ch_ids.size()<<endl;
      cout<<"n_csh_coch "<<match_sh.nCoincidenceCSCChambers()<<endl;
      for (auto id: csc_csh_ch_ids){
        auto csc_simhits = match_sh.hitsInChamber(id);
        auto csc_simhits_gp = match_sh.simHitsMeanPosition(csc_simhits);
        cout<<"cshtrk "<<trk_no<<": "<<eta<<" "<<t.momentum().phi()
            <<" | "<<csc_simhits.size()<<" "<<csc_simhits_gp.phi()<<endl;
      }

      int ncch = match_sh.chamberIdsCSC().size();
      auto csc_sh_ids = match_sh.detIdsCSC();
      for(auto d: csc_sh_ids)
      {
        CSCDetId id(d);
        auto strips = match_sh.hitStripsInDetId(d);
        for(auto s: strips)
        {
          cout<<"cscch_strip "<<ncch<<" "<<s<<" "<<id.chamber()<<" "<<strips.size()<<endl;
        }
      }
    }

    if (verbose_ > 1) // ---- GEMDigiMatcher debug printouts
    {
      cout<<"n_gd_ids "<<match_gd.detIds().size()<<endl;
      cout<<"n_gd_ids_copad "<<match_gd.detIdsWithCoPads().size()<<endl;
      auto gem_gd_sch_ids = match_gd.superChamberIds();
      cout<<"n_gd_ids_sch "<<gem_gd_sch_ids.size()<<endl;
      cout<<"n_gd_ids_cosch "<<match_gd.superChamberIdsWithCoPads().size()<<endl;
      cout<<"n_gd_pad "<<match_gd.nPads()<<endl;
      cout<<"n_gd_copad "<<match_gd.nCoPads()<<endl;
      for (auto id: gem_gd_sch_ids)
      {
        auto gem_digis = match_gd.digisInSuperChamber(id);
        auto gem_digis_gp = match_gd.digisMeanPosition(gem_digis);
        cout<<"gdtrk "<<trk_no<<": "<<eta<<" "<<t.momentum().phi()<<" "<<t.vertIndex()
              <<" | "<<gem_digis.size()<<" "<<gem_digis_gp.phi()<<endl;
      }
    }

    if (verbose_ > 1) // ---- CSCDigiMatcher debug printouts
    {
      cout<<"n_sd_ids "<<match_cd.detIdsStrip().size()<<endl;
      auto csc_sd_ch_ids = match_cd.chamberIdsStrip();
      cout<<"n_sd_ids_ch "<<csc_sd_ch_ids.size()<<endl;
      //cout<<"n_sd_lay "<<cdm.nLayersWithStripInChamber(id)<<endl;
      cout<<"n_sd_coch "<<match_cd.nCoincidenceStripChambers()<<endl;
      for (auto id: csc_sd_ch_ids){
        auto csc_digis = match_cd.stripDigisInChamber(id);
        auto csc_digis_gp = match_cd.digisMeanPosition(csc_digis);
        cout<<"sdtrk "<<trk_no<<": "<<eta<<" "<<t.momentum().phi()
              <<" | "<<csc_digis.size()<<" "<<csc_digis_gp.phi()<<endl;
      }

      cout<<"n_wd_ids "<<match_cd.detIdsWire().size()<<endl;
      auto csc_wd_ch_ids = match_cd.chamberIdsWire();
      cout<<"n_wd_ids_ch "<<csc_wd_ch_ids.size()<<endl;
      //cout<<"n_wd_lay "<<cdm.nLayersWithStripInChamber(id)<<endl;
      cout<<"n_wd_coch "<<match_cd.nCoincidenceWireChambers()<<endl;
    }

    // debug possible mismatch in number of pads from digis and simhits
    if (verbose_ > 0 && match_gd.nPads() != match_sh.nPadsWithHits())
    {
      cout<<"mismatch "<<match_sh.nPadsWithHits()<<" "<<match_gd.nPads()<<endl;
      auto gdids = match_gd.detIds();
      for (auto d: gdids)
      {
        auto pad_ns = match_gd.padNumbersInDetId(d);
        cout<<"gd "<<GEMDetId(d)<<" ";
        copy(pad_ns.begin(), pad_ns.end(), ostream_iterator<int>(cout, " "));
        cout<<endl;
      }
      auto shids = match_sh.detIdsGEM();
      for (auto d: shids)
      {
        auto pad_ns = match_sh.hitPadsInDetId(d);
        cout<<"sh "<<GEMDetId(d)<<" ";
        copy(pad_ns.begin(), pad_ns.end(), ostream_iterator<int>(cout, " "));
        cout<<endl;
      }
    }


    // fill the information for delta-tree
    // only for tracks with enough hit layers in CSC and at least a pad in GEM
    if ( match_gd.nPads() > 0 &&
         match_cd.nCoincidenceStripChambers(4) > 0 &&
         match_cd.nCoincidenceWireChambers(4) > 0 )
    {
      trk_d_.pt = t.momentum().pt();
      trk_d_.phi = t.momentum().phi();
      trk_d_.eta = t.momentum().eta();
      trk_d_.charge = t.charge();

      auto csc_sd_ch_ids = match_cd.chamberIdsStrip();
      auto gem_d_sch_ids = match_gd.superChamberIds();
      if (verbose_) cout<<"will match csc & gem  "<<csc_sd_ch_ids.size()<<" "<<gem_d_sch_ids.size()<<endl;
      for (auto csc_d: csc_sd_ch_ids)
      {
        CSCDetId csc_id(csc_d);

        // require CSC chamber to have at least 4 layers with comparator digis
        if (match_cd.nLayersWithStripInChamber(csc_d) < 4) continue;

        bool is_odd = ( csc_id.chamber() % 2 == 1);
        int region = (csc_id.endcap() == 1) ? 1 : -1;

        auto csc_sh = match_sh.hitsInChamber(csc_d);
        GlobalPoint csc_sh_gp = match_sh.simHitsMeanPosition(csc_sh);

        // CSC trigger strips and wire digis
        auto csc_sd = match_cd.stripDigisInChamber(csc_d);
        auto csc_wd = match_cd.wireDigisInChamber(csc_d);

        GlobalPoint csc_dg_gp = match_cd.digisCSCMedianPosition(csc_sd, csc_wd);

        //GlobalPoint csc_sd_gp = match_cd.digisMeanPosition(csc_sd);
        //cout<<"test csc_dg_gp  "<<csc_sd_gp<<" "<<csc_dg_gp<<" "<<csc_sd_gp.phi() - csc_dg_gp.phi()<<endl;

        if ( std::abs(csc_dg_gp.z()) < 0.001 ) { cout<<"bad csc_dg_gp"<<endl; continue; }

        auto lct_digi = match_lct.lctInChamber(csc_d);
        GlobalPoint csc_lct_gp;
        if (is_valid(lct_digi))
        {
          csc_lct_gp = match_lct.digiPosition(lct_digi);
        }


        // match with signal in GEM in corresponding superchamber
        for(auto gem_d: gem_d_sch_ids)
        {
          GEMDetId gem_id(gem_d);

          // gotta be the same endcap
          if (gem_id.region() != region) continue;
          // gotta be the same chamber#
          if (gem_id.chamber() != csc_id.chamber()) continue;

          auto gem_sh = match_sh.hitsInSuperChamber(gem_d);
          GlobalPoint gem_sh_gp = match_sh.simHitsMeanPosition(gem_sh);

          auto gem_dg = match_gd.digisInSuperChamber(gem_d);
          //GlobalPoint gem_dg_gp = match_gd.digisMeanPosition(gem_dg);
          auto gem_dg_and_gp = match_gd.digiInGEMClosestToCSC(gem_dg, csc_dg_gp);
          //auto best_gem_dg = gem_dg_and_gp.first;
          GlobalPoint gem_dg_gp = gem_dg_and_gp.second;

          auto gem_pads = match_gd.padsInSuperChamber(gem_d);
          //GlobalPoint gem_pads_gp = match_gd.digisMeanPosition(gem_pads);
          auto gem_pad_and_gp = match_gd.digiInGEMClosestToCSC(gem_pads, csc_dg_gp);
          auto best_gem_pad = gem_pad_and_gp.first;
          GlobalPoint gem_pad_gp = gem_pad_and_gp.second;

          if (gem_sh.size() == 0 || gem_dg.size() == 0 || gem_pads.size() == 0) continue;

          /*
          float avg_roll = 0.;
          for (auto& d: gem_pads )
          {
            GEMDetId id(digi_id(d));
            avg_roll += id.roll();
          }
          avg_roll = avg_roll/gem_pads.size();
          */
          GEMDetId id_of_best_gem(digi_id(best_gem_pad));

          trk_d_.odd = is_odd;
          trk_d_.chamber = csc_id.chamber();
          trk_d_.endcap = csc_id.endcap();
          trk_d_.roll = id_of_best_gem.roll();
          trk_d_.csc_sh_phi = csc_sh_gp.phi();
          trk_d_.csc_dg_phi = csc_dg_gp.phi();
          trk_d_.gem_sh_phi = gem_sh_gp.phi();
          trk_d_.gem_dg_phi = gem_dg_gp.phi();
          trk_d_.gem_pad_phi = gem_pad_gp.phi();
          trk_d_.dphi_sh = deltaPhi(csc_sh_gp.phi(), gem_sh_gp.phi());
          trk_d_.dphi_dg = deltaPhi(csc_dg_gp.phi(), gem_dg_gp.phi());
          trk_d_.dphi_pad = deltaPhi(csc_dg_gp.phi(), gem_pad_gp.phi());
          trk_d_.csc_sh_eta = csc_sh_gp.eta();
          trk_d_.csc_dg_eta = csc_dg_gp.eta();
          trk_d_.gem_sh_eta = gem_sh_gp.eta();
          trk_d_.gem_dg_eta = gem_dg_gp.eta();
          trk_d_.gem_pad_eta = gem_pad_gp.eta();
          trk_d_.deta_sh = csc_sh_gp.eta() - gem_sh_gp.eta();
          trk_d_.deta_dg = csc_dg_gp.eta() - gem_dg_gp.eta();
          trk_d_.deta_pad = csc_dg_gp.eta() - gem_pad_gp.eta();
          trk_d_.bend = -99;
          trk_d_.csc_lct_phi = -99.;
          trk_d_.dphi_lct_pad = -99.;
          trk_d_.csc_lct_eta = -99.;
          trk_d_.deta_lct_pad = -99.;
          if (std::abs(csc_lct_gp.z()) > 0.001)
          {
            trk_d_.bend = LCT_BEND_PATTERN[digi_pattern(lct_digi)];
            trk_d_.csc_lct_phi = csc_lct_gp.phi();
            trk_d_.dphi_lct_pad = deltaPhi(csc_lct_gp.phi(), gem_pad_gp.phi());
            trk_d_.csc_lct_eta = csc_lct_gp.eta();
            trk_d_.deta_lct_pad = csc_lct_gp.eta() - gem_pad_gp.eta();
          }

          tree_delta_->Fill();

          /*
          if (csc_id.endcap()==1)
          {
            auto best_gem_dg = gem_dg_and_gp.first;
            GEMDetId id_of_best_dg(digi_id(best_gem_dg));
            cout<<"funny_deta "<<gem_dg_gp.eta() - gem_pad_gp.eta()<<" "
                <<digi_channel(best_gem_pad)<<" "<<digi_channel(best_gem_dg)<<" "
                <<id_of_best_gem.roll()<<" "<<id_of_best_dg.roll()<<" "
                <<id_of_best_gem.layer()<<" "<<id_of_best_dg.layer()<<" "
                <<match_gd.nLayersWithDigisInSuperChamber(gem_d)<<endl;
          }*/

          if (verbose_ > 1) // debug printout for the stuff in delta-tree
          {
            cout<<"got match "<<csc_id<<"  "<<gem_id<<endl;
            cout<<"matchdphis "<<is_odd<<" "<<csc_id.chamber()<<" "
                <<csc_sh_gp.phi()<<" "<<csc_dg_gp.phi()<<" "<<gem_sh_gp.phi()<<" "<<gem_dg_gp.phi()<<" "<<gem_pad_gp.phi()<<" "
                <<trk_d_.dphi_sh<<" "<<trk_d_.dphi_dg<<" "<<trk_d_.dphi_pad<<"   "
                <<csc_sh_gp.eta()<<" "<<csc_dg_gp.eta()<<" "<<gem_sh_gp.eta()<<" "<<gem_dg_gp.eta()<<" "<<gem_pad_gp.eta()<<" "
                <<trk_d_.deta_sh<<" "<<trk_d_.deta_dg<<" "<<trk_d_.deta_pad<<endl;
          }
        }

      }
    }

    trk_no++;
  }
}


void GEMCSCAnalyzer::bookSimTracksTrees()
{
  edm::Service< TFileService > fs;

  tree_delta_ = fs->make<TTree>("trk_delta", "trk_delta");
  tree_delta_->Branch("odd", &trk_d_.odd);
  tree_delta_->Branch("charge", &trk_d_.charge);
  tree_delta_->Branch("chamber", &trk_d_.chamber);
  tree_delta_->Branch("endcap", &trk_d_.endcap);
  tree_delta_->Branch("roll", &trk_d_.roll);
  tree_delta_->Branch("bend", &trk_d_.bend);
  tree_delta_->Branch("pt", &trk_d_.pt);
  tree_delta_->Branch("eta", &trk_d_.eta);
  tree_delta_->Branch("phi", &trk_d_.phi);
  tree_delta_->Branch("csc_sh_phi", &trk_d_.csc_sh_phi);
  tree_delta_->Branch("csc_dg_phi", &trk_d_.csc_dg_phi);
  tree_delta_->Branch("gem_sh_phi", &trk_d_.gem_sh_phi);
  tree_delta_->Branch("gem_dg_phi", &trk_d_.gem_dg_phi);
  tree_delta_->Branch("gem_pad_phi", &trk_d_.gem_pad_phi);
  tree_delta_->Branch("dphi_sh", &trk_d_.dphi_sh);
  tree_delta_->Branch("dphi_dg", &trk_d_.dphi_dg);
  tree_delta_->Branch("dphi_pad", &trk_d_.dphi_pad);
  tree_delta_->Branch("csc_sh_eta", &trk_d_.csc_sh_eta);
  tree_delta_->Branch("csc_dg_eta", &trk_d_.csc_dg_eta);
  tree_delta_->Branch("gem_sh_eta", &trk_d_.gem_sh_eta);
  tree_delta_->Branch("gem_dg_eta", &trk_d_.gem_dg_eta);
  tree_delta_->Branch("gem_pad_eta", &trk_d_.gem_pad_eta);
  tree_delta_->Branch("deta_sh", &trk_d_.deta_sh);
  tree_delta_->Branch("deta_dg", &trk_d_.deta_dg);
  tree_delta_->Branch("deta_pad", &trk_d_.deta_pad);
  tree_delta_->Branch("csc_lct_phi", &trk_d_.csc_lct_phi);
  tree_delta_->Branch("dphi_lct_pad", &trk_d_.dphi_lct_pad);
  tree_delta_->Branch("csc_lct_eta", &trk_d_.csc_lct_eta);
  tree_delta_->Branch("deta_lct_pad", &trk_d_.deta_lct_pad);
  //tree_delta_->Branch("", &trk_d_.);

  /*
  tree_trk_ = fs->make< TTree >("trk", "trk");
  tree_trk_->Branch("charge", &trk_.charge);
  */
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GEMCSCAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(GEMCSCAnalyzer);
