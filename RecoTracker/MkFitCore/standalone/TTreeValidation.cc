#include "TTreeValidation.h"
#include "Event.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

#ifndef NO_ROOT

namespace mkfit {

  TTreeValidation::TTreeValidation(std::string fileName, const TrackerInfo* trk_info) {
    std::lock_guard<std::mutex> locker(glock_);
    gROOT->ProcessLine("#include <vector>");

    ntotallayers_fit_ = trk_info->n_layers();

    // KPM via DSR's ROOT wizardry: ROOT's context management implicitly assumes that a file is opened and
    // closed on the same thread. To avoid the problem, we declare a local
    // TContext object; when it goes out of scope, its destructor unregisters
    // the context, guaranteeing the context is unregistered in the same thread
    // it was registered in. (do this for tfiles and trees
    TDirectory::TContext contextEraser;
    f_ = std::unique_ptr<TFile>(TFile::Open(fileName.c_str(), "recreate"));

    if (Config::sim_val_for_cmssw || Config::sim_val) {
      TTreeValidation::initializeEfficiencyTree();
      TTreeValidation::initializeFakeRateTree();
    }
    if (Config::cmssw_val) {
      TTreeValidation::initializeCMSSWEfficiencyTree();
      TTreeValidation::initializeCMSSWFakeRateTree();
    }
    if (Config::fit_val) {
      for (int i = 0; i < nfvs_; ++i)
        fvs_[i].resize(ntotallayers_fit_);
      TTreeValidation::initializeFitTree();
    }
    TTreeValidation::initializeConfigTree();
  }

  void TTreeValidation::initializeEfficiencyTree() {
    // efficiency validation
    efftree_ = std::make_unique<TTree>("efftree", "efftree");
    efftree_->SetDirectory(0);

    efftree_->Branch("evtID", &evtID_eff_);
    efftree_->Branch("mcID", &mcID_eff_);

    efftree_->Branch("nHits_mc", &nHits_mc_eff_);
    efftree_->Branch("nLayers_mc", &nLayers_mc_eff_);
    efftree_->Branch("lastlyr_mc", &lastlyr_mc_eff_);

    efftree_->Branch("seedID_seed", &seedID_seed_eff_);
    efftree_->Branch("seedID_build", &seedID_build_eff_);
    efftree_->Branch("seedID_fit", &seedID_fit_eff_);

    efftree_->Branch("x_mc_gen", &x_mc_gen_eff_);
    efftree_->Branch("y_mc_gen", &y_mc_gen_eff_);
    efftree_->Branch("z_mc_gen", &z_mc_gen_eff_);

    efftree_->Branch("pt_mc_gen", &pt_mc_gen_eff_);
    efftree_->Branch("phi_mc_gen", &phi_mc_gen_eff_);
    efftree_->Branch("eta_mc_gen", &eta_mc_gen_eff_);

    efftree_->Branch("mcmask_seed", &mcmask_seed_eff_);
    efftree_->Branch("mcmask_build", &mcmask_build_eff_);
    efftree_->Branch("mcmask_fit", &mcmask_fit_eff_);

    efftree_->Branch("mcTSmask_seed", &mcTSmask_seed_eff_);
    efftree_->Branch("mcTSmask_build", &mcTSmask_build_eff_);
    efftree_->Branch("mcTSmask_fit", &mcTSmask_fit_eff_);

    efftree_->Branch("xhit_seed", &xhit_seed_eff_);
    efftree_->Branch("xhit_build", &xhit_build_eff_);
    efftree_->Branch("xhit_fit", &xhit_fit_eff_);

    efftree_->Branch("yhit_seed", &yhit_seed_eff_);
    efftree_->Branch("yhit_build", &yhit_build_eff_);
    efftree_->Branch("yhit_fit", &yhit_fit_eff_);

    efftree_->Branch("zhit_seed", &zhit_seed_eff_);
    efftree_->Branch("zhit_build", &zhit_build_eff_);
    efftree_->Branch("zhit_fit", &zhit_fit_eff_);

    efftree_->Branch("pt_mc_seed", &pt_mc_seed_eff_);
    efftree_->Branch("pt_seed", &pt_seed_eff_);
    efftree_->Branch("ept_seed", &ept_seed_eff_);
    efftree_->Branch("pt_mc_build", &pt_mc_build_eff_);
    efftree_->Branch("pt_build", &pt_build_eff_);
    efftree_->Branch("ept_build", &ept_build_eff_);
    efftree_->Branch("pt_mc_fit", &pt_mc_fit_eff_);
    efftree_->Branch("pt_fit", &pt_fit_eff_);
    efftree_->Branch("ept_fit", &ept_fit_eff_);

    efftree_->Branch("phi_mc_seed", &phi_mc_seed_eff_);
    efftree_->Branch("phi_seed", &phi_seed_eff_);
    efftree_->Branch("ephi_seed", &ephi_seed_eff_);
    efftree_->Branch("phi_mc_build", &phi_mc_build_eff_);
    efftree_->Branch("phi_build", &phi_build_eff_);
    efftree_->Branch("ephi_build", &ephi_build_eff_);
    efftree_->Branch("phi_mc_fit", &phi_mc_fit_eff_);
    efftree_->Branch("phi_fit", &phi_fit_eff_);
    efftree_->Branch("ephi_fit", &ephi_fit_eff_);

    efftree_->Branch("eta_mc_seed", &eta_mc_seed_eff_);
    efftree_->Branch("eta_seed", &eta_seed_eff_);
    efftree_->Branch("eeta_seed", &eeta_seed_eff_);
    efftree_->Branch("eta_mc_build", &eta_mc_build_eff_);
    efftree_->Branch("eta_build", &eta_build_eff_);
    efftree_->Branch("eeta_build", &eeta_build_eff_);
    efftree_->Branch("eta_mc_fit", &eta_mc_fit_eff_);
    efftree_->Branch("eta_fit", &eta_fit_eff_);
    efftree_->Branch("eeta_fit", &eeta_fit_eff_);

    efftree_->Branch("nHits_seed", &nHits_seed_eff_);
    efftree_->Branch("nHits_build", &nHits_build_eff_);
    efftree_->Branch("nHits_fit", &nHits_fit_eff_);

    efftree_->Branch("nLayers_seed", &nLayers_seed_eff_);
    efftree_->Branch("nLayers_build", &nLayers_build_eff_);
    efftree_->Branch("nLayers_fit", &nLayers_fit_eff_);

    efftree_->Branch("nHitsMatched_seed", &nHitsMatched_seed_eff_);
    efftree_->Branch("nHitsMatched_build", &nHitsMatched_build_eff_);
    efftree_->Branch("nHitsMatched_fit", &nHitsMatched_fit_eff_);

    efftree_->Branch("fracHitsMatched_seed", &fracHitsMatched_seed_eff_);
    efftree_->Branch("fracHitsMatched_build", &fracHitsMatched_build_eff_);
    efftree_->Branch("fracHitsMatched_fit", &fracHitsMatched_fit_eff_);

    efftree_->Branch("lastlyr_seed", &lastlyr_seed_eff_);
    efftree_->Branch("lastlyr_build", &lastlyr_build_eff_);
    efftree_->Branch("lastlyr_fit", &lastlyr_fit_eff_);

    efftree_->Branch("dphi_seed", &dphi_seed_eff_);
    efftree_->Branch("dphi_build", &dphi_build_eff_);
    efftree_->Branch("dphi_fit", &dphi_fit_eff_);

    efftree_->Branch("hitchi2_seed", &hitchi2_seed_eff_);
    efftree_->Branch("hitchi2_build", &hitchi2_build_eff_);
    efftree_->Branch("hitchi2_fit", &hitchi2_fit_eff_);

    efftree_->Branch("score_seed", &score_seed_eff_);
    efftree_->Branch("score_build", &score_build_eff_);
    efftree_->Branch("score_fit", &score_fit_eff_);

    efftree_->Branch("helixchi2_seed", &helixchi2_seed_eff_);
    efftree_->Branch("helixchi2_build", &helixchi2_build_eff_);
    efftree_->Branch("helixchi2_fit", &helixchi2_fit_eff_);

    efftree_->Branch("duplmask_seed", &duplmask_seed_eff_);
    efftree_->Branch("duplmask_build", &duplmask_build_eff_);
    efftree_->Branch("duplmask_fit", &duplmask_fit_eff_);

    efftree_->Branch("nTkMatches_seed", &nTkMatches_seed_eff_);
    efftree_->Branch("nTkMatches_build", &nTkMatches_build_eff_);
    efftree_->Branch("nTkMatches_fit", &nTkMatches_fit_eff_);

    efftree_->Branch("itermask_seed", &itermask_seed_eff_);
    efftree_->Branch("itermask_build", &itermask_build_eff_);
    efftree_->Branch("itermask_fit", &itermask_fit_eff_);
    efftree_->Branch("iterduplmask_seed", &iterduplmask_seed_eff_);
    efftree_->Branch("iterduplmask_build", &iterduplmask_build_eff_);
    efftree_->Branch("iterduplmask_fit", &iterduplmask_fit_eff_);
    efftree_->Branch("algo_seed", &algo_seed_eff_);

    if (Config::keepHitInfo) {
      efftree_->Branch("hitlyrs_mc", &hitlyrs_mc_eff_);
      efftree_->Branch("hitlyrs_seed", &hitlyrs_seed_eff_);
      efftree_->Branch("hitlyrs_build", &hitlyrs_build_eff_);
      efftree_->Branch("hitlyrs_fit", &hitlyrs_fit_eff_);

      efftree_->Branch("hitidxs_mc", &hitidxs_mc_eff_);
      efftree_->Branch("hitidxs_seed", &hitidxs_seed_eff_);
      efftree_->Branch("hitidxs_build", &hitidxs_build_eff_);
      efftree_->Branch("hitidxs_fit", &hitidxs_fit_eff_);

      efftree_->Branch("hitmcTkIDs_mc", &hitmcTkIDs_mc_eff_);
      efftree_->Branch("hitmcTkIDs_seed", &hitmcTkIDs_seed_eff_);
      efftree_->Branch("hitmcTkIDs_build", &hitmcTkIDs_build_eff_);
      efftree_->Branch("hitmcTkIDs_fit", &hitmcTkIDs_fit_eff_);

      efftree_->Branch("hitxs_mc", &hitxs_mc_eff_);
      efftree_->Branch("hitxs_seed", &hitxs_seed_eff_);
      efftree_->Branch("hitxs_build", &hitxs_build_eff_);
      efftree_->Branch("hitxs_fit", &hitxs_fit_eff_);

      efftree_->Branch("hitys_mc", &hitys_mc_eff_);
      efftree_->Branch("hitys_seed", &hitys_seed_eff_);
      efftree_->Branch("hitys_build", &hitys_build_eff_);
      efftree_->Branch("hitys_fit", &hitys_fit_eff_);

      efftree_->Branch("hitzs_mc", &hitzs_mc_eff_);
      efftree_->Branch("hitzs_seed", &hitzs_seed_eff_);
      efftree_->Branch("hitzs_build", &hitzs_build_eff_);
      efftree_->Branch("hitzs_fit", &hitzs_fit_eff_);
    }
  }

  void TTreeValidation::initializeFakeRateTree() {
    // fake rate validation
    frtree_ = std::make_unique<TTree>("frtree", "frtree");
    frtree_->SetDirectory(0);

    frtree_->Branch("evtID", &evtID_FR_);
    frtree_->Branch("seedID", &seedID_FR_);

    frtree_->Branch("seedmask_seed", &seedmask_seed_FR_);
    frtree_->Branch("seedmask_build", &seedmask_build_FR_);
    frtree_->Branch("seedmask_fit", &seedmask_fit_FR_);

    frtree_->Branch("xhit_seed", &xhit_seed_FR_);
    frtree_->Branch("xhit_build", &xhit_build_FR_);
    frtree_->Branch("xhit_fit", &xhit_fit_FR_);

    frtree_->Branch("yhit_seed", &yhit_seed_FR_);
    frtree_->Branch("yhit_build", &yhit_build_FR_);
    frtree_->Branch("yhit_fit", &yhit_fit_FR_);

    frtree_->Branch("zhit_seed", &zhit_seed_FR_);
    frtree_->Branch("zhit_build", &zhit_build_FR_);
    frtree_->Branch("zhit_fit", &zhit_fit_FR_);

    frtree_->Branch("pt_seed", &pt_seed_FR_);
    frtree_->Branch("ept_seed", &ept_seed_FR_);
    frtree_->Branch("pt_build", &pt_build_FR_);
    frtree_->Branch("ept_build", &ept_build_FR_);
    frtree_->Branch("pt_fit", &pt_fit_FR_);
    frtree_->Branch("ept_fit", &ept_fit_FR_);

    frtree_->Branch("phi_seed", &phi_seed_FR_);
    frtree_->Branch("ephi_seed", &ephi_seed_FR_);
    frtree_->Branch("phi_build", &phi_build_FR_);
    frtree_->Branch("ephi_build", &ephi_build_FR_);
    frtree_->Branch("phi_fit", &phi_fit_FR_);
    frtree_->Branch("ephi_fit", &ephi_fit_FR_);

    frtree_->Branch("eta_seed", &eta_seed_FR_);
    frtree_->Branch("eeta_seed", &eeta_seed_FR_);
    frtree_->Branch("eta_build", &eta_build_FR_);
    frtree_->Branch("eeta_build", &eeta_build_FR_);
    frtree_->Branch("eta_fit", &eta_fit_FR_);
    frtree_->Branch("eeta_fit", &eeta_fit_FR_);

    frtree_->Branch("nHits_seed", &nHits_seed_FR_);
    frtree_->Branch("nHits_build", &nHits_build_FR_);
    frtree_->Branch("nHits_fit", &nHits_fit_FR_);

    frtree_->Branch("nLayers_seed", &nLayers_seed_FR_);
    frtree_->Branch("nLayers_build", &nLayers_build_FR_);
    frtree_->Branch("nLayers_fit", &nLayers_fit_FR_);

    frtree_->Branch("nHitsMatched_seed", &nHitsMatched_seed_FR_);
    frtree_->Branch("nHitsMatched_build", &nHitsMatched_build_FR_);
    frtree_->Branch("nHitsMatched_fit", &nHitsMatched_fit_FR_);

    frtree_->Branch("fracHitsMatched_seed", &fracHitsMatched_seed_FR_);
    frtree_->Branch("fracHitsMatched_build", &fracHitsMatched_build_FR_);
    frtree_->Branch("fracHitsMatched_fit", &fracHitsMatched_fit_FR_);

    frtree_->Branch("lastlyr_seed", &lastlyr_seed_FR_);
    frtree_->Branch("lastlyr_build", &lastlyr_build_FR_);
    frtree_->Branch("lastlyr_fit", &lastlyr_fit_FR_);

    frtree_->Branch("dphi_seed", &dphi_seed_FR_);
    frtree_->Branch("dphi_build", &dphi_build_FR_);
    frtree_->Branch("dphi_fit", &dphi_fit_FR_);

    frtree_->Branch("hitchi2_seed", &hitchi2_seed_FR_);
    frtree_->Branch("hitchi2_build", &hitchi2_build_FR_);
    frtree_->Branch("hitchi2_fit", &hitchi2_fit_FR_);

    frtree_->Branch("score_seed", &score_seed_FR_);
    frtree_->Branch("score_build", &score_build_FR_);
    frtree_->Branch("score_fit", &score_fit_FR_);

    // sim info of seed,build,fit tracks
    frtree_->Branch("mcID_seed", &mcID_seed_FR_);
    frtree_->Branch("mcID_build", &mcID_build_FR_);
    frtree_->Branch("mcID_fit", &mcID_fit_FR_);

    frtree_->Branch("mcmask_seed", &mcmask_seed_FR_);
    frtree_->Branch("mcmask_build", &mcmask_build_FR_);
    frtree_->Branch("mcmask_fit", &mcmask_fit_FR_);

    frtree_->Branch("mcTSmask_seed", &mcTSmask_seed_FR_);
    frtree_->Branch("mcTSmask_build", &mcTSmask_build_FR_);
    frtree_->Branch("mcTSmask_fit", &mcTSmask_fit_FR_);

    frtree_->Branch("pt_mc_seed", &pt_mc_seed_FR_);
    frtree_->Branch("pt_mc_build", &pt_mc_build_FR_);
    frtree_->Branch("pt_mc_fit", &pt_mc_fit_FR_);

    frtree_->Branch("phi_mc_seed", &phi_mc_seed_FR_);
    frtree_->Branch("phi_mc_build", &phi_mc_build_FR_);
    frtree_->Branch("phi_mc_fit", &phi_mc_fit_FR_);

    frtree_->Branch("eta_mc_seed", &eta_mc_seed_FR_);
    frtree_->Branch("eta_mc_build", &eta_mc_build_FR_);
    frtree_->Branch("eta_mc_fit", &eta_mc_fit_FR_);

    frtree_->Branch("nHits_mc_seed", &nHits_mc_seed_FR_);
    frtree_->Branch("nHits_mc_build", &nHits_mc_build_FR_);
    frtree_->Branch("nHits_mc_fit", &nHits_mc_fit_FR_);

    frtree_->Branch("nLayers_mc_seed", &nLayers_mc_seed_FR_);
    frtree_->Branch("nLayers_mc_build", &nLayers_mc_build_FR_);
    frtree_->Branch("nLayers_mc_fit", &nLayers_mc_fit_FR_);

    frtree_->Branch("lastlyr_mc_seed", &lastlyr_mc_seed_FR_);
    frtree_->Branch("lastlyr_mc_build", &lastlyr_mc_build_FR_);
    frtree_->Branch("lastlyr_mc_fit", &lastlyr_mc_fit_FR_);

    frtree_->Branch("helixchi2_seed", &helixchi2_seed_FR_);
    frtree_->Branch("helixchi2_build", &helixchi2_build_FR_);
    frtree_->Branch("helixchi2_fit", &helixchi2_fit_FR_);

    frtree_->Branch("duplmask_seed", &duplmask_seed_FR_);
    frtree_->Branch("duplmask_build", &duplmask_build_FR_);
    frtree_->Branch("duplmask_fit", &duplmask_fit_FR_);

    frtree_->Branch("iTkMatches_seed", &iTkMatches_seed_FR_);
    frtree_->Branch("iTkMatches_build", &iTkMatches_build_FR_);
    frtree_->Branch("iTkMatches_fit", &iTkMatches_fit_FR_);

    frtree_->Branch("algorithm", &algorithm_FR_);

    if (Config::keepHitInfo) {
      frtree_->Branch("hitlyrs_seed", &hitlyrs_seed_FR_);
      frtree_->Branch("hitlyrs_mc_seed", &hitlyrs_mc_seed_FR_);
      frtree_->Branch("hitlyrs_build", &hitlyrs_build_FR_);
      frtree_->Branch("hitlyrs_mc_build", &hitlyrs_mc_build_FR_);
      frtree_->Branch("hitlyrs_fit", &hitlyrs_fit_FR_);
      frtree_->Branch("hitlyrs_mc_fit", &hitlyrs_mc_fit_FR_);

      frtree_->Branch("hitidxs_seed", &hitidxs_seed_FR_);
      frtree_->Branch("hitidxs_mc_seed", &hitidxs_mc_seed_FR_);
      frtree_->Branch("hitidxs_build", &hitidxs_build_FR_);
      frtree_->Branch("hitidxs_mc_build", &hitidxs_mc_build_FR_);
      frtree_->Branch("hitidxs_fit", &hitidxs_fit_FR_);
      frtree_->Branch("hitidxs_mc_fit", &hitidxs_mc_fit_FR_);

      frtree_->Branch("hitmcTkIDs_seed", &hitmcTkIDs_seed_FR_);
      frtree_->Branch("hitmcTkIDs_mc_seed", &hitmcTkIDs_mc_seed_FR_);
      frtree_->Branch("hitmcTkIDs_build", &hitmcTkIDs_build_FR_);
      frtree_->Branch("hitmcTkIDs_mc_build", &hitmcTkIDs_mc_build_FR_);
      frtree_->Branch("hitmcTkIDs_fit", &hitmcTkIDs_fit_FR_);
      frtree_->Branch("hitmcTkIDs_mc_fit", &hitmcTkIDs_mc_fit_FR_);

      frtree_->Branch("hitxs_seed", &hitxs_seed_FR_);
      frtree_->Branch("hitxs_mc_seed", &hitxs_mc_seed_FR_);
      frtree_->Branch("hitxs_build", &hitxs_build_FR_);
      frtree_->Branch("hitxs_mc_build", &hitxs_mc_build_FR_);
      frtree_->Branch("hitxs_fit", &hitxs_fit_FR_);
      frtree_->Branch("hitxs_mc_fit", &hitxs_mc_fit_FR_);

      frtree_->Branch("hitys_seed", &hitys_seed_FR_);
      frtree_->Branch("hitys_mc_seed", &hitys_mc_seed_FR_);
      frtree_->Branch("hitys_build", &hitys_build_FR_);
      frtree_->Branch("hitys_mc_build", &hitys_mc_build_FR_);
      frtree_->Branch("hitys_fit", &hitys_fit_FR_);
      frtree_->Branch("hitys_mc_fit", &hitys_mc_fit_FR_);

      frtree_->Branch("hitzs_seed", &hitzs_seed_FR_);
      frtree_->Branch("hitzs_mc_seed", &hitzs_mc_seed_FR_);
      frtree_->Branch("hitzs_build", &hitzs_build_FR_);
      frtree_->Branch("hitzs_mc_build", &hitzs_mc_build_FR_);
      frtree_->Branch("hitzs_fit", &hitzs_fit_FR_);
      frtree_->Branch("hitzs_mc_fit", &hitzs_mc_fit_FR_);
    }
  }

  void TTreeValidation::initializeConfigTree() {
    // include config ++ real seeding parameters ...
    configtree_ = std::make_unique<TTree>("configtree", "configtree");
    configtree_->SetDirectory(0);

    configtree_->Branch("Ntracks", &Ntracks_);
    configtree_->Branch("Nevents", &Nevents_);

    configtree_->Branch("nLayers", &nLayers_);

    configtree_->Branch("nlayers_per_seed", &nlayers_per_seed_);
    configtree_->Branch("maxCand", &maxCand_);
    configtree_->Branch("chi2Cut_min", &chi2Cut_min_);
    configtree_->Branch("nSigma", &nSigma_);
    configtree_->Branch("minDPhi", &minDPhi_);
    configtree_->Branch("maxDPhi", &maxDPhi_);
    configtree_->Branch("minDEta", &minDEta_);
    configtree_->Branch("maxDEta", &maxDEta_);

    configtree_->Branch("beamspotX", &beamspotX_);
    configtree_->Branch("beamspotY", &beamspotY_);
    configtree_->Branch("beamspotZ", &beamspotZ_);

    configtree_->Branch("minSimPt", &minSimPt_);
    configtree_->Branch("maxSimPt", &maxSimPt_);

    configtree_->Branch("hitposerrXY", &hitposerrXY_);
    configtree_->Branch("hitposerrZ", &hitposerrZ_);
    configtree_->Branch("hitposerrR", &hitposerrR_);

    configtree_->Branch("varXY", &varXY_);
    configtree_->Branch("varZ", &varZ_);

    configtree_->Branch("ptinverr049", &ptinverr049_);
    configtree_->Branch("phierr049", &phierr049_);
    configtree_->Branch("thetaerr049", &thetaerr049_);
    configtree_->Branch("ptinverr012", &ptinverr012_);
    configtree_->Branch("phierr012", &phierr012_);
    configtree_->Branch("thetaerr012", &thetaerr012_);
  }

  void TTreeValidation::initializeCMSSWEfficiencyTree() {
    // cmssw reco track efficiency validation
    cmsswefftree_ = std::make_unique<TTree>("cmsswefftree", "cmsswefftree");
    cmsswefftree_->SetDirectory(0);

    cmsswefftree_->Branch("evtID", &evtID_ceff_);
    cmsswefftree_->Branch("cmsswID", &cmsswID_ceff_);
    cmsswefftree_->Branch("seedID_cmssw", &seedID_cmssw_ceff_);

    // CMSSW
    cmsswefftree_->Branch("x_cmssw", &x_cmssw_ceff_);
    cmsswefftree_->Branch("y_cmssw", &y_cmssw_ceff_);
    cmsswefftree_->Branch("z_cmssw", &z_cmssw_ceff_);

    cmsswefftree_->Branch("pt_cmssw", &pt_cmssw_ceff_);
    cmsswefftree_->Branch("phi_cmssw", &phi_cmssw_ceff_);
    cmsswefftree_->Branch("eta_cmssw", &eta_cmssw_ceff_);

    cmsswefftree_->Branch("nHits_cmssw", &nHits_cmssw_ceff_);
    cmsswefftree_->Branch("nLayers_cmssw", &nLayers_cmssw_ceff_);
    cmsswefftree_->Branch("lastlyr_cmssw", &lastlyr_cmssw_ceff_);

    // Build
    cmsswefftree_->Branch("cmsswmask_build", &cmsswmask_build_ceff_);
    cmsswefftree_->Branch("seedID_build", &seedID_build_ceff_);
    cmsswefftree_->Branch("mcTrackID_build", &mcTrackID_build_ceff_);

    cmsswefftree_->Branch("pt_build", &pt_build_ceff_);
    cmsswefftree_->Branch("ept_build", &ept_build_ceff_);
    cmsswefftree_->Branch("phi_build", &phi_build_ceff_);
    cmsswefftree_->Branch("ephi_build", &ephi_build_ceff_);
    cmsswefftree_->Branch("eta_build", &eta_build_ceff_);
    cmsswefftree_->Branch("eeta_build", &eeta_build_ceff_);

    cmsswefftree_->Branch("x_mc_build", &x_mc_build_ceff_);
    cmsswefftree_->Branch("y_mc_build", &y_mc_build_ceff_);
    cmsswefftree_->Branch("z_mc_build", &z_mc_build_ceff_);
    cmsswefftree_->Branch("pt_mc_build", &pt_mc_build_ceff_);
    cmsswefftree_->Branch("phi_mc_build", &phi_mc_build_ceff_);
    cmsswefftree_->Branch("eta_mc_build", &eta_mc_build_ceff_);

    cmsswefftree_->Branch("nHits_build", &nHits_build_ceff_);
    cmsswefftree_->Branch("nLayers_build", &nLayers_build_ceff_);
    cmsswefftree_->Branch("nHitsMatched_build", &nHitsMatched_build_ceff_);
    cmsswefftree_->Branch("fracHitsMatched_build", &fracHitsMatched_build_ceff_);
    cmsswefftree_->Branch("lastlyr_build", &lastlyr_build_ceff_);

    cmsswefftree_->Branch("xhit_build", &xhit_build_ceff_);
    cmsswefftree_->Branch("yhit_build", &yhit_build_ceff_);
    cmsswefftree_->Branch("zhit_build", &zhit_build_ceff_);

    cmsswefftree_->Branch("hitchi2_build", &hitchi2_build_ceff_);
    cmsswefftree_->Branch("helixchi2_build", &helixchi2_build_ceff_);
    cmsswefftree_->Branch("score_build", &score_build_ceff_);
    cmsswefftree_->Branch("dphi_build", &dphi_build_ceff_);

    cmsswefftree_->Branch("duplmask_build", &duplmask_build_ceff_);
    cmsswefftree_->Branch("nTkMatches_build", &nTkMatches_build_ceff_);

    cmsswefftree_->Branch("itermask_build", &itermask_build_ceff_);
    cmsswefftree_->Branch("iterduplmask_build", &iterduplmask_build_ceff_);

    // Fit
    cmsswefftree_->Branch("cmsswmask_fit", &cmsswmask_fit_ceff_);
    cmsswefftree_->Branch("seedID_fit", &seedID_fit_ceff_);
    cmsswefftree_->Branch("mcTrackID_fit", &mcTrackID_fit_ceff_);

    cmsswefftree_->Branch("pt_fit", &pt_fit_ceff_);
    cmsswefftree_->Branch("ept_fit", &ept_fit_ceff_);
    cmsswefftree_->Branch("phi_fit", &phi_fit_ceff_);
    cmsswefftree_->Branch("ephi_fit", &ephi_fit_ceff_);
    cmsswefftree_->Branch("eta_fit", &eta_fit_ceff_);
    cmsswefftree_->Branch("eeta_fit", &eeta_fit_ceff_);

    cmsswefftree_->Branch("x_mc_fit", &x_mc_fit_ceff_);
    cmsswefftree_->Branch("y_mc_fit", &y_mc_fit_ceff_);
    cmsswefftree_->Branch("z_mc_fit", &z_mc_fit_ceff_);
    cmsswefftree_->Branch("pt_mc_fit", &pt_mc_fit_ceff_);
    cmsswefftree_->Branch("phi_mc_fit", &phi_mc_fit_ceff_);
    cmsswefftree_->Branch("eta_mc_fit", &eta_mc_fit_ceff_);

    cmsswefftree_->Branch("nHits_fit", &nHits_fit_ceff_);
    cmsswefftree_->Branch("nLayers_fit", &nLayers_fit_ceff_);
    cmsswefftree_->Branch("nHitsMatched_fit", &nHitsMatched_fit_ceff_);
    cmsswefftree_->Branch("fracHitsMatched_fit", &fracHitsMatched_fit_ceff_);
    cmsswefftree_->Branch("lastlyr_fit", &lastlyr_fit_ceff_);

    cmsswefftree_->Branch("xhit_fit", &xhit_fit_ceff_);
    cmsswefftree_->Branch("yhit_fit", &yhit_fit_ceff_);
    cmsswefftree_->Branch("zhit_fit", &zhit_fit_ceff_);

    cmsswefftree_->Branch("hitchi2_fit", &hitchi2_fit_ceff_);
    cmsswefftree_->Branch("helixchi2_fit", &helixchi2_fit_ceff_);
    cmsswefftree_->Branch("score_fit", &score_fit_ceff_);
    cmsswefftree_->Branch("dphi_fit", &dphi_fit_ceff_);

    cmsswefftree_->Branch("duplmask_fit", &duplmask_fit_ceff_);
    cmsswefftree_->Branch("nTkMatches_fit", &nTkMatches_fit_ceff_);

    cmsswefftree_->Branch("itermask_fit", &itermask_fit_ceff_);
    cmsswefftree_->Branch("iterduplmask_fit", &iterduplmask_fit_ceff_);

    cmsswefftree_->Branch("algo_seed", &algo_seed_ceff_);

    if (Config::keepHitInfo) {
      cmsswefftree_->Branch("hitlyrs_cmssw", &hitlyrs_cmssw_ceff_);
      cmsswefftree_->Branch("hitlyrs_build", &hitlyrs_build_ceff_);
      cmsswefftree_->Branch("hitlyrs_mc_build", &hitlyrs_mc_build_ceff_);
      cmsswefftree_->Branch("hitlyrs_fit", &hitlyrs_fit_ceff_);
      cmsswefftree_->Branch("hitlyrs_mc_fit", &hitlyrs_mc_fit_ceff_);

      cmsswefftree_->Branch("hitidxs_cmssw", &hitidxs_cmssw_ceff_);
      cmsswefftree_->Branch("hitidxs_build", &hitidxs_build_ceff_);
      cmsswefftree_->Branch("hitidxs_mc_build", &hitidxs_mc_build_ceff_);
      cmsswefftree_->Branch("hitidxs_fit", &hitidxs_fit_ceff_);
      cmsswefftree_->Branch("hitidxs_mc_fit", &hitidxs_mc_fit_ceff_);
    }
  }

  void TTreeValidation::initializeCMSSWFakeRateTree() {
    // cmssw reco track efficiency validation
    cmsswfrtree_ = std::make_unique<TTree>("cmsswfrtree", "cmsswfrtree");
    cmsswfrtree_->SetDirectory(0);

    cmsswfrtree_->Branch("evtID", &evtID_cFR_);
    cmsswfrtree_->Branch("seedID", &seedID_cFR_);
    cmsswfrtree_->Branch("mcTrackID", &mcTrackID_cFR_);

    // mc
    cmsswfrtree_->Branch("x_mc", &x_mc_cFR_);
    cmsswfrtree_->Branch("y_mc", &y_mc_cFR_);
    cmsswfrtree_->Branch("z_mc", &z_mc_cFR_);
    cmsswfrtree_->Branch("pt_mc", &pt_mc_cFR_);
    cmsswfrtree_->Branch("phi_mc", &phi_mc_cFR_);
    cmsswfrtree_->Branch("eta_mc", &eta_mc_cFR_);

    // build
    cmsswfrtree_->Branch("cmsswID_build", &cmsswID_build_cFR_);
    cmsswfrtree_->Branch("cmsswmask_build", &cmsswmask_build_cFR_);

    cmsswfrtree_->Branch("pt_build", &pt_build_cFR_);
    cmsswfrtree_->Branch("ept_build", &ept_build_cFR_);
    cmsswfrtree_->Branch("phi_build", &phi_build_cFR_);
    cmsswfrtree_->Branch("ephi_build", &ephi_build_cFR_);
    cmsswfrtree_->Branch("eta_build", &eta_build_cFR_);
    cmsswfrtree_->Branch("eeta_build", &eeta_build_cFR_);

    cmsswfrtree_->Branch("nHits_build", &nHits_build_cFR_);
    cmsswfrtree_->Branch("nLayers_build", &nLayers_build_cFR_);
    cmsswfrtree_->Branch("nHitsMatched_build", &nHitsMatched_build_cFR_);
    cmsswfrtree_->Branch("fracHitsMatched_build", &fracHitsMatched_build_cFR_);
    cmsswfrtree_->Branch("lastlyr_build", &lastlyr_build_cFR_);

    cmsswfrtree_->Branch("xhit_build", &xhit_build_cFR_);
    cmsswfrtree_->Branch("yhit_build", &yhit_build_cFR_);
    cmsswfrtree_->Branch("zhit_build", &zhit_build_cFR_);

    cmsswfrtree_->Branch("hitchi2_build", &hitchi2_build_cFR_);
    cmsswfrtree_->Branch("helixchi2_build", &helixchi2_build_cFR_);
    cmsswfrtree_->Branch("score_build", &score_build_cFR_);
    cmsswfrtree_->Branch("dphi_build", &dphi_build_cFR_);

    cmsswfrtree_->Branch("duplmask_build", &duplmask_build_cFR_);
    cmsswfrtree_->Branch("iTkMatches_build", &iTkMatches_build_cFR_);

    cmsswfrtree_->Branch("seedID_cmssw_build", &seedID_cmssw_build_cFR_);

    cmsswfrtree_->Branch("x_cmssw_build", &x_cmssw_build_cFR_);
    cmsswfrtree_->Branch("y_cmssw_build", &y_cmssw_build_cFR_);
    cmsswfrtree_->Branch("z_cmssw_build", &z_cmssw_build_cFR_);

    cmsswfrtree_->Branch("pt_cmssw_build", &pt_cmssw_build_cFR_);
    cmsswfrtree_->Branch("phi_cmssw_build", &phi_cmssw_build_cFR_);
    cmsswfrtree_->Branch("eta_cmssw_build", &eta_cmssw_build_cFR_);

    cmsswfrtree_->Branch("nHits_cmssw_build", &nHits_cmssw_build_cFR_);
    cmsswfrtree_->Branch("nLayers_cmssw_build", &nLayers_cmssw_build_cFR_);
    cmsswfrtree_->Branch("lastlyr_cmssw_build", &lastlyr_cmssw_build_cFR_);

    // fit
    cmsswfrtree_->Branch("cmsswID_fit", &cmsswID_fit_cFR_);
    cmsswfrtree_->Branch("cmsswmask_fit", &cmsswmask_fit_cFR_);

    cmsswfrtree_->Branch("pt_fit", &pt_fit_cFR_);
    cmsswfrtree_->Branch("ept_fit", &ept_fit_cFR_);
    cmsswfrtree_->Branch("phi_fit", &phi_fit_cFR_);
    cmsswfrtree_->Branch("ephi_fit", &ephi_fit_cFR_);
    cmsswfrtree_->Branch("eta_fit", &eta_fit_cFR_);
    cmsswfrtree_->Branch("eeta_fit", &eeta_fit_cFR_);

    cmsswfrtree_->Branch("nHits_fit", &nHits_fit_cFR_);
    cmsswfrtree_->Branch("nLayers_fit", &nLayers_fit_cFR_);
    cmsswfrtree_->Branch("nHitsMatched_fit", &nHitsMatched_fit_cFR_);
    cmsswfrtree_->Branch("fracHitsMatched_fit", &fracHitsMatched_fit_cFR_);
    cmsswfrtree_->Branch("lastlyr_fit", &lastlyr_fit_cFR_);

    cmsswfrtree_->Branch("xhit_fit", &xhit_fit_cFR_);
    cmsswfrtree_->Branch("yhit_fit", &yhit_fit_cFR_);
    cmsswfrtree_->Branch("zhit_fit", &zhit_fit_cFR_);

    cmsswfrtree_->Branch("hitchi2_fit", &hitchi2_fit_cFR_);
    cmsswfrtree_->Branch("helixchi2_fit", &helixchi2_fit_cFR_);
    cmsswfrtree_->Branch("score_fit", &score_fit_cFR_);
    cmsswfrtree_->Branch("dphi_fit", &dphi_fit_cFR_);

    cmsswfrtree_->Branch("duplmask_fit", &duplmask_fit_cFR_);
    cmsswfrtree_->Branch("iTkMatches_fit", &iTkMatches_fit_cFR_);

    cmsswfrtree_->Branch("seedID_cmssw_fit", &seedID_cmssw_fit_cFR_);

    cmsswfrtree_->Branch("x_cmssw_fit", &x_cmssw_fit_cFR_);
    cmsswfrtree_->Branch("y_cmssw_fit", &y_cmssw_fit_cFR_);
    cmsswfrtree_->Branch("z_cmssw_fit", &z_cmssw_fit_cFR_);

    cmsswfrtree_->Branch("pt_cmssw_fit", &pt_cmssw_fit_cFR_);
    cmsswfrtree_->Branch("phi_cmssw_fit", &phi_cmssw_fit_cFR_);
    cmsswfrtree_->Branch("eta_cmssw_fit", &eta_cmssw_fit_cFR_);

    cmsswfrtree_->Branch("nHits_cmssw_fit", &nHits_cmssw_fit_cFR_);
    cmsswfrtree_->Branch("nLayers_cmssw_fit", &nLayers_cmssw_fit_cFR_);
    cmsswfrtree_->Branch("lastlyr_cmssw_fit", &lastlyr_cmssw_fit_cFR_);

    cmsswfrtree_->Branch("algorithm", &algorithm_cFR_);

    if (Config::keepHitInfo) {
      cmsswfrtree_->Branch("hitlyrs_mc", &hitlyrs_mc_cFR_);
      cmsswfrtree_->Branch("hitlyrs_build", &hitlyrs_build_cFR_);
      cmsswfrtree_->Branch("hitlyrs_cmssw_build", &hitlyrs_cmssw_build_cFR_);
      cmsswfrtree_->Branch("hitlyrs_fit", &hitlyrs_fit_cFR_);
      cmsswfrtree_->Branch("hitlyrs_cmssw_fit", &hitlyrs_cmssw_fit_cFR_);

      cmsswfrtree_->Branch("hitidxs_mc", &hitidxs_mc_cFR_);
      cmsswfrtree_->Branch("hitidxs_build", &hitidxs_build_cFR_);
      cmsswfrtree_->Branch("hitidxs_cmssw_build", &hitidxs_cmssw_build_cFR_);
      cmsswfrtree_->Branch("hitidxs_fit", &hitidxs_fit_cFR_);
      cmsswfrtree_->Branch("hitidxs_cmssw_fit", &hitidxs_cmssw_fit_cFR_);
    }
  }

  void TTreeValidation::initializeFitTree() {
    fittree_ = std::make_unique<TTree>("fittree", "fittree");
    fittree_->SetDirectory(0);

    fittree_->Branch("ntotallayers", &ntotallayers_fit_, "ntotallayers_fit_/I");
    fittree_->Branch("tkid", &tkid_fit_, "tkid_fit_/I");
    fittree_->Branch("evtid", &evtid_fit_, "evtid_fit_/I");

    fittree_->Branch("z_prop", &z_prop_fit_, "z_prop_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("ez_prop", &ez_prop_fit_, "ez_prop_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("z_hit", &z_hit_fit_, "z_hit_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("ez_hit", &ez_hit_fit_, "ez_hit_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("z_sim", &z_sim_fit_, "z_sim_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("ez_sim", &ez_sim_fit_, "ez_sim_fit_[ntotallayers_fit_]/F");

    fittree_->Branch("pphi_prop", &pphi_prop_fit_, "pphi_prop_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("epphi_prop", &epphi_prop_fit_, "epphi_prop_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("pphi_hit", &pphi_hit_fit_, "pphi_hit_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("epphi_hit", &epphi_hit_fit_, "epphi_hit_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("pphi_sim", &pphi_sim_fit_, "pphi_sim_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("epphi_sim", &epphi_sim_fit_, "epphi_sim_fit_[ntotallayers_fit_]/F");

    fittree_->Branch("pt_up", &pt_up_fit_, "pt_up_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("ept_up", &ept_up_fit_, "ept_up_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("pt_sim", &pt_sim_fit_, "pt_sim_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("ept_sim", &ept_sim_fit_, "ept_sim_fit_[ntotallayers_fit_]/F");

    fittree_->Branch("mphi_up", &mphi_up_fit_, "mphi_up_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("emphi_up", &emphi_up_fit_, "emphi_up_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("mphi_sim", &mphi_sim_fit_, "mphi_sim_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("emphi_sim", &emphi_sim_fit_, "emphi_sim_fit_[ntotallayers_fit_]/F");

    fittree_->Branch("meta_up", &meta_up_fit_, "meta_up_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("emeta_up", &emeta_up_fit_, "emeta_up_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("meta_sim", &meta_sim_fit_, "meta_sim_fit_[ntotallayers_fit_]/F");
    fittree_->Branch("emeta_sim", &emeta_sim_fit_, "emeta_sim_fit_[ntotallayers_fit_]/F");
  }

  void TTreeValidation::alignTracks(TrackVec& evt_tracks, TrackExtraVec& evt_extras, bool alignExtra) {
    std::lock_guard<std::mutex> locker(glock_);

    // redo trackExtras first if necessary
    if (alignExtra) {
      TrackExtraVec trackExtra_tmp(evt_tracks.size());

      // align temporary tkExVec with new track collection ordering
      for (int itrack = 0; itrack < (int)evt_tracks.size(); itrack++) {
        trackExtra_tmp[itrack] = evt_extras[evt_tracks[itrack].label()];  // label is old seedID!
      }

      // now copy the temporary back in the old one
      evt_extras = trackExtra_tmp;
    }

    // redo track labels to match index in vector
    for (int itrack = 0; itrack < (int)evt_tracks.size(); itrack++) {
      evt_tracks[itrack].setLabel(itrack);
    }
  }

  void TTreeValidation::collectFitInfo(const FitVal& tmpfitval, int tkid, int layer) {
    std::lock_guard<std::mutex> locker(glock_);

    fitValTkMapMap_[tkid][layer] = tmpfitval;
  }

  void TTreeValidation::resetValidationMaps() {
    std::lock_guard<std::mutex> locker(glock_);
    // reset fit validation map
    fitValTkMapMap_.clear();

    // reset map of sim tracks to reco tracks
    simToSeedMap_.clear();
    simToBuildMap_.clear();
    simToFitMap_.clear();

    // reset map of seed tracks to reco tracks
    seedToBuildMap_.clear();
    seedToFitMap_.clear();

    // reset map of cmssw tracks to reco tracks
    cmsswToBuildMap_.clear();
    cmsswToFitMap_.clear();

    // reset special map of seed labels to cmssw tracks
    seedToCmsswMap_.clear();
    cmsswToSeedMap_.clear();

    // reset special map of matching build tracks exactly to cmssw tracks through seedIDs
    buildToCmsswMap_.clear();

    // reset special maps used for pairing build to fit tracks CMSSW only
    buildToFitMap_.clear();
    fitToBuildMap_.clear();

    // reset special maps used for associating seed tracks to reco tracks for sim_val_for_cmssw
    candToSeedMapDumbCMSSW_.clear();
    fitToSeedMapDumbCMSSW_.clear();
  }

  void TTreeValidation::setTrackExtras(Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);

    const auto& layerhits = ev.layerHits_;

    if (Config::sim_val_for_cmssw || Config::sim_val) {
      const auto& simhits = ev.simHitsInfo_;
      const auto& simtracks = ev.simTracks_;
      const auto& seedtracks = ev.seedTracks_;
      auto& seedextras = ev.seedTracksExtra_;
      const auto& buildtracks = ev.candidateTracks_;
      auto& buildextras = ev.candidateTracksExtra_;
      const auto& fittracks = ev.fitTracks_;
      auto& fitextras = ev.fitTracksExtra_;

      // set mcTrackID for seed tracks
      for (int itrack = 0; itrack < (int)seedtracks.size(); itrack++) {
        const auto& track = seedtracks[itrack];
        auto& extra = seedextras[itrack];

        extra.findMatchingSeedHits(track, track, layerhits);
        extra.setMCTrackIDInfo(
            track,
            layerhits,
            simhits,
            simtracks,
            true,
            (Config::seedInput == simSeeds));  // otherwise seeds are completely unmatched in ToyMC Sim Seeds
      }

      // set mcTrackID for built tracks
      for (int itrack = 0; itrack < (int)buildtracks.size(); itrack++) {
        const auto& track = buildtracks[itrack];
        auto& extra = buildextras[itrack];

        if (Config::sim_val) {
          extra.findMatchingSeedHits(track, seedtracks[track.label()], layerhits);
        } else if (Config::sim_val_for_cmssw) {
          extra.findMatchingSeedHits(track, seedtracks[candToSeedMapDumbCMSSW_[track.label()]], layerhits);
        }

        extra.setMCTrackIDInfo(track, layerhits, simhits, simtracks, false, (Config::seedInput == simSeeds));
      }

      // set mcTrackID for fit tracks
      for (int itrack = 0; itrack < (int)fittracks.size(); itrack++) {
        const auto& track = fittracks[itrack];
        auto& extra = fitextras[itrack];

        if (Config::sim_val) {
          extra.findMatchingSeedHits(track, seedtracks[track.label()], layerhits);
        } else if (Config::sim_val_for_cmssw) {
          extra.findMatchingSeedHits(track, seedtracks[fitToSeedMapDumbCMSSW_[track.label()]], layerhits);
        }

        extra.setMCTrackIDInfo(track, layerhits, simhits, simtracks, false, (Config::seedInput == simSeeds));
      }
    }

    if (Config::cmssw_val) {
      // store mcTrackID and seedID correctly
      storeSeedAndMCID(ev);

      const auto& cmsswtracks = ev.cmsswTracks_;
      const auto& cmsswextras = ev.cmsswTracksExtra_;
      const auto& seedtracks = ev.seedTracks_;
      const auto& buildtracks = ev.candidateTracks_;
      auto& buildextras = ev.candidateTracksExtra_;
      const auto& fittracks = ev.fitTracks_;
      auto& fitextras = ev.fitTracksExtra_;

      // store seed hits, reduced parameters, hit map of cmssw tracks, and global hit map
      RedTrackVec reducedCMSSW;
      LayIdxIDVecMapMap cmsswHitIDMap;
      setupCMSSWMatching(ev, reducedCMSSW, cmsswHitIDMap);

      // set cmsswTrackID for built tracks
      for (int itrack = 0; itrack < (int)buildtracks.size(); itrack++) {
        const auto& track = buildtracks[itrack];
        auto& extra = buildextras[itrack];

        // set vector of hitsOnTrack for seed
        extra.findMatchingSeedHits(track,
                                   seedtracks[track.label()],
                                   layerhits);  // itrack == track.label() == seedtrack index == seedtrack.label()

        if (Config::cmsswMatchingFW == trkParamBased) {
          extra.setCMSSWTrackIDInfoByTrkParams(track, layerhits, cmsswtracks, reducedCMSSW, true);
        } else if (Config::cmsswMatchingFW == hitBased) {
          extra.setCMSSWTrackIDInfoByHits(track,
                                          cmsswHitIDMap,
                                          cmsswtracks,
                                          cmsswextras,
                                          reducedCMSSW,
                                          -1);             // == -1 for not passing truth info about cmssw tracks
        } else if (Config::cmsswMatchingFW == labelBased)  // can only be used if using pure seeds!
        {
          extra.setCMSSWTrackIDInfoByHits(track,
                                          cmsswHitIDMap,
                                          cmsswtracks,
                                          cmsswextras,
                                          reducedCMSSW,
                                          reducedCMSSW[cmsswtracks[buildToCmsswMap_[track.label()]].label()].label());
        } else {
          std::cerr << "Specified CMSSW validation, but using an incorrect matching option! Exiting..." << std::endl;
          exit(1);
        }
      }

      // set cmsswTrackID for fit tracks
      for (int itrack = 0; itrack < (int)fittracks.size(); itrack++) {
        const auto& track = fittracks[itrack];
        auto& extra = fitextras[itrack];

        // set vector of hitsOnTrack for seed
        extra.findMatchingSeedHits(track,
                                   seedtracks[track.label()],
                                   layerhits);  // itrack == track.label() == seedtrack index == seedtrack.label()

        if (Config::cmsswMatchingBK == trkParamBased) {
          extra.setCMSSWTrackIDInfoByTrkParams(track, layerhits, cmsswtracks, reducedCMSSW, true);
        } else if (Config::cmsswMatchingBK == hitBased) {
          extra.setCMSSWTrackIDInfoByHits(track,
                                          cmsswHitIDMap,
                                          cmsswtracks,
                                          cmsswextras,
                                          reducedCMSSW,
                                          -1);             // == -1 not passing truth info about cmssw
        } else if (Config::cmsswMatchingBK == labelBased)  // can only be used if using pure seeds!
        {
          extra.setCMSSWTrackIDInfoByHits(
              track,
              cmsswHitIDMap,
              cmsswtracks,
              cmsswextras,
              reducedCMSSW,
              reducedCMSSW[cmsswtracks[buildToCmsswMap_[fitToBuildMap_[track.label()]]].label()].label());
        } else {
          std::cerr << "Specified CMSSW validation, but using an incorrect matching option! Exiting..." << std::endl;
          exit(1);
        }
      }
    }
  }

  void TTreeValidation::makeSimTkToRecoTksMaps(Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);
    // map sim track ids to reco labels sort by each (simTracks set in order by default!)
    TTreeValidation::mapRefTkToRecoTks(ev.seedTracks_, ev.seedTracksExtra_, simToSeedMap_);
    TTreeValidation::mapRefTkToRecoTks(ev.candidateTracks_, ev.candidateTracksExtra_, simToBuildMap_);
    TTreeValidation::mapRefTkToRecoTks(ev.fitTracks_, ev.fitTracksExtra_, simToFitMap_);
  }

  void TTreeValidation::mapRefTkToRecoTks(const TrackVec& evt_tracks,
                                          TrackExtraVec& evt_extras,
                                          TkIDToTkIDVecMap& refTkMap) {
    for (auto itrack = 0; itrack < (int)evt_tracks.size(); ++itrack) {
      auto&& track(evt_tracks[itrack]);
      auto&& extra(evt_extras[itrack]);
      if (Config::sim_val_for_cmssw || Config::sim_val) {
        if (extra.mcTrackID() >= 0)  // skip fakes, don't store them at all in sim map
        {
          refTkMap[extra.mcTrackID()].push_back(
              track.label());  // store vector of reco tk labels, mapped to the sim track label (i.e. mcTrackID)
        }
      }
      if (Config::cmssw_val) {
        if (extra.cmsswTrackID() >= 0)  // skip fakes, don't store them at all in cmssw map
        {
          refTkMap[extra.cmsswTrackID()].push_back(
              track.label());  // store vector of reco tk labels, mapped to the cmssw track label (i.e. cmsswTrackID)
        }
      }
    }

    for (auto&& refTkMatches : refTkMap) {
      if (refTkMatches.second.size() < 2)  // no duplicates
      {
        auto& extra(evt_extras[refTkMatches.second[0]]);
        extra.setDuplicateInfo(0, bool(false));
      } else  // sort duplicates (ghosts) to keep best one --> best score
      {
        // really should sort on indices with a reduced data structure... this is a hack way to do this for now...
        // e.g. std::pair<int, int> (label, score)
        TrackVec tmpMatches;
        for (auto&& label :
             refTkMatches.second)  // loop over vector of reco track labels, push back the track with each label
        {
          tmpMatches.emplace_back(evt_tracks[label]);
        }
        //std::sort(tmpMatches.begin(), tmpMatches.end(), sortByHitsChi2); // sort the tracks
        std::sort(tmpMatches.begin(), tmpMatches.end(), sortByScoreCand);  // sort the tracks
        for (auto itrack = 0; itrack < (int)tmpMatches.size();
             itrack++)  // loop over sorted tracks, now set the vector of sorted labels match
        {
          refTkMatches.second[itrack] = tmpMatches[itrack].label();
        }

        int duplicateID = 0;
        for (auto&& label : refTkMatches.second)  // loop over vector of reco tracsk
        {
          auto& extra(evt_extras[label]);
          extra.setDuplicateInfo(duplicateID, bool(true));
          duplicateID++;  // used in fake rate trees!
        }
      }
    }
  }

  void TTreeValidation::makeSeedTkToRecoTkMaps(Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);
    // map seed to reco tracks --> seed track collection assumed to map to itself, unless we use some cuts
    TTreeValidation::mapSeedTkToRecoTk(ev.candidateTracks_, ev.candidateTracksExtra_, seedToBuildMap_);
    TTreeValidation::mapSeedTkToRecoTk(ev.fitTracks_, ev.fitTracksExtra_, seedToFitMap_);
  }

  void TTreeValidation::mapSeedTkToRecoTk(const TrackVec& evt_tracks,
                                          const TrackExtraVec& evt_extras,
                                          TkIDToTkIDMap& seedTkMap) {
    for (auto&& track : evt_tracks) {
      seedTkMap[evt_extras[track.label()].seedID()] = track.label();
    }
  }

  void TTreeValidation::makeRecoTkToRecoTkMaps(Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);
    TTreeValidation::makeRecoTkToRecoTkMap(
        buildToFitMap_, ev.candidateTracks_, ev.candidateTracksExtra_, ev.fitTracks_, ev.fitTracksExtra_);
    TTreeValidation::makeRecoTkToRecoTkMap(
        fitToBuildMap_, ev.fitTracks_, ev.fitTracksExtra_, ev.candidateTracks_, ev.candidateTracksExtra_);
  }

  void TTreeValidation::makeRecoTkToRecoTkMap(TkIDToTkIDMap& refToPairMap,
                                              const TrackVec& reftracks,
                                              const TrackExtraVec& refextras,
                                              const TrackVec& pairtracks,
                                              const TrackExtraVec& pairextras) {
    // at this point in the code, the labels of the tracks point their position inside the vector
    // while the seedID is the label prior to relabeling (in reality, this is the MC track ID)
    for (auto&& reftrack : reftracks) {
      const auto& refextra = refextras[reftrack.label()];
      for (auto&& pairtrack : pairtracks) {
        const auto& pairextra = pairextras[pairtrack.label()];
        if (refextra.seedID() == pairextra.seedID()) {
          refToPairMap[reftrack.label()] = pairtrack.label();
          break;
        }
      }
    }
  }

  void TTreeValidation::makeCMSSWTkToRecoTksMaps(Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);
    // can reuse this function
    TTreeValidation::mapRefTkToRecoTks(ev.candidateTracks_, ev.candidateTracksExtra_, cmsswToBuildMap_);
    TTreeValidation::mapRefTkToRecoTks(ev.fitTracks_, ev.fitTracksExtra_, cmsswToFitMap_);
  }

  void TTreeValidation::makeSeedTkToCMSSWTkMap(Event& ev) {
    const auto& seedtracks = ev.seedTracks_;
    const auto& cmsswtracks = ev.cmsswTracks_;
    for (int itrack = 0; itrack < (int)seedtracks.size(); itrack++) {
      for (auto&& cmsswtrack : cmsswtracks) {
        if (cmsswtrack.label() == itrack) {
          seedToCmsswMap_[seedtracks[itrack].label()] = cmsswtrack.label();
          break;
        }
      }
    }
  }

  void TTreeValidation::makeCMSSWTkToSeedTkMap(Event& ev) {
    const auto& seedtracks = ev.seedTracks_;

    for (const auto& seedToCmsswPair : seedToCmsswMap_) {
      const auto seedlabel =
          seedToCmsswPair
              .first;  // !! in cmssw validation, seed label != seed index in vector as they are not aligned!! --> need to find itrack!
      const auto cmsswlabel = seedToCmsswPair.second;  // however, cmssw tracks ARE aligned for label == index

      for (int itrack = 0; itrack < (int)seedtracks.size(); itrack++) {
        const auto& seedtrack = seedtracks[itrack];
        if (seedtrack.label() == seedlabel) {
          cmsswToSeedMap_[cmsswlabel] = itrack;
          break;
        }
      }
    }
  }

  void TTreeValidation::makeRecoTkToSeedTkMapsDumbCMSSW(Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);
    // special functions for matching seeds to reco tracks for sim_val_for_cmssw
    TTreeValidation::makeRecoTkToSeedTkMapDumbCMSSW(
        ev.candidateTracksExtra_, ev.seedTracksExtra_, candToSeedMapDumbCMSSW_);
    TTreeValidation::makeRecoTkToSeedTkMapDumbCMSSW(ev.fitTracksExtra_, ev.seedTracksExtra_, fitToSeedMapDumbCMSSW_);
  }

  void TTreeValidation::makeRecoTkToSeedTkMapDumbCMSSW(const TrackExtraVec& recoextras,
                                                       const TrackExtraVec& seedextras,
                                                       TkIDToTkIDMap& recoToSeedMap) {
    for (int itrack = 0; itrack < (int)recoextras.size(); itrack++) {
      const auto reco_seedID = recoextras[itrack].seedID();
      for (int jtrack = 0; jtrack < (int)seedextras.size(); jtrack++) {
        const auto seed_seedID = seedextras[jtrack].seedID();
        if (reco_seedID == seed_seedID) {
          recoToSeedMap[itrack] = jtrack;
          break;
        }
      }
    }
  }

  void TTreeValidation::setTrackScoresDumbCMSSW(Event& ev) {
    auto& seedtracks = ev.seedTracks_;
    auto& candtracks = ev.candidateTracks_;
    auto& fittracks = ev.fitTracks_;
    auto score_calc = IterationConfig::get_track_scorer("default");

    // first compute score...
    for (auto& seedtrack : seedtracks) {
      seedtrack.setScore(getScoreCand(score_calc, seedtrack));
    }

    // ...then use map to set seed type to for build/fit tracks and compute scores
    for (const auto& candToSeedPair : candToSeedMapDumbCMSSW_) {
      auto& candtrack = candtracks[candToSeedPair.first];

      candtrack.setScore(getScoreCand(score_calc, candtrack));
    }
    for (const auto& fitToSeedPair : fitToSeedMapDumbCMSSW_) {
      auto& fittrack = fittracks[fitToSeedPair.first];

      fittrack.setScore(getScoreCand(score_calc, fittrack));
    }
  }

  void TTreeValidation::storeSeedAndMCID(Event& ev) {
    const auto& buildtracks = ev.candidateTracks_;
    auto& buildextras = ev.candidateTracksExtra_;

    const auto& fittracks = ev.fitTracks_;
    auto& fitextras = ev.fitTracksExtra_;

    const auto& cmsswtracks = ev.cmsswTracks_;
    auto& cmsswextras = ev.cmsswTracksExtra_;

    // first set candidate tracks, use as base for fittracks
    int newlabel = -1;
    for (int itrack = 0; itrack < (int)buildtracks.size(); itrack++) {
      auto& extra = buildextras[itrack];
      const int seedID = extra.seedID();

      extra.setmcTrackID(seedID);

      if (seedToCmsswMap_.count(seedID)) {
        extra.setseedID(seedToCmsswMap_[seedID]);
        if (Config::cmsswMatchingFW == labelBased || Config::cmsswMatchingBK == labelBased) {
          for (int ctrack = 0; ctrack < (int)cmsswextras.size(); ctrack++) {
            if (cmsswextras[ctrack].seedID() == extra.seedID()) {
              buildToCmsswMap_[itrack] = cmsswtracks[ctrack].label();  // cmsstracks[ctrack].label() == ctrack!
              break;
            }
          }
        }
      } else {
        extra.setseedID(--newlabel);
      }
    }

    // set according to candidate tracks for fit tracks through map
    for (int itrack = 0; itrack < (int)fittracks.size(); itrack++) {
      auto& extra = fitextras[itrack];

      extra.setmcTrackID(buildextras[fitToBuildMap_[itrack]].mcTrackID());
      extra.setseedID(buildextras[fitToBuildMap_[itrack]].seedID());
    }
  }

  void TTreeValidation::setupCMSSWMatching(const Event& ev,
                                           RedTrackVec& reducedCMSSW,
                                           LayIdxIDVecMapMap& cmsswHitIDMap) {
    // get the tracks + hits + extras
    const auto& layerhits = ev.layerHits_;
    const auto& cmsswtracks = ev.cmsswTracks_;
    auto& cmsswextras = ev.cmsswTracksExtra_;
    const auto& seedtracks = ev.seedTracks_;

    // resize accordingly
    reducedCMSSW.resize(cmsswtracks.size());

    for (int itrack = 0; itrack < (int)cmsswtracks.size(); itrack++) {
      // get the needed tracks and extras
      auto& cmsswextra = cmsswextras[itrack];
      const auto& cmsswtrack = cmsswtracks[itrack];
      const auto& seedtrack = seedtracks[cmsswToSeedMap_[cmsswtrack.label()]];  // since cmsswtrack.label() == itrack

      // set seed hits!
      cmsswextra.findMatchingSeedHits(cmsswtrack, seedtrack, layerhits);

      // get tmp vars
      const auto seedID = cmsswextra.seedID();
      const auto& params = cmsswtrack.parameters();
      SVector2 tmpv(params[3], params[5]);

      HitLayerMap tmpmap;
      for (int ihit = 0; ihit < cmsswtrack.nTotalHits(); ihit++) {
        const int lyr = cmsswtrack.getHitLyr(ihit);
        const int idx = cmsswtrack.getHitIdx(ihit);

        // don't bother with storing seed layers in reduced cmssw
        if (cmsswextra.isSeedHit(lyr, idx))
          continue;

        if (lyr >= 0 && idx >= 0) {
          tmpmap[lyr].push_back(idx);
          cmsswHitIDMap[lyr][idx].push_back(cmsswtrack.label());
        }
      }

      // index inside object is label (as cmsswtracks are now aligned)
      reducedCMSSW[itrack] = ReducedTrack(cmsswtrack.label(), seedID, tmpv, cmsswtrack.momPhi(), tmpmap);
    }
  }

  int TTreeValidation::getLastFoundHit(const int trackMCHitID, const int mcTrackID, const Event& ev) {
    int mcHitID = -1;
    if (ev.simHitsInfo_[trackMCHitID].mcTrackID() == mcTrackID) {
      mcHitID = trackMCHitID;
    } else {
      mcHitID = ev.simTracks_[mcTrackID].getMCHitIDFromLayer(ev.layerHits_, ev.simHitsInfo_[trackMCHitID].layer());
    }
    return mcHitID;
  }

  int TTreeValidation::getMaskAssignment(const int refID) {
    // initialize
    auto refmask = -99;

    if (refID >= 0)  // seed track matched to seed and sim
    {
      refmask = 1;  // matched track to sim
    } else if (refID == -10) {
      refmask = -2;
    } else {
      if (Config::inclusiveShorts)  // only used by standard simval!
      {
        if (refID == -1 || refID == -5 || refID == -8 || refID == -9) {
          refmask = 0;
        } else if (refID == -2) {
          refmask = 2;
        } else  // mcID == -3,-4,-6,-7
        {
          refmask = -1;
        }
      } else  // only count long tracks (in mtvLike: all reco tracks are counted!)
      {
        if (refID == -1 || refID == -9) {
          refmask = 0;
        } else if (Config::mtvLikeValidation && refID == -4) {
          refmask = 2;
        } else  // mcID == -2,-3,-4,-5,-6,-7,-8: standard simval
        {
          refmask = -1;
        }
      }
    }  // end check over not matched

    return refmask;
  }

  void TTreeValidation::resetFitBranches() {
    for (int ilayer = 0; ilayer < ntotallayers_fit_; ++ilayer) {
      z_prop_fit_[ilayer] = -1000.f;
      ez_prop_fit_[ilayer] = -1000.f;
      z_hit_fit_[ilayer] = -1000.f;
      ez_hit_fit_[ilayer] = -1000.f;
      z_sim_fit_[ilayer] = -1000.f;
      ez_sim_fit_[ilayer] = -1000.f;

      pphi_prop_fit_[ilayer] = -1000.f;
      epphi_prop_fit_[ilayer] = -1000.f;
      pphi_hit_fit_[ilayer] = -1000.f;
      epphi_hit_fit_[ilayer] = -1000.f;
      pphi_sim_fit_[ilayer] = -1000.f;
      epphi_sim_fit_[ilayer] = -1000.f;

      pt_up_fit_[ilayer] = -1000.f;
      ept_up_fit_[ilayer] = -1000.f;
      pt_sim_fit_[ilayer] = -1000.f;
      ept_sim_fit_[ilayer] = -1000.f;

      mphi_up_fit_[ilayer] = -1000.f;
      emphi_up_fit_[ilayer] = -1000.f;
      mphi_sim_fit_[ilayer] = -1000.f;
      emphi_sim_fit_[ilayer] = -1000.f;

      meta_up_fit_[ilayer] = -1000.f;
      emeta_up_fit_[ilayer] = -1000.f;
      meta_sim_fit_[ilayer] = -1000.f;
      emeta_sim_fit_[ilayer] = -1000.f;
    }
  }

  void TTreeValidation::fillFitTree(const Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);

    evtid_fit_ = ev.evtID();
    const auto& simtracks = ev.simTracks_;
    const auto& layerhits = ev.layerHits_;
    const auto& simtrackstates = ev.simTrackStates_;

    for (auto&& fitvalmapmap : fitValTkMapMap_) {
      TTreeValidation::resetFitBranches();

      tkid_fit_ = fitvalmapmap.first;  // seed id (label) is the same as the mcID

      const auto& simtrack = simtracks[tkid_fit_];
      const auto& fitvalmap = fitvalmapmap.second;
      for (int ilayer = 0; ilayer < ntotallayers_fit_; ++ilayer) {
        if (fitvalmap.count(ilayer)) {
          const auto& hit = layerhits[ilayer][simtrack.getHitIdx(ilayer)];
          const auto& initTS = simtrackstates.at(hit.mcHitID());
          const auto& fitval = fitvalmap.at(ilayer);

          z_hit_fit_[ilayer] = hit.z();
          ez_hit_fit_[ilayer] = std::sqrt(hit.ezz());
          z_sim_fit_[ilayer] = initTS.z();
          ez_sim_fit_[ilayer] = initTS.ezz();
          z_prop_fit_[ilayer] = fitval.ppz;
          ez_prop_fit_[ilayer] = fitval.eppz;

          pphi_hit_fit_[ilayer] = hit.phi();
          epphi_hit_fit_[ilayer] = std::sqrt(hit.ephi());
          pphi_sim_fit_[ilayer] = initTS.posPhi();
          epphi_sim_fit_[ilayer] = initTS.eposPhi();
          pphi_prop_fit_[ilayer] = fitval.ppphi;
          epphi_prop_fit_[ilayer] = fitval.eppphi;

          pt_up_fit_[ilayer] = fitval.upt;
          ept_up_fit_[ilayer] = fitval.eupt;
          pt_sim_fit_[ilayer] = initTS.pT();
          ept_sim_fit_[ilayer] = initTS.epT();

          mphi_up_fit_[ilayer] = fitval.umphi;
          emphi_up_fit_[ilayer] = fitval.eumphi;
          mphi_sim_fit_[ilayer] = initTS.momPhi();
          emphi_sim_fit_[ilayer] = initTS.emomPhi();

          meta_up_fit_[ilayer] = fitval.umeta;
          emeta_up_fit_[ilayer] = fitval.eumeta;
          meta_sim_fit_[ilayer] = initTS.momEta();
          emeta_sim_fit_[ilayer] = initTS.emomEta();
        }
      }
      fittree_->Fill();
    }
  }

  void TTreeValidation::fillFullHitInfo(const Event& ev,
                                        const Track& track,
                                        std::vector<int>& lyrs,
                                        std::vector<int>& idxs,
                                        std::vector<int>& mcTkIDs,
                                        std::vector<float>& xs,
                                        std::vector<float>& ys,
                                        std::vector<float>& zs) {
    // get event info
    const auto& layerHits = ev.layerHits_;
    const auto& simHitsInfo = ev.simHitsInfo_;

    // resize vectors
    const auto nTotalHits = track.nTotalHits();
    lyrs.resize(nTotalHits);
    idxs.resize(nTotalHits);
    mcTkIDs.resize(nTotalHits, -99);
    xs.resize(nTotalHits, -9999.f);
    ys.resize(nTotalHits, -9999.f);
    zs.resize(nTotalHits, -9999.f);

    // loop over size of total hits
    for (auto ihit = 0; ihit < nTotalHits; ihit++) {
      const auto lyr = track.getHitLyr(ihit);
      const auto idx = track.getHitIdx(ihit);

      lyrs[ihit] = lyr;
      idxs[ihit] = idx;

      if (lyr < 0)
        continue;
      if (idx < 0)
        continue;

      const auto& hit = layerHits[lyr][idx];
      mcTkIDs[ihit] = hit.mcTrackID(simHitsInfo);
      xs[ihit] = hit.x();
      ys[ihit] = hit.y();
      zs[ihit] = hit.z();
    }
  }

  void TTreeValidation::fillMinHitInfo(const Track& track, std::vector<int>& lyrs, std::vector<int>& idxs) {
    for (int ihit = 0; ihit < track.nTotalHits(); ihit++) {
      lyrs.emplace_back(track.getHitLyr(ihit));
      idxs.emplace_back(track.getHitIdx(ihit));
    }
  }

  void TTreeValidation::fillEfficiencyTree(const Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);

    const auto ievt = ev.evtID();
    const auto& evt_sim_tracks = ev.simTracks_;
    const auto& evt_seed_tracks = ev.seedTracks_;
    const auto& evt_seed_extras = ev.seedTracksExtra_;
    const auto& evt_build_tracks = ev.candidateTracks_;
    const auto& evt_build_extras = ev.candidateTracksExtra_;
    const auto& evt_fit_tracks = ev.fitTracks_;
    const auto& evt_fit_extras = ev.fitTracksExtra_;
    const auto& evt_layer_hits = ev.layerHits_;
    const auto& evt_sim_trackstates = ev.simTrackStates_;

    unsigned int count = 0;
    for (const auto& simtrack : evt_sim_tracks) {
      // clear the branches first
      if (Config::keepHitInfo) {
        hitlyrs_mc_eff_.clear();
        hitlyrs_seed_eff_.clear();
        hitlyrs_build_eff_.clear();
        hitlyrs_fit_eff_.clear();

        hitidxs_mc_eff_.clear();
        hitidxs_seed_eff_.clear();
        hitidxs_build_eff_.clear();
        hitidxs_fit_eff_.clear();

        hitmcTkIDs_mc_eff_.clear();
        hitmcTkIDs_seed_eff_.clear();
        hitmcTkIDs_build_eff_.clear();
        hitmcTkIDs_fit_eff_.clear();

        hitxs_mc_eff_.clear();
        hitxs_seed_eff_.clear();
        hitxs_build_eff_.clear();
        hitxs_fit_eff_.clear();

        hitys_mc_eff_.clear();
        hitys_seed_eff_.clear();
        hitys_build_eff_.clear();
        hitys_fit_eff_.clear();

        hitzs_mc_eff_.clear();
        hitzs_seed_eff_.clear();
        hitzs_build_eff_.clear();
        hitzs_fit_eff_.clear();
      }

      evtID_eff_ = ievt;
      mcID_eff_ = simtrack.label();

      // generated values
      x_mc_gen_eff_ = simtrack.x();
      y_mc_gen_eff_ = simtrack.y();
      z_mc_gen_eff_ = simtrack.z();

      pt_mc_gen_eff_ = simtrack.pT();
      phi_mc_gen_eff_ = simtrack.momPhi();
      eta_mc_gen_eff_ = simtrack.momEta();
      nHits_mc_eff_ = simtrack.nFoundHits();  // could be that the sim track skips layers!
      nLayers_mc_eff_ = simtrack.nUniqueLayers();
      lastlyr_mc_eff_ = simtrack.getLastFoundHitLyr();

      itermask_seed_eff_ = 0;
      itermask_build_eff_ = 0;
      itermask_fit_eff_ = 0;
      iterduplmask_seed_eff_ = 0;
      iterduplmask_build_eff_ = 0;
      iterduplmask_fit_eff_ = 0;
      algo_seed_eff_ = 0;

      if (Config::mtvRequireSeeds) {
        for (auto aa : ev.simTracksExtra_[count].seedAlgos()) {
          algo_seed_eff_ = (algo_seed_eff_ | (1 << aa));
        }
      }
      count++;

      // hit indices
      if (Config::keepHitInfo)
        TTreeValidation::fillFullHitInfo(ev,
                                         simtrack,
                                         hitlyrs_mc_eff_,
                                         hitidxs_mc_eff_,
                                         hitmcTkIDs_mc_eff_,
                                         hitxs_mc_eff_,
                                         hitys_mc_eff_,
                                         hitzs_mc_eff_);

      // matched seed track
      if (simToSeedMap_.count(mcID_eff_) &&
          simtrack
              .isFindable())  // recoToSim match : save best match with best score, i.e. simToSeedMap_[matched SimID][first element in vector]
      {
        for (unsigned int ii = 0; ii < simToSeedMap_[mcID_eff_].size(); ii++) {
          const int theAlgo = evt_seed_tracks[simToSeedMap_[mcID_eff_][ii]].algoint();
          if ((itermask_seed_eff_ >> theAlgo) & 1)
            iterduplmask_seed_eff_ = (iterduplmask_seed_eff_ | (1 << theAlgo));  //filled at the second time
          itermask_seed_eff_ = (itermask_seed_eff_ | (1 << theAlgo));
        }
        const auto& seedtrack =
            evt_seed_tracks[simToSeedMap_[mcID_eff_][0]];            // returns seedTrack best matched to sim track
        const auto& seedextra = evt_seed_extras[seedtrack.label()];  // returns track extra best aligned with seed track
        mcmask_seed_eff_ = 1;                                        // quick logic for matched

        seedID_seed_eff_ = seedextra.seedID();

        // use this to access correct sim track layer params
        const int mcHitID =
            TTreeValidation::getLastFoundHit(seedtrack.getLastFoundMCHitID(evt_layer_hits), mcID_eff_, ev);
        if (mcHitID >= 0 && Config::readSimTrackStates) {
          const TrackState& initLayTS = evt_sim_trackstates[mcHitID];

          pt_mc_seed_eff_ = initLayTS.pT();
          phi_mc_seed_eff_ = initLayTS.momPhi();
          eta_mc_seed_eff_ = initLayTS.momEta();
          helixchi2_seed_eff_ = computeHelixChi2(initLayTS.parameters, seedtrack.parameters(), seedtrack.errors());

          mcTSmask_seed_eff_ = 1;
        } else if (Config::tryToSaveSimInfo)  // can enter this block if: we actually read sim track states, but could not find the mchit OR we chose not to read the sim track states
        {
          // reuse info already set
          pt_mc_seed_eff_ = pt_mc_gen_eff_;
          phi_mc_seed_eff_ = phi_mc_gen_eff_;
          eta_mc_seed_eff_ = eta_mc_gen_eff_;
          helixchi2_seed_eff_ = computeHelixChi2(simtrack.parameters(), seedtrack.parameters(), seedtrack.errors());

          mcTSmask_seed_eff_ = 0;
        } else {
          pt_mc_seed_eff_ = -101;
          phi_mc_seed_eff_ = -101;
          eta_mc_seed_eff_ = -101;
          helixchi2_seed_eff_ = -101;

          mcTSmask_seed_eff_ = -2;
        }

        // last hit info
        const Hit& lasthit = evt_layer_hits[seedtrack.getLastFoundHitLyr()][seedtrack.getLastFoundHitIdx()];
        xhit_seed_eff_ = lasthit.x();
        yhit_seed_eff_ = lasthit.y();
        zhit_seed_eff_ = lasthit.z();

        pt_seed_eff_ = seedtrack.pT();
        ept_seed_eff_ = seedtrack.epT();
        phi_seed_eff_ = seedtrack.momPhi();
        ephi_seed_eff_ = seedtrack.emomPhi();
        eta_seed_eff_ = seedtrack.momEta();
        eeta_seed_eff_ = seedtrack.emomEta();

        // rest of mc info
        nHits_seed_eff_ = seedtrack.nFoundHits();
        nLayers_seed_eff_ = seedtrack.nUniqueLayers();
        nHitsMatched_seed_eff_ = seedextra.nHitsMatched();
        fracHitsMatched_seed_eff_ = seedextra.fracHitsMatched();
        lastlyr_seed_eff_ = seedtrack.getLastFoundHitLyr();

        // swim dphi
        dphi_seed_eff_ = seedextra.dPhi();

        // quality info
        hitchi2_seed_eff_ = seedtrack.chi2();  // currently not being used
        score_seed_eff_ = seedtrack.score();   // currently a constant by definition

        duplmask_seed_eff_ = seedextra.isDuplicate();
        nTkMatches_seed_eff_ = simToSeedMap_[mcID_eff_].size();  // n reco matches to this sim track.

        // hit indices
        if (Config::keepHitInfo)
          TTreeValidation::fillFullHitInfo(ev,
                                           seedtrack,
                                           hitlyrs_seed_eff_,
                                           hitidxs_seed_eff_,
                                           hitmcTkIDs_seed_eff_,
                                           hitxs_seed_eff_,
                                           hitys_seed_eff_,
                                           hitzs_seed_eff_);
      } else  // unmatched simTracks ... put -99 for all reco values to denote unmatched
      {
        mcmask_seed_eff_ = (simtrack.isFindable() ? 0 : -1);  // quick logic for not matched

        seedID_seed_eff_ = -99;

        pt_mc_seed_eff_ = -99;
        phi_mc_seed_eff_ = -99;
        eta_mc_seed_eff_ = -99;
        helixchi2_seed_eff_ = -99;

        mcTSmask_seed_eff_ = -1;  // mask means unmatched sim track

        xhit_seed_eff_ = -2000;
        yhit_seed_eff_ = -2000;
        zhit_seed_eff_ = -2000;

        pt_seed_eff_ = -99;
        ept_seed_eff_ = -99;
        phi_seed_eff_ = -99;
        ephi_seed_eff_ = -99;
        eta_seed_eff_ = -99;
        eeta_seed_eff_ = -99;

        nHits_seed_eff_ = -99;
        nLayers_seed_eff_ = -99;
        nHitsMatched_seed_eff_ = -99;
        fracHitsMatched_seed_eff_ = -99;
        lastlyr_seed_eff_ = -99;

        dphi_seed_eff_ = -99;

        hitchi2_seed_eff_ = -99;
        score_seed_eff_ = -17000;

        duplmask_seed_eff_ = -1;     // mask means unmatched sim track
        nTkMatches_seed_eff_ = -99;  // unmatched
      }

      // matched build track
      if (simToBuildMap_.count(mcID_eff_) &&
          simtrack
              .isFindable())  // recoToSim match : save best match with best score i.e. simToBuildMap_[matched SimID][first element in vector]
      {
        for (unsigned int ii = 0; ii < simToBuildMap_[mcID_eff_].size(); ii++) {
          const int theAlgo = evt_build_tracks[simToBuildMap_[mcID_eff_][ii]].algoint();
          if ((itermask_build_eff_ >> theAlgo) & 1)
            iterduplmask_build_eff_ = (iterduplmask_build_eff_ | (1 << theAlgo));  //filled at the second time
          itermask_build_eff_ = (itermask_build_eff_ | (1 << theAlgo));
        }
        const auto& buildtrack =
            evt_build_tracks[simToBuildMap_[mcID_eff_][0]];  // returns buildTrack best matched to sim track
        const auto& buildextra =
            evt_build_extras[buildtrack.label()];  // returns track extra best aligned with build track
        mcmask_build_eff_ = 1;                     // quick logic for matched

        seedID_build_eff_ = buildextra.seedID();

        // use this to access correct sim track layer params
        const int mcHitID =
            TTreeValidation::getLastFoundHit(buildtrack.getLastFoundMCHitID(evt_layer_hits), mcID_eff_, ev);
        if (mcHitID >= 0 && Config::readSimTrackStates) {
          const TrackState& initLayTS = evt_sim_trackstates[mcHitID];

          pt_mc_build_eff_ = initLayTS.pT();
          phi_mc_build_eff_ = initLayTS.momPhi();
          eta_mc_build_eff_ = initLayTS.momEta();
          helixchi2_build_eff_ = computeHelixChi2(initLayTS.parameters, buildtrack.parameters(), buildtrack.errors());

          mcTSmask_build_eff_ = 1;
        } else if (Config::tryToSaveSimInfo)  // can enter this block if: we actually read sim track states, but could not find the mchit OR we chose not to read the sim track states
        {
          // reuse info already set
          pt_mc_build_eff_ = pt_mc_gen_eff_;
          phi_mc_build_eff_ = phi_mc_gen_eff_;
          eta_mc_build_eff_ = eta_mc_gen_eff_;
          helixchi2_build_eff_ = computeHelixChi2(simtrack.parameters(), buildtrack.parameters(), buildtrack.errors());

          mcTSmask_build_eff_ = 0;
        } else {
          pt_mc_build_eff_ = -101;
          phi_mc_build_eff_ = -101;
          eta_mc_build_eff_ = -101;
          helixchi2_build_eff_ = -101;

          mcTSmask_build_eff_ = -2;
        }

        // last hit info
        const Hit& lasthit = evt_layer_hits[buildtrack.getLastFoundHitLyr()][buildtrack.getLastFoundHitIdx()];
        xhit_build_eff_ = lasthit.x();
        yhit_build_eff_ = lasthit.y();
        zhit_build_eff_ = lasthit.z();

        pt_build_eff_ = buildtrack.pT();
        ept_build_eff_ = buildtrack.epT();
        phi_build_eff_ = buildtrack.momPhi();
        ephi_build_eff_ = buildtrack.emomPhi();
        eta_build_eff_ = buildtrack.momEta();
        eeta_build_eff_ = buildtrack.emomEta();

        nHits_build_eff_ = buildtrack.nFoundHits();
        nLayers_build_eff_ = buildtrack.nUniqueLayers();
        nHitsMatched_build_eff_ = buildextra.nHitsMatched();
        fracHitsMatched_build_eff_ = buildextra.fracHitsMatched();
        lastlyr_build_eff_ = buildtrack.getLastFoundHitLyr();

        // swim dphi
        dphi_build_eff_ = buildextra.dPhi();

        // quality info
        hitchi2_build_eff_ = buildtrack.chi2();
        score_build_eff_ = buildtrack.score();

        duplmask_build_eff_ = buildextra.isDuplicate();
        nTkMatches_build_eff_ = simToBuildMap_[mcID_eff_].size();  // n reco matches to this sim track.

        // hit indices
        if (Config::keepHitInfo)
          TTreeValidation::fillFullHitInfo(ev,
                                           buildtrack,
                                           hitlyrs_build_eff_,
                                           hitidxs_build_eff_,
                                           hitmcTkIDs_build_eff_,
                                           hitxs_build_eff_,
                                           hitys_build_eff_,
                                           hitzs_build_eff_);
      } else  // unmatched simTracks ... put -99 for all reco values to denote unmatched
      {
        mcmask_build_eff_ = (simtrack.isFindable() ? 0 : -1);  // quick logic for not matched

        seedID_build_eff_ = -99;

        pt_mc_build_eff_ = -99;
        phi_mc_build_eff_ = -99;
        eta_mc_build_eff_ = -99;
        helixchi2_build_eff_ = -99;

        mcTSmask_build_eff_ = -1;

        xhit_build_eff_ = -2000;
        yhit_build_eff_ = -2000;
        zhit_build_eff_ = -2000;

        pt_build_eff_ = -99;
        ept_build_eff_ = -99;
        phi_build_eff_ = -99;
        ephi_build_eff_ = -99;
        eta_build_eff_ = -99;
        eeta_build_eff_ = -99;

        nHits_build_eff_ = -99;
        nLayers_build_eff_ = -99;
        nHitsMatched_build_eff_ = -99;
        fracHitsMatched_build_eff_ = -99;
        lastlyr_build_eff_ = -99;

        dphi_build_eff_ = -99;

        hitchi2_build_eff_ = -99;
        score_build_eff_ = -17000;

        duplmask_build_eff_ = -1;     // mask means unmatched sim track
        nTkMatches_build_eff_ = -99;  // unmatched
      }

      // matched fit track
      if (simToFitMap_.count(mcID_eff_) &&
          simtrack
              .isFindable())  // recoToSim match : save best match with best score i.e. simToFitMap_[matched SimID][first element in vector]
      {
        for (unsigned int ii = 0; ii < simToFitMap_[mcID_eff_].size(); ii++) {
          const int theAlgo = evt_fit_tracks[simToFitMap_[mcID_eff_][ii]].algoint();
          if ((itermask_fit_eff_ >> theAlgo) & 1)
            iterduplmask_fit_eff_ = (iterduplmask_fit_eff_ | (1 << theAlgo));  //filled at the second time
          itermask_fit_eff_ = (itermask_fit_eff_ | (1 << theAlgo));
        }
        const auto& fittrack =
            evt_fit_tracks[simToFitMap_[mcID_eff_][0]];           // returns fitTrack best matched to sim track
        const auto& fitextra = evt_fit_extras[fittrack.label()];  // returns track extra best aligned with fit track
        mcmask_fit_eff_ = 1;                                      // quick logic for matched

        seedID_fit_eff_ = fitextra.seedID();

        // use this to access correct sim track layer params
        const int mcHitID =
            TTreeValidation::getLastFoundHit(fittrack.getLastFoundMCHitID(evt_layer_hits), mcID_eff_, ev);
        if (mcHitID >= 0 && Config::readSimTrackStates) {
          const TrackState& initLayTS = evt_sim_trackstates[mcHitID];

          pt_mc_fit_eff_ = initLayTS.pT();
          phi_mc_fit_eff_ = initLayTS.momPhi();
          eta_mc_fit_eff_ = initLayTS.momEta();
          helixchi2_fit_eff_ = computeHelixChi2(initLayTS.parameters, fittrack.parameters(), fittrack.errors());

          mcTSmask_fit_eff_ = 1;
        } else if (Config::tryToSaveSimInfo)  // can enter this block if: we actually read sim track states, but could not find the mchit OR we chose not to read the sim track states
        {
          // reuse info already set
          pt_mc_fit_eff_ = pt_mc_gen_eff_;
          phi_mc_fit_eff_ = phi_mc_gen_eff_;
          eta_mc_fit_eff_ = eta_mc_gen_eff_;
          helixchi2_fit_eff_ = computeHelixChi2(simtrack.parameters(), fittrack.parameters(), fittrack.errors());

          mcTSmask_fit_eff_ = 0;
        } else {
          pt_mc_fit_eff_ = -101;
          phi_mc_fit_eff_ = -101;
          eta_mc_fit_eff_ = -101;
          helixchi2_fit_eff_ = -101;

          mcTSmask_fit_eff_ = -2;
        }

        // last hit info
        const Hit& lasthit = evt_layer_hits[fittrack.getLastFoundHitLyr()][fittrack.getLastFoundHitIdx()];
        xhit_fit_eff_ = lasthit.x();
        yhit_fit_eff_ = lasthit.y();
        zhit_fit_eff_ = lasthit.z();

        pt_fit_eff_ = fittrack.pT();
        ept_fit_eff_ = fittrack.epT();
        phi_fit_eff_ = fittrack.momPhi();
        ephi_fit_eff_ = fittrack.emomPhi();
        eta_fit_eff_ = fittrack.momEta();
        eeta_fit_eff_ = fittrack.emomEta();

        // rest of mc info
        nHits_fit_eff_ = fittrack.nFoundHits();
        nLayers_fit_eff_ = fittrack.nUniqueLayers();
        nHitsMatched_fit_eff_ = fitextra.nHitsMatched();
        fracHitsMatched_fit_eff_ = fitextra.fracHitsMatched();
        lastlyr_fit_eff_ = fittrack.getLastFoundHitLyr();

        // swim dphi
        dphi_fit_eff_ = fitextra.dPhi();

        // quality info
        hitchi2_fit_eff_ = fittrack.chi2();  // -10 when not used
        score_fit_eff_ = fittrack.score();

        duplmask_fit_eff_ = fitextra.isDuplicate();
        nTkMatches_fit_eff_ = simToFitMap_[mcID_eff_].size();  // n reco matches to this sim track.

        // hit indices
        if (Config::keepHitInfo)
          TTreeValidation::fillFullHitInfo(ev,
                                           fittrack,
                                           hitlyrs_fit_eff_,
                                           hitidxs_fit_eff_,
                                           hitmcTkIDs_fit_eff_,
                                           hitxs_fit_eff_,
                                           hitys_fit_eff_,
                                           hitzs_fit_eff_);
      } else  // unmatched simTracks ... put -99 for all reco values to denote unmatched
      {
        mcmask_fit_eff_ = (simtrack.isFindable() ? 0 : -1);  // quick logic for not matched

        seedID_fit_eff_ = -99;

        pt_mc_fit_eff_ = -99;
        phi_mc_fit_eff_ = -99;
        eta_mc_fit_eff_ = -99;
        helixchi2_fit_eff_ = -99;

        mcTSmask_fit_eff_ = -1;

        xhit_fit_eff_ = -2000;
        yhit_fit_eff_ = -2000;
        zhit_fit_eff_ = -2000;

        pt_fit_eff_ = -99;
        ept_fit_eff_ = -99;
        phi_fit_eff_ = -99;
        ephi_fit_eff_ = -99;
        eta_fit_eff_ = -99;
        eeta_fit_eff_ = -99;

        nHits_fit_eff_ = -99;
        nLayers_fit_eff_ = -99;
        nHitsMatched_fit_eff_ = -99;
        fracHitsMatched_fit_eff_ = -99;
        lastlyr_fit_eff_ = -99;

        dphi_fit_eff_ = -99;

        hitchi2_fit_eff_ = -99;
        score_fit_eff_ = -17000;

        duplmask_fit_eff_ = -1;     // mask means unmatched sim track
        nTkMatches_fit_eff_ = -99;  // unmatched
      }

      efftree_->Fill();  // fill it once per sim track!
    }
  }

  void TTreeValidation::fillFakeRateTree(const Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);

    const auto ievt = ev.evtID();
    const auto& evt_sim_tracks =
        ev.simTracks_;  // store sim info at that final layer!!! --> gen info stored only in eff tree
    const auto& evt_seed_tracks = ev.seedTracks_;
    const auto& evt_seed_extras = ev.seedTracksExtra_;
    const auto& evt_build_tracks = ev.candidateTracks_;
    const auto& evt_build_extras = ev.candidateTracksExtra_;
    const auto& evt_fit_tracks = ev.fitTracks_;
    const auto& evt_fit_extras = ev.fitTracksExtra_;
    const auto& evt_layer_hits = ev.layerHits_;
    const auto& evt_sim_trackstates = ev.simTrackStates_;

    for (const auto& seedtrack : evt_seed_tracks) {
      if (Config::keepHitInfo) {
        hitlyrs_seed_FR_.clear();
        hitlyrs_mc_seed_FR_.clear();
        hitlyrs_build_FR_.clear();
        hitlyrs_mc_build_FR_.clear();
        hitlyrs_fit_FR_.clear();
        hitlyrs_mc_fit_FR_.clear();

        hitidxs_seed_FR_.clear();
        hitidxs_mc_seed_FR_.clear();
        hitidxs_build_FR_.clear();
        hitidxs_mc_build_FR_.clear();
        hitidxs_fit_FR_.clear();
        hitidxs_mc_fit_FR_.clear();

        hitmcTkIDs_seed_FR_.clear();
        hitmcTkIDs_mc_seed_FR_.clear();
        hitmcTkIDs_build_FR_.clear();
        hitmcTkIDs_mc_build_FR_.clear();
        hitmcTkIDs_fit_FR_.clear();
        hitmcTkIDs_mc_fit_FR_.clear();

        hitxs_seed_FR_.clear();
        hitxs_mc_seed_FR_.clear();
        hitxs_build_FR_.clear();
        hitxs_mc_build_FR_.clear();
        hitxs_fit_FR_.clear();
        hitxs_mc_fit_FR_.clear();

        hitys_seed_FR_.clear();
        hitys_mc_seed_FR_.clear();
        hitys_build_FR_.clear();
        hitys_mc_build_FR_.clear();
        hitys_fit_FR_.clear();
        hitys_mc_fit_FR_.clear();

        hitzs_seed_FR_.clear();
        hitzs_mc_seed_FR_.clear();
        hitzs_build_FR_.clear();
        hitzs_mc_build_FR_.clear();
        hitzs_fit_FR_.clear();
        hitzs_mc_fit_FR_.clear();
      }

      evtID_FR_ = ievt;

      // seed info
      const auto& seedextra = evt_seed_extras[seedtrack.label()];
      seedID_FR_ = seedextra.seedID();
      seedmask_seed_FR_ =
          1;  // automatically set to 1, because at the moment no cuts on seeds after conformal+KF fit.  seed triplets filtered by RZ chi2 before fitting.

      // last hit info
      // const Hit& lasthit = evt_layer_hits[seedtrack.getLastFoundHitLyr()][seedtrack.getLastFoundHitIdx()];
      xhit_seed_FR_ = 0;  //lasthit.x();
      yhit_seed_FR_ = 0;  //lasthit.y();
      zhit_seed_FR_ = 0;  //lasthit.z();

      pt_seed_FR_ = seedtrack.pT();
      ept_seed_FR_ = seedtrack.epT();
      phi_seed_FR_ = seedtrack.momPhi();
      ephi_seed_FR_ = seedtrack.emomPhi();
      eta_seed_FR_ = seedtrack.momEta();
      eeta_seed_FR_ = seedtrack.emomEta();

      nHits_seed_FR_ = seedtrack.nFoundHits();
      nLayers_seed_FR_ = seedtrack.nUniqueLayers();
      nHitsMatched_seed_FR_ = seedextra.nHitsMatched();
      fracHitsMatched_seed_FR_ = seedextra.fracHitsMatched();
      lastlyr_seed_FR_ = seedtrack.getLastFoundHitLyr();

      algorithm_FR_ = seedtrack.algoint();

      // swim dphi
      dphi_seed_FR_ = seedextra.dPhi();

      // quality info
      hitchi2_seed_FR_ = seedtrack.chi2();  //--> not currently used
      score_seed_FR_ = seedtrack.score();

      if (Config::keepHitInfo)
        TTreeValidation::fillFullHitInfo(ev,
                                         seedtrack,
                                         hitlyrs_seed_FR_,
                                         hitidxs_seed_FR_,
                                         hitmcTkIDs_seed_FR_,
                                         hitxs_seed_FR_,
                                         hitys_seed_FR_,
                                         hitzs_seed_FR_);

      // sim info for seed track
      mcID_seed_FR_ = seedextra.mcTrackID();
      mcmask_seed_FR_ = TTreeValidation::getMaskAssignment(mcID_seed_FR_);

      if (mcmask_seed_FR_ == 1)  // matched track to sim
      {
        const auto& simtrack = evt_sim_tracks[mcID_seed_FR_];

        const int mcHitID =
            TTreeValidation::getLastFoundHit(seedtrack.getLastFoundMCHitID(evt_layer_hits), mcID_seed_FR_, ev);
        if (mcHitID >= 0 && Config::readSimTrackStates) {
          const TrackState& initLayTS = evt_sim_trackstates[mcHitID];
          pt_mc_seed_FR_ = initLayTS.pT();
          phi_mc_seed_FR_ = initLayTS.momPhi();
          eta_mc_seed_FR_ = initLayTS.momEta();
          helixchi2_seed_FR_ = computeHelixChi2(initLayTS.parameters, seedtrack.parameters(), seedtrack.errors());

          mcTSmask_seed_FR_ = 1;
        } else if (Config::tryToSaveSimInfo) {
          pt_mc_seed_FR_ = simtrack.pT();
          phi_mc_seed_FR_ = simtrack.momPhi();
          eta_mc_seed_FR_ = simtrack.momEta();
          helixchi2_seed_FR_ = computeHelixChi2(simtrack.parameters(), seedtrack.parameters(), seedtrack.errors());

          mcTSmask_seed_FR_ = 0;
        } else {
          pt_mc_seed_FR_ = -101;
          phi_mc_seed_FR_ = -101;
          eta_mc_seed_FR_ = -101;
          helixchi2_seed_FR_ = -101;

          mcTSmask_seed_FR_ = -2;
        }

        nHits_mc_seed_FR_ = simtrack.nFoundHits();
        nLayers_mc_seed_FR_ = simtrack.nUniqueLayers();
        lastlyr_mc_seed_FR_ = simtrack.getLastFoundHitLyr();

        duplmask_seed_FR_ = seedextra.isDuplicate();
        iTkMatches_seed_FR_ =
            seedextra
                .duplicateID();  // ith duplicate seed track, i = 0 "best" match, i > 0 "still matched, real reco, not as good as i-1 track"

        if (Config::keepHitInfo)
          TTreeValidation::fillFullHitInfo(ev,
                                           simtrack,
                                           hitlyrs_mc_seed_FR_,
                                           hitidxs_mc_seed_FR_,
                                           hitmcTkIDs_mc_seed_FR_,
                                           hitxs_mc_seed_FR_,
                                           hitys_mc_seed_FR_,
                                           hitzs_mc_seed_FR_);
      } else {
        // -99 for all sim info for reco tracks not associated to reco tracks
        pt_mc_seed_FR_ = -99;
        phi_mc_seed_FR_ = -99;
        eta_mc_seed_FR_ = -99;
        helixchi2_seed_FR_ = -99;

        mcTSmask_seed_FR_ = -1;

        nHits_mc_seed_FR_ = -99;
        nLayers_mc_seed_FR_ = -99;
        lastlyr_mc_seed_FR_ = -99;

        duplmask_seed_FR_ = -1;
        iTkMatches_seed_FR_ = -99;
      }

      //==========================//

      // fill build information if track still alive
      if (seedToBuildMap_.count(seedID_FR_)) {
        seedmask_build_FR_ = 1;  // quick logic

        const auto& buildtrack = evt_build_tracks[seedToBuildMap_[seedID_FR_]];
        const auto& buildextra = evt_build_extras[buildtrack.label()];

        // last hit info
        const Hit& lasthit = evt_layer_hits[buildtrack.getLastFoundHitLyr()][buildtrack.getLastFoundHitIdx()];
        xhit_build_FR_ = lasthit.x();
        yhit_build_FR_ = lasthit.y();
        zhit_build_FR_ = lasthit.z();

        pt_build_FR_ = buildtrack.pT();
        ept_build_FR_ = buildtrack.epT();
        phi_build_FR_ = buildtrack.momPhi();
        ephi_build_FR_ = buildtrack.emomPhi();
        eta_build_FR_ = buildtrack.momEta();
        eeta_build_FR_ = buildtrack.emomEta();

        nHits_build_FR_ = buildtrack.nFoundHits();
        nLayers_build_FR_ = buildtrack.nUniqueLayers();
        nHitsMatched_build_FR_ = buildextra.nHitsMatched();
        fracHitsMatched_build_FR_ = buildextra.fracHitsMatched();
        lastlyr_build_FR_ = buildtrack.getLastFoundHitLyr();

        // swim dphi
        dphi_build_FR_ = buildextra.dPhi();

        // quality info
        hitchi2_build_FR_ = buildtrack.chi2();
        score_build_FR_ = buildtrack.score();

        if (Config::keepHitInfo)
          TTreeValidation::fillFullHitInfo(ev,
                                           buildtrack,
                                           hitlyrs_build_FR_,
                                           hitidxs_build_FR_,
                                           hitmcTkIDs_build_FR_,
                                           hitxs_build_FR_,
                                           hitys_build_FR_,
                                           hitzs_build_FR_);

        // sim info for build track
        mcID_build_FR_ = buildextra.mcTrackID();
        mcmask_build_FR_ = TTreeValidation::getMaskAssignment(mcID_build_FR_);

        if (mcmask_build_FR_ == 1)  // build track matched to seed and sim
        {
          const auto& simtrack = evt_sim_tracks[mcID_build_FR_];

          const int mcHitID =
              TTreeValidation::getLastFoundHit(buildtrack.getLastFoundMCHitID(evt_layer_hits), mcID_build_FR_, ev);
          if (mcHitID >= 0 && Config::readSimTrackStates) {
            const TrackState& initLayTS = evt_sim_trackstates[mcHitID];
            pt_mc_build_FR_ = initLayTS.pT();
            phi_mc_build_FR_ = initLayTS.momPhi();
            eta_mc_build_FR_ = initLayTS.momEta();
            helixchi2_build_FR_ = computeHelixChi2(initLayTS.parameters, buildtrack.parameters(), buildtrack.errors());

            mcTSmask_build_FR_ = 1;
          } else if (Config::tryToSaveSimInfo) {
            pt_mc_build_FR_ = simtrack.pT();
            phi_mc_build_FR_ = simtrack.momPhi();
            eta_mc_build_FR_ = simtrack.momEta();
            helixchi2_build_FR_ = computeHelixChi2(simtrack.parameters(), buildtrack.parameters(), buildtrack.errors());

            mcTSmask_build_FR_ = 0;
          } else {
            pt_mc_build_FR_ = -101;
            phi_mc_build_FR_ = -101;
            eta_mc_build_FR_ = -101;
            helixchi2_build_FR_ = -101;

            mcTSmask_build_FR_ = -2;
          }

          nHits_mc_build_FR_ = simtrack.nFoundHits();
          nLayers_mc_build_FR_ = simtrack.nUniqueLayers();
          lastlyr_mc_build_FR_ = simtrack.getLastFoundHitLyr();

          duplmask_build_FR_ = buildextra.isDuplicate();
          iTkMatches_build_FR_ =
              buildextra
                  .duplicateID();  // ith duplicate build track, i = 0 "best" match, i > 0 "still matched, real reco, not as good as i-1 track"

          if (Config::keepHitInfo)
            TTreeValidation::fillFullHitInfo(ev,
                                             simtrack,
                                             hitlyrs_mc_build_FR_,
                                             hitidxs_mc_build_FR_,
                                             hitmcTkIDs_mc_build_FR_,
                                             hitxs_mc_build_FR_,
                                             hitys_mc_build_FR_,
                                             hitzs_mc_build_FR_);
        } else  // build track matched only to seed not to sim
        {
          // -99 for all sim info for reco tracks not associated to reco tracks
          pt_mc_build_FR_ = -99;
          phi_mc_build_FR_ = -99;
          eta_mc_build_FR_ = -99;
          helixchi2_build_FR_ = -99;

          mcTSmask_build_FR_ = -1;

          nHits_mc_build_FR_ = -99;
          nLayers_mc_build_FR_ = -99;
          lastlyr_mc_build_FR_ = -99;

          duplmask_build_FR_ = -1;
          iTkMatches_build_FR_ = -99;
        }  // matched seed to build, not build to sim
      }

      else  // seed has no matching build track (therefore no matching sim to build track)
      {
        seedmask_build_FR_ = 0;  // quick logic

        // -3000 for position info if no build track for seed
        xhit_build_FR_ = -3000;
        yhit_build_FR_ = -3000;
        zhit_build_FR_ = -3000;

        // -100 for all reco info as no actual build track for this seed
        pt_build_FR_ = -100;
        ept_build_FR_ = -100;
        phi_build_FR_ = -100;
        ephi_build_FR_ = -100;
        eta_build_FR_ = -100;
        eeta_build_FR_ = -100;

        nHits_build_FR_ = -100;
        nLayers_build_FR_ = -100;
        nHitsMatched_build_FR_ = -100;
        fracHitsMatched_build_FR_ = -100;
        lastlyr_build_FR_ = -100;

        dphi_build_FR_ = -100;

        hitchi2_build_FR_ = -100;
        score_build_FR_ = -5001;

        // keep -100 for all sim variables as no such reco exists for this seed
        mcmask_build_FR_ = -2;  // do not want to count towards build FR
        mcID_build_FR_ = -100;

        pt_mc_build_FR_ = -100;
        phi_mc_build_FR_ = -100;
        eta_mc_build_FR_ = -100;
        helixchi2_build_FR_ = -100;

        mcTSmask_build_FR_ = -3;

        nHits_mc_build_FR_ = -100;
        nLayers_mc_build_FR_ = -100;
        lastlyr_mc_build_FR_ = -100;

        duplmask_build_FR_ = -2;
        iTkMatches_build_FR_ = -100;
      }

      //============================// fit tracks
      if (seedToFitMap_.count(seedID_FR_)) {
        seedmask_fit_FR_ = 1;  // quick logic

        const auto& fittrack = evt_fit_tracks[seedToFitMap_[seedID_FR_]];
        const auto& fitextra = evt_fit_extras[fittrack.label()];

        // last hit info
        const Hit& lasthit = evt_layer_hits[fittrack.getLastFoundHitLyr()][fittrack.getLastFoundHitIdx()];
        xhit_fit_FR_ = lasthit.x();
        yhit_fit_FR_ = lasthit.y();
        zhit_fit_FR_ = lasthit.z();

        pt_fit_FR_ = fittrack.pT();
        ept_fit_FR_ = fittrack.epT();
        phi_fit_FR_ = fittrack.momPhi();
        ephi_fit_FR_ = fittrack.emomPhi();
        eta_fit_FR_ = fittrack.momEta();
        eeta_fit_FR_ = fittrack.emomEta();

        nHits_fit_FR_ = fittrack.nFoundHits();
        nLayers_fit_FR_ = fittrack.nUniqueLayers();
        nHitsMatched_fit_FR_ = fitextra.nHitsMatched();
        fracHitsMatched_fit_FR_ = fitextra.fracHitsMatched();
        lastlyr_fit_FR_ = fittrack.getLastFoundHitLyr();

        // swim dphi
        dphi_fit_FR_ = fitextra.dPhi();

        // quality info
        hitchi2_fit_FR_ = fittrack.chi2();  // -10 when not used
        score_fit_FR_ = fittrack.score();

        if (Config::keepHitInfo)
          TTreeValidation::fillFullHitInfo(ev,
                                           fittrack,
                                           hitlyrs_fit_FR_,
                                           hitidxs_fit_FR_,
                                           hitmcTkIDs_fit_FR_,
                                           hitxs_fit_FR_,
                                           hitys_fit_FR_,
                                           hitzs_fit_FR_);

        // sim info for fit track
        mcID_fit_FR_ = fitextra.mcTrackID();
        mcmask_fit_FR_ = TTreeValidation::getMaskAssignment(mcID_fit_FR_);

        if (mcmask_fit_FR_ == 1)  // fit track matched to seed and sim
        {
          const auto& simtrack = evt_sim_tracks[mcID_fit_FR_];

          const int mcHitID = TTreeValidation::getLastFoundHit(
              fittrack.getLastFoundMCHitID(evt_layer_hits), mcID_fit_FR_, ev);  // only works for outward fit for now
          if (mcHitID >= 0 && Config::readSimTrackStates) {
            const TrackState& initLayTS = evt_sim_trackstates[mcHitID];
            pt_mc_fit_FR_ = initLayTS.pT();
            phi_mc_fit_FR_ = initLayTS.momPhi();
            eta_mc_fit_FR_ = initLayTS.momEta();
            helixchi2_fit_FR_ = computeHelixChi2(initLayTS.parameters, fittrack.parameters(), fittrack.errors());

            mcTSmask_fit_FR_ = 1;
          } else if (Config::tryToSaveSimInfo) {
            pt_mc_fit_FR_ = simtrack.pT();
            phi_mc_fit_FR_ = simtrack.momPhi();
            eta_mc_fit_FR_ = simtrack.momEta();
            helixchi2_fit_FR_ = computeHelixChi2(simtrack.parameters(), fittrack.parameters(), fittrack.errors());

            mcTSmask_fit_FR_ = 0;
          } else {
            pt_mc_fit_FR_ = -101;
            phi_mc_fit_FR_ = -101;
            eta_mc_fit_FR_ = -101;
            helixchi2_fit_FR_ = -101;

            mcTSmask_fit_FR_ = -2;
          }

          nHits_mc_fit_FR_ = simtrack.nFoundHits();
          nLayers_mc_fit_FR_ = simtrack.nUniqueLayers();
          lastlyr_mc_fit_FR_ = simtrack.getLastFoundHitLyr();

          duplmask_fit_FR_ = fitextra.isDuplicate();
          iTkMatches_fit_FR_ =
              fitextra
                  .duplicateID();  // ith duplicate fit track, i = 0 "best" match, i > 0 "still matched, real reco, not as good as i-1 track"

          if (Config::keepHitInfo)
            TTreeValidation::fillFullHitInfo(ev,
                                             simtrack,
                                             hitlyrs_mc_fit_FR_,
                                             hitidxs_mc_fit_FR_,
                                             hitmcTkIDs_mc_fit_FR_,
                                             hitxs_mc_fit_FR_,
                                             hitys_mc_fit_FR_,
                                             hitzs_mc_fit_FR_);
        } else  // fit track matched only to seed not to sim
        {
          // -99 for all sim info for reco tracks not associated to reco tracks
          pt_mc_fit_FR_ = -99;
          phi_mc_fit_FR_ = -99;
          eta_mc_fit_FR_ = -99;
          helixchi2_fit_FR_ = -99;

          mcTSmask_fit_FR_ = -1;

          nHits_mc_fit_FR_ = -99;
          nLayers_mc_fit_FR_ = -99;
          lastlyr_mc_fit_FR_ = -99;

          duplmask_fit_FR_ = -1;
          iTkMatches_fit_FR_ = -99;
        }  // matched seed to fit, not fit to sim
      }

      else  // seed has no matching fit track (therefore no matching sim to fit track)
      {
        seedmask_fit_FR_ = 0;  // quick logic

        // -3000 for position info if no fit track for seed
        xhit_fit_FR_ = -3000;
        yhit_fit_FR_ = -3000;
        zhit_fit_FR_ = -3000;

        // -100 for all reco info as no actual fit track for this seed
        pt_fit_FR_ = -100;
        ept_fit_FR_ = -100;
        phi_fit_FR_ = -100;
        ephi_fit_FR_ = -100;
        eta_fit_FR_ = -100;
        eeta_fit_FR_ = -100;

        nHits_fit_FR_ = -100;
        nLayers_fit_FR_ = -100;
        nHitsMatched_fit_FR_ = -100;
        fracHitsMatched_fit_FR_ = -100;
        lastlyr_fit_FR_ = -100;

        dphi_fit_FR_ = -100;

        hitchi2_fit_FR_ = -100;
        score_fit_FR_ = -5001;

        // keep -100 for all sim variables as no such reco exists for this seed
        mcmask_fit_FR_ = -2;  // do not want to count towards fit FR
        mcID_fit_FR_ = -100;

        pt_mc_fit_FR_ = -100;
        phi_mc_fit_FR_ = -100;
        eta_mc_fit_FR_ = -100;
        helixchi2_fit_FR_ = -100;

        mcTSmask_fit_FR_ = -3;

        nHits_mc_fit_FR_ = -100;
        nLayers_mc_fit_FR_ = -100;
        lastlyr_mc_fit_FR_ = -100;

        duplmask_fit_FR_ = -2;
        iTkMatches_fit_FR_ = -100;
      }

      frtree_->Fill();  // fill once per seed!
    }                   // end of seed to seed loop
  }

  void TTreeValidation::fillConfigTree() {
    std::lock_guard<std::mutex> locker(glock_);

    Ntracks_ = Config::nTracks;
    Nevents_ = Config::nEvents;

    nLayers_ = Config::nLayers;

    nlayers_per_seed_ = Config::ItrInfo[0].m_params.nlayers_per_seed;
    maxCand_ = Config::ItrInfo[0].m_params.maxCandsPerSeed;
    chi2Cut_min_ = Config::ItrInfo[0].m_params.chi2Cut_min;
    nSigma_ = Config::nSigma;
    minDPhi_ = Config::minDPhi;
    maxDPhi_ = Config::maxDPhi;
    minDEta_ = Config::minDEta;
    maxDEta_ = Config::maxDEta;

    beamspotX_ = Config::beamspotX;
    beamspotY_ = Config::beamspotY;
    beamspotZ_ = Config::beamspotZ;

    minSimPt_ = Config::minSimPt;
    maxSimPt_ = Config::maxSimPt;

    hitposerrXY_ = Config::hitposerrXY;
    hitposerrZ_ = Config::hitposerrZ;
    hitposerrR_ = Config::hitposerrR;
    varXY_ = Config::varXY;
    varZ_ = Config::varZ;

    ptinverr049_ = Config::ptinverr049;
    phierr049_ = Config::phierr049;
    thetaerr049_ = Config::thetaerr049;
    ptinverr012_ = Config::ptinverr012;
    phierr012_ = Config::phierr012;
    thetaerr012_ = Config::thetaerr012;

    configtree_->Fill();
  }

  void TTreeValidation::fillCMSSWEfficiencyTree(const Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);

    const auto ievt = ev.evtID();
    const auto& evt_sim_tracks = ev.simTracks_;
    const auto& evt_cmssw_tracks = ev.cmsswTracks_;
    const auto& evt_cmssw_extras = ev.cmsswTracksExtra_;
    const auto& evt_build_tracks = ev.candidateTracks_;
    const auto& evt_build_extras = ev.candidateTracksExtra_;
    const auto& evt_fit_tracks = ev.fitTracks_;
    const auto& evt_fit_extras = ev.fitTracksExtra_;
    const auto& evt_layer_hits = ev.layerHits_;

    for (const auto& cmsswtrack : evt_cmssw_tracks) {
      // clear hit info
      if (Config::keepHitInfo) {
        hitlyrs_cmssw_ceff_.clear();
        hitlyrs_build_ceff_.clear();
        hitlyrs_mc_build_ceff_.clear();
        hitlyrs_fit_ceff_.clear();
        hitlyrs_mc_fit_ceff_.clear();

        hitidxs_cmssw_ceff_.clear();
        hitidxs_build_ceff_.clear();
        hitidxs_mc_build_ceff_.clear();
        hitidxs_fit_ceff_.clear();
        hitidxs_mc_fit_ceff_.clear();
      }

      const auto& cmsswextra = evt_cmssw_extras[cmsswtrack.label()];

      evtID_ceff_ = ievt;
      cmsswID_ceff_ = cmsswtrack.label();
      seedID_cmssw_ceff_ = cmsswextra.seedID();

      // PCA parameters
      x_cmssw_ceff_ = cmsswtrack.x();
      y_cmssw_ceff_ = cmsswtrack.y();
      z_cmssw_ceff_ = cmsswtrack.z();

      pt_cmssw_ceff_ = cmsswtrack.pT();
      phi_cmssw_ceff_ = cmsswtrack.momPhi();
      eta_cmssw_ceff_ = cmsswtrack.momEta();

      nHits_cmssw_ceff_ = cmsswtrack.nFoundHits();
      nLayers_cmssw_ceff_ = cmsswtrack.nUniqueLayers();
      lastlyr_cmssw_ceff_ = cmsswtrack.getLastFoundHitLyr();

      itermask_build_ceff_ = 0;
      itermask_fit_ceff_ = 0;
      iterduplmask_build_ceff_ = 0;
      iterduplmask_fit_ceff_ = 0;
      algo_seed_ceff_ = 0;

      for (auto aa : cmsswextra.seedAlgos())
        algo_seed_ceff_ = (algo_seed_ceff_ | (1 << aa));

      if (Config::keepHitInfo)
        TTreeValidation::fillMinHitInfo(cmsswtrack, hitlyrs_cmssw_ceff_, hitidxs_cmssw_ceff_);

      // matched build track
      if (cmsswToBuildMap_.count(cmsswID_ceff_) &&
          cmsswtrack
              .isFindable())  // recoToCmssw match : save best match with best score i.e. cmsswToBuildMap_[matched CmsswID][first element in vector]
      {
        for (unsigned int ii = 0; ii < cmsswToBuildMap_[cmsswID_ceff_].size(); ii++) {
          const int theAlgo = evt_build_tracks[cmsswToBuildMap_[cmsswID_ceff_][ii]].algoint();
          if ((itermask_build_ceff_ >> theAlgo) & 1)
            iterduplmask_build_ceff_ = (iterduplmask_build_ceff_ | (1 << theAlgo));  //filled at the second time
          itermask_build_ceff_ = (itermask_build_ceff_ | (1 << theAlgo));
        }

        const auto& buildtrack =
            evt_build_tracks[cmsswToBuildMap_[cmsswID_ceff_][0]];  // returns buildTrack best matched to cmssw track
        const auto& buildextra =
            evt_build_extras[buildtrack.label()];  // returns track extra best aligned with build track
        cmsswmask_build_ceff_ = 1;                 // quick logic for matched

        seedID_build_ceff_ = buildextra.seedID();
        mcTrackID_build_ceff_ = buildextra.mcTrackID();

        // track parameters
        pt_build_ceff_ = buildtrack.pT();
        ept_build_ceff_ = buildtrack.epT();
        phi_build_ceff_ = buildtrack.momPhi();
        ephi_build_ceff_ = buildtrack.emomPhi();
        eta_build_ceff_ = buildtrack.momEta();
        eeta_build_ceff_ = buildtrack.emomEta();

        // gen info
        if (mcTrackID_build_ceff_ >= 0) {
          const auto& simtrack = evt_sim_tracks[mcTrackID_build_ceff_];
          x_mc_build_ceff_ = simtrack.x();
          y_mc_build_ceff_ = simtrack.y();
          z_mc_build_ceff_ = simtrack.z();
          pt_mc_build_ceff_ = simtrack.pT();
          phi_mc_build_ceff_ = simtrack.momPhi();
          eta_mc_build_ceff_ = simtrack.momEta();

          if (Config::keepHitInfo)
            TTreeValidation::fillMinHitInfo(simtrack, hitlyrs_mc_build_ceff_, hitidxs_mc_build_ceff_);
        } else {
          x_mc_build_ceff_ = -1000;
          y_mc_build_ceff_ = -1000;
          z_mc_build_ceff_ = -1000;
          pt_mc_build_ceff_ = -99;
          phi_mc_build_ceff_ = -99;
          eta_mc_build_ceff_ = -99;
        }

        // hit/layer info
        nHits_build_ceff_ = buildtrack.nFoundHits();
        nLayers_build_ceff_ = buildtrack.nUniqueLayers();
        nHitsMatched_build_ceff_ = buildextra.nHitsMatched();
        fracHitsMatched_build_ceff_ = buildextra.fracHitsMatched();
        lastlyr_build_ceff_ = buildtrack.getLastFoundHitLyr();

        // hit info
        const Hit& lasthit = evt_layer_hits[buildtrack.getLastFoundHitLyr()][buildtrack.getLastFoundHitIdx()];
        xhit_build_ceff_ = lasthit.x();
        yhit_build_ceff_ = lasthit.y();
        zhit_build_ceff_ = lasthit.z();

        // quality info
        hitchi2_build_ceff_ = buildtrack.chi2();
        helixchi2_build_ceff_ = buildextra.helixChi2();
        score_build_ceff_ = buildtrack.score();

        // swim dphi
        dphi_build_ceff_ = buildextra.dPhi();

        // duplicate info
        duplmask_build_ceff_ = buildextra.isDuplicate();
        nTkMatches_build_ceff_ = cmsswToBuildMap_[cmsswID_ceff_].size();  // n reco matches to this cmssw track.

        if (Config::keepHitInfo)
          TTreeValidation::fillMinHitInfo(buildtrack, hitlyrs_build_ceff_, hitidxs_build_ceff_);
      } else  // unmatched cmsswtracks ... put -99 for all reco values to denote unmatched
      {
        cmsswmask_build_ceff_ = (cmsswtrack.isFindable() ? 0 : -1);  // quick logic for not matched

        seedID_build_ceff_ = -99;
        mcTrackID_build_ceff_ = -99;

        pt_build_ceff_ = -99;
        ept_build_ceff_ = -99;
        phi_build_ceff_ = -99;
        ephi_build_ceff_ = -99;
        eta_build_ceff_ = -99;
        eeta_build_ceff_ = -99;

        x_mc_build_ceff_ = -2000;
        y_mc_build_ceff_ = -2000;
        z_mc_build_ceff_ = -2000;
        pt_mc_build_ceff_ = -99;
        phi_mc_build_ceff_ = -99;
        eta_mc_build_ceff_ = -99;

        nHits_build_ceff_ = -99;
        nLayers_build_ceff_ = -99;
        nHitsMatched_build_ceff_ = -99;
        fracHitsMatched_build_ceff_ = -99;
        lastlyr_build_ceff_ = -99;

        xhit_build_ceff_ = -2000;
        yhit_build_ceff_ = -2000;
        zhit_build_ceff_ = -2000;

        hitchi2_build_ceff_ = -99;
        helixchi2_build_ceff_ = -99;
        score_build_ceff_ = -17000;

        dphi_build_ceff_ = -99;

        duplmask_build_ceff_ = -1;     // mask means unmatched cmssw track
        nTkMatches_build_ceff_ = -99;  // unmatched
      }

      // matched fit track
      if (cmsswToFitMap_.count(cmsswID_ceff_) &&
          cmsswtrack
              .isFindable())  // recoToCmssw match : save best match with best score i.e. cmsswToFitMap_[matched CmsswID][first element in vector]
      {
        for (unsigned int ii = 0; ii < cmsswToFitMap_[cmsswID_ceff_].size(); ii++) {
          const int theAlgo = evt_build_tracks[cmsswToFitMap_[cmsswID_ceff_][ii]].algoint();
          if ((itermask_fit_ceff_ >> theAlgo) & 1)
            iterduplmask_fit_ceff_ = (iterduplmask_fit_ceff_ | (1 << theAlgo));  //filled at the second time
          itermask_fit_ceff_ = (itermask_fit_ceff_ | (1 << theAlgo));
        }

        const auto& fittrack =
            evt_fit_tracks[cmsswToFitMap_[cmsswID_ceff_][0]];     // returns fitTrack best matched to cmssw track
        const auto& fitextra = evt_fit_extras[fittrack.label()];  // returns track extra best aligned with fit track
        cmsswmask_fit_ceff_ = 1;                                  // quick logic for matched

        seedID_fit_ceff_ = fitextra.seedID();
        mcTrackID_fit_ceff_ = fitextra.mcTrackID();

        // track parameters
        pt_fit_ceff_ = fittrack.pT();
        ept_fit_ceff_ = fittrack.epT();
        phi_fit_ceff_ = fittrack.momPhi();
        ephi_fit_ceff_ = fittrack.emomPhi();
        eta_fit_ceff_ = fittrack.momEta();
        eeta_fit_ceff_ = fittrack.emomEta();

        // gen info
        if (mcTrackID_fit_ceff_ >= 0) {
          const auto& simtrack = evt_sim_tracks[mcTrackID_fit_ceff_];
          x_mc_fit_ceff_ = simtrack.x();
          y_mc_fit_ceff_ = simtrack.y();
          z_mc_fit_ceff_ = simtrack.z();
          pt_mc_fit_ceff_ = simtrack.pT();
          phi_mc_fit_ceff_ = simtrack.momPhi();
          eta_mc_fit_ceff_ = simtrack.momEta();

          if (Config::keepHitInfo)
            TTreeValidation::fillMinHitInfo(simtrack, hitlyrs_mc_fit_ceff_, hitidxs_mc_fit_ceff_);
        } else {
          x_mc_fit_ceff_ = -1000;
          y_mc_fit_ceff_ = -1000;
          z_mc_fit_ceff_ = -1000;
          pt_mc_fit_ceff_ = -99;
          phi_mc_fit_ceff_ = -99;
          eta_mc_fit_ceff_ = -99;
        }

        // hit/layer info
        nHits_fit_ceff_ = fittrack.nFoundHits();
        nLayers_fit_ceff_ = fittrack.nUniqueLayers();
        nHitsMatched_fit_ceff_ = fitextra.nHitsMatched();
        fracHitsMatched_fit_ceff_ = fitextra.fracHitsMatched();
        lastlyr_fit_ceff_ = fittrack.getLastFoundHitLyr();

        // hit info
        const Hit& lasthit = evt_layer_hits[fittrack.getLastFoundHitLyr()][fittrack.getLastFoundHitIdx()];
        xhit_fit_ceff_ = lasthit.x();
        yhit_fit_ceff_ = lasthit.y();
        zhit_fit_ceff_ = lasthit.z();

        // quality info
        hitchi2_fit_ceff_ = fittrack.chi2();
        helixchi2_fit_ceff_ = fitextra.helixChi2();
        score_fit_ceff_ = fittrack.score();

        // swim dphi
        dphi_fit_ceff_ = fitextra.dPhi();

        // duplicate info
        duplmask_fit_ceff_ = fitextra.isDuplicate();
        nTkMatches_fit_ceff_ = cmsswToFitMap_[cmsswID_ceff_].size();  // n reco matches to this cmssw track.

        if (Config::keepHitInfo)
          TTreeValidation::fillMinHitInfo(fittrack, hitlyrs_fit_ceff_, hitidxs_fit_ceff_);
      } else  // unmatched cmsswtracks ... put -99 for all reco values to denote unmatched
      {
        cmsswmask_fit_ceff_ = (cmsswtrack.isFindable() ? 0 : -1);  // quick logic for not matched

        seedID_fit_ceff_ = -99;
        mcTrackID_fit_ceff_ = -99;

        pt_fit_ceff_ = -99;
        ept_fit_ceff_ = -99;
        phi_fit_ceff_ = -99;
        ephi_fit_ceff_ = -99;
        eta_fit_ceff_ = -99;
        eeta_fit_ceff_ = -99;

        x_mc_fit_ceff_ = -2000;
        y_mc_fit_ceff_ = -2000;
        z_mc_fit_ceff_ = -2000;
        pt_mc_fit_ceff_ = -99;
        phi_mc_fit_ceff_ = -99;
        eta_mc_fit_ceff_ = -99;

        nHits_fit_ceff_ = -99;
        nLayers_fit_ceff_ = -99;
        nHitsMatched_fit_ceff_ = -99;
        fracHitsMatched_fit_ceff_ = -99;
        lastlyr_fit_ceff_ = -99;

        xhit_fit_ceff_ = -2000;
        yhit_fit_ceff_ = -2000;
        zhit_fit_ceff_ = -2000;

        hitchi2_fit_ceff_ = -99;
        helixchi2_fit_ceff_ = -99;
        score_fit_ceff_ = -17000;

        dphi_fit_ceff_ = -99;

        duplmask_fit_ceff_ = -1;     // mask means unmatched cmssw track
        nTkMatches_fit_ceff_ = -99;  // unmatched
      }

      cmsswefftree_->Fill();
    }
  }

  void TTreeValidation::fillCMSSWFakeRateTree(const Event& ev) {
    std::lock_guard<std::mutex> locker(glock_);

    auto ievt = ev.evtID();
    const auto& evt_sim_tracks = ev.simTracks_;
    const auto& evt_cmssw_tracks = ev.cmsswTracks_;
    const auto& evt_cmssw_extras = ev.cmsswTracksExtra_;
    const auto& evt_build_tracks = ev.candidateTracks_;
    const auto& evt_build_extras = ev.candidateTracksExtra_;
    const auto& evt_fit_tracks = ev.fitTracks_;
    const auto& evt_fit_extras = ev.fitTracksExtra_;
    const auto& evt_layer_hits = ev.layerHits_;

    for (const auto& buildtrack : evt_build_tracks) {
      if (Config::keepHitInfo) {
        hitlyrs_mc_cFR_.clear();
        hitlyrs_build_cFR_.clear();
        hitlyrs_cmssw_build_cFR_.clear();
        hitlyrs_fit_cFR_.clear();
        hitlyrs_cmssw_fit_cFR_.clear();

        hitidxs_mc_cFR_.clear();
        hitidxs_build_cFR_.clear();
        hitidxs_cmssw_build_cFR_.clear();
        hitidxs_fit_cFR_.clear();
        hitidxs_cmssw_fit_cFR_.clear();
      }

      algorithm_cFR_ = buildtrack.algoint();

      const auto& buildextra = evt_build_extras[buildtrack.label()];

      // same for fit and build tracks
      evtID_cFR_ = ievt;
      seedID_cFR_ = buildextra.seedID();
      mcTrackID_cFR_ = buildextra.mcTrackID();

      // track parameters
      pt_build_cFR_ = buildtrack.pT();
      ept_build_cFR_ = buildtrack.epT();
      phi_build_cFR_ = buildtrack.momPhi();
      ephi_build_cFR_ = buildtrack.emomPhi();
      eta_build_cFR_ = buildtrack.momEta();
      eeta_build_cFR_ = buildtrack.emomEta();

      // gen info
      if (mcTrackID_cFR_ >= 0) {
        const auto& simtrack = evt_sim_tracks[mcTrackID_cFR_];
        x_mc_cFR_ = simtrack.x();
        y_mc_cFR_ = simtrack.y();
        z_mc_cFR_ = simtrack.z();
        pt_mc_cFR_ = simtrack.pT();
        phi_mc_cFR_ = simtrack.momPhi();
        eta_mc_cFR_ = simtrack.momEta();

        if (Config::keepHitInfo)
          TTreeValidation::fillMinHitInfo(simtrack, hitlyrs_mc_cFR_, hitidxs_mc_cFR_);
      } else {
        x_mc_cFR_ = -1000;
        y_mc_cFR_ = -1000;
        z_mc_cFR_ = -1000;
        pt_mc_cFR_ = -99;
        phi_mc_cFR_ = -99;
        eta_mc_cFR_ = -99;
      }

      // hit/layer info
      nHits_build_cFR_ = buildtrack.nFoundHits();
      nLayers_build_cFR_ = buildtrack.nUniqueLayers();
      nHitsMatched_build_cFR_ = buildextra.nHitsMatched();
      fracHitsMatched_build_cFR_ = buildextra.fracHitsMatched();
      lastlyr_build_cFR_ = buildtrack.getLastFoundHitLyr();

      // hit info
      const Hit& lasthit = evt_layer_hits[buildtrack.getLastFoundHitLyr()][buildtrack.getLastFoundHitIdx()];
      xhit_build_cFR_ = lasthit.x();
      yhit_build_cFR_ = lasthit.y();
      zhit_build_cFR_ = lasthit.z();

      // quality info
      hitchi2_build_cFR_ = buildtrack.chi2();
      helixchi2_build_cFR_ = buildextra.helixChi2();
      score_build_cFR_ = buildtrack.score();

      // stored dphi
      dphi_build_cFR_ = buildextra.dPhi();

      if (Config::keepHitInfo)
        TTreeValidation::fillMinHitInfo(buildtrack, hitlyrs_build_cFR_, hitidxs_build_cFR_);

      // cmssw match?
      cmsswID_build_cFR_ = buildextra.cmsswTrackID();
      cmsswmask_build_cFR_ = TTreeValidation::getMaskAssignment(cmsswID_build_cFR_);

      if (cmsswmask_build_cFR_ == 1)  // matched track to cmssw
      {
        const auto& cmsswtrack = evt_cmssw_tracks[cmsswID_build_cFR_];
        const auto& cmsswextra = evt_cmssw_extras[cmsswtrack.label()];

        seedID_cmssw_build_cFR_ = cmsswextra.seedID();

        x_cmssw_build_cFR_ = cmsswtrack.x();
        y_cmssw_build_cFR_ = cmsswtrack.y();
        z_cmssw_build_cFR_ = cmsswtrack.z();

        pt_cmssw_build_cFR_ = cmsswtrack.pT();
        phi_cmssw_build_cFR_ = cmsswtrack.momPhi();
        eta_cmssw_build_cFR_ = cmsswtrack.momEta();

        nHits_cmssw_build_cFR_ = cmsswtrack.nFoundHits();
        nLayers_cmssw_build_cFR_ = cmsswtrack.nUniqueLayers();
        lastlyr_cmssw_build_cFR_ = cmsswtrack.getLastFoundHitLyr();

        // duplicate info
        duplmask_build_cFR_ = buildextra.isDuplicate();
        iTkMatches_build_cFR_ = buildextra.duplicateID();

        if (Config::keepHitInfo)
          TTreeValidation::fillMinHitInfo(cmsswtrack, hitlyrs_cmssw_build_cFR_, hitidxs_cmssw_build_cFR_);
      } else  // unmatched cmsswtracks ... put -99 for all reco values to denote unmatched
      {
        seedID_cmssw_build_cFR_ = -99;

        x_cmssw_build_cFR_ = -2000;
        y_cmssw_build_cFR_ = -2000;
        z_cmssw_build_cFR_ = -2000;

        pt_cmssw_build_cFR_ = -99;
        phi_cmssw_build_cFR_ = -99;
        eta_cmssw_build_cFR_ = -99;

        nHits_cmssw_build_cFR_ = -99;
        nLayers_cmssw_build_cFR_ = -99;
        lastlyr_cmssw_build_cFR_ = -99;

        duplmask_build_cFR_ = -1;
        iTkMatches_build_cFR_ = -99;
      }

      // ensure there is a fit track to mess with
      if (buildToFitMap_.count(buildtrack.label())) {
        const auto& fittrack = evt_fit_tracks[buildToFitMap_[buildtrack.label()]];
        const auto& fitextra = evt_fit_extras[fittrack.label()];

        // track parameters
        pt_fit_cFR_ = fittrack.pT();
        ept_fit_cFR_ = fittrack.epT();
        phi_fit_cFR_ = fittrack.momPhi();
        ephi_fit_cFR_ = fittrack.emomPhi();
        eta_fit_cFR_ = fittrack.momEta();
        eeta_fit_cFR_ = fittrack.emomEta();

        // hit/layer info
        nHits_fit_cFR_ = fittrack.nFoundHits();
        nLayers_fit_cFR_ = fittrack.nUniqueLayers();
        nHitsMatched_fit_cFR_ = fitextra.nHitsMatched();
        fracHitsMatched_fit_cFR_ = fitextra.fracHitsMatched();
        lastlyr_fit_cFR_ = fittrack.getLastFoundHitLyr();

        // hit info
        const Hit& lasthit = evt_layer_hits[fittrack.getLastFoundHitLyr()][fittrack.getLastFoundHitIdx()];
        xhit_fit_cFR_ = lasthit.x();
        yhit_fit_cFR_ = lasthit.y();
        zhit_fit_cFR_ = lasthit.z();

        // chi2 info
        hitchi2_fit_cFR_ = fittrack.chi2();
        helixchi2_fit_cFR_ = fitextra.helixChi2();
        score_fit_cFR_ = fittrack.score();

        // stored dphi
        dphi_fit_cFR_ = fitextra.dPhi();

        if (Config::keepHitInfo)
          TTreeValidation::fillMinHitInfo(buildtrack, hitlyrs_fit_cFR_, hitidxs_fit_cFR_);

        // cmssw match?
        cmsswID_fit_cFR_ = fitextra.cmsswTrackID();
        cmsswmask_fit_cFR_ = TTreeValidation::getMaskAssignment(cmsswID_fit_cFR_);

        if (cmsswmask_fit_cFR_ == 1)  // matched track to cmssw
        {
          const auto& cmsswtrack = evt_cmssw_tracks[cmsswID_fit_cFR_];
          const auto& cmsswextra = evt_cmssw_extras[cmsswtrack.label()];

          seedID_cmssw_fit_cFR_ = cmsswextra.seedID();

          x_cmssw_fit_cFR_ = cmsswtrack.x();
          y_cmssw_fit_cFR_ = cmsswtrack.y();
          z_cmssw_fit_cFR_ = cmsswtrack.z();

          pt_cmssw_fit_cFR_ = cmsswtrack.pT();
          phi_cmssw_fit_cFR_ = cmsswtrack.momPhi();
          eta_cmssw_fit_cFR_ = cmsswtrack.momEta();

          nHits_cmssw_fit_cFR_ = cmsswtrack.nFoundHits();
          nLayers_cmssw_fit_cFR_ = cmsswtrack.nUniqueLayers();
          lastlyr_cmssw_fit_cFR_ = cmsswtrack.getLastFoundHitLyr();

          // duplicate info
          duplmask_fit_cFR_ = fitextra.isDuplicate();
          iTkMatches_fit_cFR_ = fitextra.duplicateID();

          if (Config::keepHitInfo)
            TTreeValidation::fillMinHitInfo(fittrack, hitlyrs_cmssw_fit_cFR_, hitidxs_cmssw_fit_cFR_);
        } else  // unmatched cmsswtracks ... put -99 for all reco values to denote unmatched
        {
          seedID_cmssw_fit_cFR_ = -99;

          x_cmssw_fit_cFR_ = -2000;
          y_cmssw_fit_cFR_ = -2000;
          z_cmssw_fit_cFR_ = -2000;

          pt_cmssw_fit_cFR_ = -99;
          phi_cmssw_fit_cFR_ = -99;
          eta_cmssw_fit_cFR_ = -99;

          nHits_cmssw_fit_cFR_ = -99;
          nLayers_cmssw_fit_cFR_ = -99;
          lastlyr_cmssw_fit_cFR_ = -99;

          duplmask_fit_cFR_ = -1;
          iTkMatches_fit_cFR_ = -99;
        }
      } else  // no fit track to match to a build track!
      {
        pt_fit_cFR_ = -100;
        ept_fit_cFR_ = -100;
        phi_fit_cFR_ = -100;
        ephi_fit_cFR_ = -100;
        eta_fit_cFR_ = -100;
        eeta_fit_cFR_ = -100;

        nHits_fit_cFR_ = -100;
        nLayers_fit_cFR_ = -100;
        nHitsMatched_fit_cFR_ = -100;
        fracHitsMatched_fit_cFR_ = -100;
        lastlyr_fit_cFR_ = -100;

        xhit_fit_cFR_ = -3000;
        yhit_fit_cFR_ = -3000;
        zhit_fit_cFR_ = -3000;

        hitchi2_fit_cFR_ = -100;
        helixchi2_fit_cFR_ = -100;
        score_fit_cFR_ = -5001;
        dphi_fit_cFR_ = -100;

        cmsswID_fit_cFR_ = -100;
        cmsswmask_fit_cFR_ = -2;

        seedID_cmssw_fit_cFR_ = -100;

        x_cmssw_fit_cFR_ = -3000;
        y_cmssw_fit_cFR_ = -3000;
        z_cmssw_fit_cFR_ = -3000;

        pt_cmssw_fit_cFR_ = -100;
        phi_cmssw_fit_cFR_ = -100;
        eta_cmssw_fit_cFR_ = -100;

        nHits_cmssw_fit_cFR_ = -100;
        nLayers_cmssw_fit_cFR_ = -100;
        lastlyr_cmssw_fit_cFR_ = -100;

        duplmask_fit_cFR_ = -2;
        iTkMatches_fit_cFR_ = -100;
      }

      cmsswfrtree_->Fill();
    }
  }

  void TTreeValidation::saveTTrees() {
    std::lock_guard<std::mutex> locker(glock_);
    f_->cd();

    if (Config::sim_val_for_cmssw || Config::sim_val) {
      efftree_->SetDirectory(f_.get());
      efftree_->Write();

      frtree_->SetDirectory(f_.get());
      frtree_->Write();
    }
    if (Config::cmssw_val) {
      cmsswefftree_->SetDirectory(f_.get());
      cmsswefftree_->Write();

      cmsswfrtree_->SetDirectory(f_.get());
      cmsswfrtree_->Write();
    }
    if (Config::fit_val) {
      fittree_->SetDirectory(f_.get());
      fittree_->Write();
    }

    configtree_->SetDirectory(f_.get());
    configtree_->Write();
  }

}  // end namespace mkfit
#endif
