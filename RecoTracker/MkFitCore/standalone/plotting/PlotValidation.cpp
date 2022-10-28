#include "PlotValidation.hh"

//////////////////////////////
// Some light documentation //
//////////////////////////////

// Indices are as follows
// e == entries in trees
// i == kinematic variables: pt, phi, eta
// j == reco track collections: seed, build, fit
// k == pt cuts
// l == rates: eff, dupl rate, ineff (efftree), eff, dupl rate, fake rate (print)
// m == eta regions: brl, trans, enc (efftree)
// n == track quality plots: nHits, fracHits, track score (frtree and print)
// o == matched reco collections: all reco, fake, all match, best match (frtree), all reco, fake, and best match (print)
// p == diff of kinematic variables: dnhits, dinvpt, deta, dphi (frtree)
// q == n directories in frtree

// Variable name scheme
// *_var(s)_* == kinematic variables (index i)
// *_trk(s)_* == reco track collections (index j)
// *_ref* == variable associated to reference tracks (CMSSW or Sim)
// kinematic variable name comes before trk name to maintain consistency with branch names
// mask name also obeys this rule: which type of mask, then which reco track collection for association
// in case of "_ref_trk", this means this a reference track variable associated to a given reco track collection

// f* == data member
// s at the start == "string" version of the variable
// h at the start == a different string version

PlotValidation::PlotValidation(const TString& inName,
                               const TString& outName,
                               const Bool_t cmsswComp,
                               const int algo,
                               const Bool_t mvInput,
                               const Bool_t rmSuffix,
                               const Bool_t saveAs,
                               const TString& outType)
    : fInName(inName),
      fOutName(outName),
      fCmsswComp(cmsswComp),
      fAlgo(algo),
      fMvInput(mvInput),
      fRmSuffix(rmSuffix),
      fSaveAs(saveAs),
      fOutType(outType) {
  // Setup
  PlotValidation::SetupStyle();
  if (fAlgo > 0 && !fRmSuffix)
    fOutName = fOutName + "_iter" + algo;
  PlotValidation::MakeOutDir(fOutName);
  PlotValidation::SetupBins();
  PlotValidation::SetupCommonVars();

  // Get input root file or exit!
  fInRoot = TFile::Open(fInName.Data());
  if (fInRoot == (TFile*)NULL) {
    std::cerr << "File: " << fInName.Data() << " does not exist!!! Exiting..." << std::endl;
    exit(1);
  }
  gROOT->cd();
  efftree = (TTree*)fInRoot->Get((fCmsswComp ? "cmsswefftree" : "efftree"));
  frtree = (TTree*)fInRoot->Get((fCmsswComp ? "cmsswfrtree" : "frtree"));
  if (algo > 0)
    frtree = frtree->CopyTree(Form("algorithm==%i", algo));
  // make output root file
  fOutRoot = new TFile(fOutName + "/plots.root", "RECREATE");
}

PlotValidation::~PlotValidation() {
  delete efftree;
  delete frtree;
  delete fInRoot;
  delete fOutRoot;  // will delete all pointers to subdirectory
}

void PlotValidation::Validation(int algo) {
  std::cout << "Computing Efficiency, Inefficiency, and Duplicate Rate ..." << std::endl;
  PlotValidation::PlotEffTree(algo);

  std::cout << "Computing Fake Rate, <nHits/track>, and kinematic diffs to " << fSRefTitle.Data() << " tracks ..."
            << std::endl;
  PlotValidation::PlotFRTree(algo);

  std::cout << "Printing Totals ..." << std::endl;
  PlotValidation::PrintTotals(algo);

  if (fMvInput)
    PlotValidation::MoveInput();
}

// Loop over efficiency tree: produce efficiency, inefficiency per region of tracker, and duplicate rate
void PlotValidation::PlotEffTree(int algo) {
  ////////////////////////////////////////////
  // Declare strings for branches and plots //
  ////////////////////////////////////////////

  const TStrVec rates = {"eff", "dr", "ineff"};
  const TStrVec srates = {"Efficiency", "Duplicate Rate", "Inefficiency"};
  const UInt_t nrates = rates.size();

  const TStrVec regs = {"brl", "trans", "ec"};
  const TStrVec sregs = {"Barrel", "Transition", "Endcap"};
  const FltVec etacuts = {0, 0.9, 1.7, 2.45};
  const UInt_t nregs = regs.size();

  //////////////////////////
  // Create and new plots //
  //////////////////////////

  TEffRefMap plots;
  for (auto i = 0U; i < fNVars; i++)  // loop over fVars
  {
    const auto& var = fVars[i];
    const auto& svar = fSVars[i];
    const auto& sunit = fSUnits[i];

    // get bins for the variable of interest
    const auto& varbins = fVarBins[i];
    const Double_t* bins = &varbins[0];

    for (auto j = 0U; j < fNTrks; j++)  // loop over tracks
    {
      const auto& trk = fTrks[j];
      const auto& strk = fSTrks[j];

      for (auto k = 0U; k < fNPtCuts; k++)  // loop pver pt cuts
      {
        const auto& sptcut = fSPtCuts[k];
        const auto& hptcut = fHPtCuts[k];

        for (auto l = 0U; l < nrates; l++)  // loop over which rate
        {
          const auto& rate = rates[l];
          const auto& srate = srates[l];

          // plot names and key
          const TString plotkey = Form("%i_%i_%i_%i", i, j, k, l);
          const TString plotname = Form("%s_", fCmsswComp ? "cmssw" : "sim") + var + "_" + trk + "_pt" + hptcut;
          const TString plottitle = strk + " Track " + srate + " vs " + fSRefTitle + " " + svar + " {" + fSVarPt +
                                    " > " + sptcut + " " + fSUnitPt + "};" + svar + sunit + ";" + srate;

          // eff and dr not split by region
          if (l < 2) {
            const TString tmpname = rate + "_" + plotname;
            plots[plotkey] = new TEfficiency(tmpname.Data(), plottitle.Data(), varbins.size() - 1, bins);
          } else  // ineff split by region
          {
            for (auto m = 0U; m < nregs; m++)  // loop over regions for inefficiency
            {
              const auto& reg = regs[m];
              const auto& sreg = sregs[m];

              const TString tmpkey = Form("%s_%i", plotkey.Data(), m);
              const TString tmpname = rate + "_" + reg + "_" + plotname;
              const TString tmptitle = strk + " Track " + srate + " vs " + fSRefTitle + " " + svar + "{" + fSVarPt +
                                       " > " + sptcut + " " + fSUnitPt + ", " + sreg + "};" + svar + sunit + ";" +
                                       srate;

              plots[tmpkey] = new TEfficiency(tmpname.Data(), tmptitle.Data(), varbins.size() - 1, bins);
            }  // end loop over regions
          }    // end check over plots
        }      // end loop over plots
      }        // end loop over pt cuts
    }          // end loop over tracks
  }            // end loop over variables

  ////////////////////////////////////////
  // Floats/Ints to be filled for trees //
  ////////////////////////////////////////

  // Initialize var arrays, SetBranchAddress
  FltVec vars_ref(fNVars);            // first index is var. only for ref values! so no extra index
  TBrRefVec vars_ref_br(fNVars);      // tbranch for each var
  for (auto i = 0U; i < fNVars; i++)  // loop over trks index
  {
    const auto& var = fVars[i];
    auto& var_ref = vars_ref[i];
    auto& var_ref_br = vars_ref_br[i];

    // initialize var, branches
    var_ref = 0.;
    var_ref_br = 0;

    // Set var branch
    efftree->SetBranchAddress(
        var + "_" + ((fSRefVar == "cmssw" || var != "nLayers") ? fSRefVar : fSRefVarTrk), &var_ref, &var_ref_br);
  }

  // Initialize masks, set branch addresses
  IntVec refmask_trks(fNTrks);        // need to know if sim track associated to a given reco track type
  TBrRefVec refmask_trks_br(fNTrks);  // tbranch for each trk

  IntVec duplmask_trks(fNTrks);        // need to know if sim track associated to a given reco track type more than once
  TBrRefVec duplmask_trks_br(fNTrks);  // tbranch for each trk

  std::vector<ULong64_t> itermask_trks(fNTrks);
  TBrRefVec itermask_trks_br(fNTrks);

  std::vector<ULong64_t> iterduplmask_trks(fNTrks);
  TBrRefVec iterduplmask_trks_br(fNTrks);

  ULong64_t algoseed_trk;  // for SIMVALSEED
  TBranch* algoseed_trk_br;

  for (auto j = 0U; j < fNTrks; j++)  // loop over trks index
  {
    const auto& trk = fTrks[j];
    auto& refmask_trk = refmask_trks[j];
    auto& refmask_trk_br = refmask_trks_br[j];
    auto& duplmask_trk = duplmask_trks[j];
    auto& duplmask_trk_br = duplmask_trks_br[j];
    auto& itermask_trk = itermask_trks[j];
    auto& itermask_trk_br = itermask_trks_br[j];

    auto& iterduplmask_trk = iterduplmask_trks[j];
    auto& iterduplmask_trk_br = iterduplmask_trks_br[j];

    // initialize mcmask, branches
    refmask_trk = 0;
    refmask_trk_br = 0;

    // initialize duplmask, branches
    duplmask_trk = 0;
    duplmask_trk_br = 0;

    // initialize itermask, branches
    itermask_trk = 0;
    itermask_trk_br = 0;

    iterduplmask_trk = 0;
    iterduplmask_trk_br = 0;

    algoseed_trk = 0;
    algoseed_trk_br = 0;

    // Set branches
    efftree->SetBranchAddress(fSRefMask + "mask_" + trk, &refmask_trk, &refmask_trk_br);
    efftree->SetBranchAddress("duplmask_" + trk, &duplmask_trk, &duplmask_trk_br);
    efftree->SetBranchAddress("itermask_" + trk, &itermask_trk, &itermask_trk_br);
    efftree->SetBranchAddress("iterduplmask_" + trk, &iterduplmask_trk, &iterduplmask_trk_br);
  }
  efftree->SetBranchAddress("algo_seed", &algoseed_trk, &algoseed_trk_br);
  ///////////////////////////////////////////////////
  // Fill histos, compute rates from tree branches //
  ///////////////////////////////////////////////////

  // loop over entries
  const auto nentries = efftree->GetEntries();
  for (auto e = 0U; e < nentries; e++) {
    // get branches
    for (auto i = 0U; i < fNVars; i++) {
      auto& var_ref_br = vars_ref_br[i];

      var_ref_br->GetEntry(e);
    }
    for (auto j = 0U; j < fNTrks; j++) {
      auto& refmask_trk_br = refmask_trks_br[j];
      auto& duplmask_trk_br = duplmask_trks_br[j];
      auto& itermask_trk_br = itermask_trks_br[j];
      auto& iterduplmask_trk_br = iterduplmask_trks_br[j];

      refmask_trk_br->GetEntry(e);
      duplmask_trk_br->GetEntry(e);
      itermask_trk_br->GetEntry(e);
      iterduplmask_trk_br->GetEntry(e);
    }
    algoseed_trk_br->GetEntry(e);
    // use for cuts
    const auto pt_ref = vars_ref[0];

    // loop over plot indices
    for (auto k = 0U; k < fNPtCuts; k++)  // loop over pt cuts
    {
      const auto ptcut = fPtCuts[k];

      if (pt_ref < ptcut)
        continue;  // cut on tracks with a low pt

      for (auto i = 0U; i < fNVars; i++)  // loop over vars index
      {
        const auto var_ref = vars_ref[i];

        for (auto j = 0U; j < fNTrks; j++)  // loop over trks index
        {
          const auto refmask_trk = refmask_trks[j];
          const auto duplmask_trk = duplmask_trks[j];
          const auto itermask_trk = itermask_trks[j];
          const auto iterduplmask_trk = iterduplmask_trks[j];

          const auto effIteration = algo > 0 ? ((itermask_trk >> algo) & 1) : 1;
          const auto oneIteration = algo > 0 ? ((iterduplmask_trk >> algo) & 1) : 1;
          const auto ineffIteration = algo > 0 ? (((itermask_trk >> algo) & 1) == 0) : (refmask_trk == 0);
          const auto seedalgo_flag = (algoseed_trk > 0 && algo > 0) ? ((algoseed_trk >> algo) & 1) : 1;

          // plot key base
          const TString basekey = Form("%i_%i_%i", i, j, k);

          // efficiency calculation: need ref track to be findable
          if (refmask_trk != -1 && seedalgo_flag)
            plots[basekey + "_0"]->Fill((refmask_trk == 1) && effIteration,
                                        var_ref);  // ref track must be associated to enter numerator (==1)

          // duplicate rate calculation: need ref track to be matched at least once
          if (duplmask_trk != -1 && effIteration && seedalgo_flag)
            plots[basekey + "_1"]->Fill((duplmask_trk == 1) && oneIteration,
                                        var_ref);  // ref track is matched at least twice

          // inefficiency calculation: need ref track to be findable
          if (refmask_trk != -1) {
            for (auto m = 0U; m < regs.size(); m++) {
              const auto eta_ref = std::abs(vars_ref[1]);
              const auto etalow = etacuts[m];
              const auto etaup = etacuts[m + 1];

              // ref track must be UNassociated (==0) to enter numerator of inefficiency
              if ((eta_ref >= etalow) && (eta_ref < etaup))
                plots[Form("%s_2_%i", basekey.Data(), m)]->Fill(ineffIteration, var_ref);
            }  // end loop over regions
          }    // end check over ref tracks being findable

        }  // end loop over fPtCuts
      }    // end loop over fTrks
    }      // end loop over fVars
  }        // end loop over entry in tree

  /////////////////
  // Make output //
  /////////////////

  // make subdirs
  TStrVec dirnames = {"efficiency", "duplicaterate", "inefficiency"};
  for (auto& dirname : dirnames)
    dirname += fSRefDir;

  TDirRefVec subdirs(nrates);
  for (auto l = 0U; l < nrates; l++)
    subdirs[l] = PlotValidation::MakeSubDirs(dirnames[l]);

  // Draw, divide, and save efficiency plots
  for (auto i = 0U; i < fNVars; i++) {
    for (auto j = 0U; j < fNTrks; j++) {
      for (auto k = 0U; k < fNPtCuts; k++) {
        for (auto l = 0U; l < nrates; l++) {
          const auto& dirname = dirnames[l];
          auto& subdir = subdirs[l];

          const TString plotkey = Form("%i_%i_%i_%i", i, j, k, l);
          if (l < 2)  // efficiency and duplicate rate
          {
            PlotValidation::DrawWriteSavePlot(plots[plotkey], subdir, dirname, "AP");
            delete plots[plotkey];
          } else {
            for (auto m = 0U; m < nregs; m++) {
              const TString tmpkey = Form("%s_%i", plotkey.Data(), m);
              PlotValidation::DrawWriteSavePlot(plots[tmpkey], subdir, dirname, "AP");
              delete plots[tmpkey];
            }  // end loop over regions
          }    // end check over plots
        }      // end loop over plots
      }        // end loop over pt cuts
    }          // end loop over tracks
  }            // end loop over variables
}

// loop over fake rate tree, producing fake rate, nHits/track, score, and kinematic diffs to cmssw
void PlotValidation::PlotFRTree(int algo) {
  ////////////////////////////////////////////
  // Declare strings for branches and plots //
  ////////////////////////////////////////////

  // info for quality info (nHits,score), kinematic diffs
  const TStrVec colls = {"allreco", "fake", "allmatch", "bestmatch"};
  const TStrVec scolls = {"All Reco", "Fake", "All Match", "Best Match"};
  const UInt_t ncolls = colls.size();

  // get bins ready
  const DblVecVec trkqualbins = {fNHitsBins, fFracHitsBins, fScoreBins};

  // diffs
  const TStrVec dvars = {"dnHits", "dinvpt", "deta", "dphi"};
  const TStrVec sdvars = {"nHits", "1/p_{T}", "#eta", "#phi"};
  const UInt_t ndvars = dvars.size();

  // get bins ready
  const DblVecVec dvarbins = {fDNHitsBins, fDInvPtBins, fDEtaBins, fDPhiBins};

  //////////////////////////
  // Create and new plots //
  //////////////////////////

  TEffRefMap plots;
  TH1FRefMap hists;
  for (auto j = 0U; j < fNTrks; j++)  // loop over track collection
  {
    const auto& trk = fTrks[j];
    const auto& strk = fSTrks[j];

    for (auto k = 0U; k < fNPtCuts; k++)  // loop over pt cuts
    {
      const auto& sptcut = fSPtCuts[k];
      const auto& hptcut = fHPtCuts[k];

      // initialize efficiency plots
      for (auto i = 0U; i < fNVars; i++)  // loop over vars
      {
        const auto& var = fVars[i];
        const auto& svar = fSVars[i];
        const auto& sunit = fSUnits[i];

        // plot names and key
        const TString plotkey = Form("%i_%i_%i", i, j, k);
        const TString plotname = "fr_reco_" + var + "_" + trk + "_pt" + hptcut;
        const TString plottitle = strk + " Track Fake Rate vs Reco " + svar + " {" + fSVarPt + " > " + sptcut + " " +
                                  fSUnitPt + "};" + svar + sunit + ";Fake Rate";

        // get bins for the variable of interest
        const auto& varbins = fVarBins[i];
        const Double_t* bins = &varbins[0];

        plots[plotkey] = new TEfficiency(plotname.Data(), plottitle.Data(), varbins.size() - 1, bins);
      }  // end loop over vars for efficiency

      // initialize track quality plots
      for (auto n = 0U; n < fNTrkQual; n++)  // loop over quality vars
      {
        const auto& trkqual = fTrkQual[n];
        const auto& strkqual = fSTrkQual[n];

        // get bins for the variable of interest
        const auto& varbins = trkqualbins[n];
        const Double_t* bins = &varbins[0];

        for (auto o = 0U; o < ncolls; o++)  // loop over collection of tracks
        {
          const auto& coll = colls[o];
          const auto& scoll = scolls[o];

          // plot names and key
          const TString histkey = Form("%i_%i_%i_%i", j, k, n, o);
          const TString histname = "h_" + trkqual + "_" + coll + "_" + trk + "_pt" + hptcut;
          const TString histtitle = scoll + " " + strk + " Track vs " + strkqual + " {" + fSVarPt + " > " + sptcut +
                                    " " + fSUnitPt + "};" + strkqual + ";nTracks";

          // Numerator only type plots only!
          hists[histkey] = new TH1F(histname.Data(), histtitle.Data(), varbins.size() - 1, bins);
          hists[histkey]->Sumw2();
        }  // end loop over tracks collections
      }    // end loop over hit plots

      // initialize diff plots
      for (auto p = 0U; p < ndvars; p++)  // loop over kin diff vars
      {
        const auto& dvar = dvars[p];
        const auto& sdvar = sdvars[p];

        // get bins for the variable of interest
        const auto& varbins = dvarbins[p];
        const Double_t* bins = &varbins[0];

        // loop over collection of tracks for only matched tracks
        for (auto o = 2U; o < ncolls; o++) {
          const auto& coll = colls[o];
          const auto& scoll = scolls[o];

          // plot names and key
          const TString histkey = Form("%i_%i_d_%i_%i", j, k, p, o);
          const TString histname = "h_" + dvar + "_" + coll + "_" + trk + "_pt" + hptcut;
          const TString histtitle = "#Delta" + sdvar + "(" + scoll + " " + strk + "," + fSRefTitle + ") {" + fSVarPt +
                                    " > " + sptcut + " " + fSUnitPt + "};" + sdvar + "^{" + scoll + " " + strk + "}-" +
                                    sdvar + "^{" + fSRefTitle + "};nTracks";

          // Numerator only type plots only!
          hists[histkey] = new TH1F(histname.Data(), histtitle.Data(), varbins.size() - 1, bins);
          hists[histkey]->Sumw2();
        }  // end loop over track collections
      }    // end loop over diff plots

    }  // end loop over pt cuts
  }    // end loop over tracks

  ////////////////////////////////////////
  // Floats/Ints to be filled for trees //
  ////////////////////////////////////////

  // Initialize var_trk arrays, SetBranchAddress
  FltVecVec vars_trks(fNVars);        // first index is var, second is type of reco track
  TBrRefVecVec vars_trks_br(fNVars);  // tbranch for each var
  for (auto i = 0U; i < fNVars; i++)  // loop over vars index
  {
    const auto& var = fVars[i];
    auto& var_trks = vars_trks[i];
    auto& var_trks_br = vars_trks_br[i];

    var_trks.resize(fNTrks);
    var_trks_br.resize(fNTrks);

    for (auto j = 0U; j < fNTrks; j++)  // loop over trks index
    {
      const auto& trk = fTrks[j];
      auto& var_trk = var_trks[j];
      auto& var_trk_br = var_trks_br[j];

      // initialize var, branches
      var_trk = 0.;
      var_trk_br = 0;

      //Set var+trk branch
      frtree->SetBranchAddress(var + "_" + trk, &var_trk, &var_trk_br);
    }  // end loop over tracks
  }    // end loop over vars

  // Initialize masks
  IntVec refmask_trks(fNTrks);           // need to know if ref track associated to a given reco track type
  TBrRefVec refmask_trks_br(fNTrks);     // tbranch for each trk
  IntVec iTkMatches_trks(fNTrks);        // want which matched track!
  TBrRefVec iTkMatches_trks_br(fNTrks);  // tbranch for each trk

  // Initialize nhits_trk branches
  IntVec nHits_trks(fNTrks);           // nHits / track
  TBrRefVec nHits_trks_br(fNTrks);     // branch per track
  FltVec fracHits_trks(fNTrks);        // fraction of hits matched (most) / track
  TBrRefVec fracHits_trks_br(fNTrks);  // branch per track
  IntVec score_trks(fNTrks);           // track score
  TBrRefVec score_trks_br(fNTrks);     // branch per track

  // Initialize diff branches
  FltVec nLayers_ref_trks(fNTrks);  // sim/cmssw nUnique layers
  TBrRefVec nLayers_ref_trks_br(fNTrks);
  FltVec pt_ref_trks(fNTrks);  // sim/cmssw pt
  TBrRefVec pt_ref_trks_br(fNTrks);
  FltVec eta_ref_trks(fNTrks);  // cmssw eta
  TBrRefVec eta_ref_trks_br(fNTrks);
  FltVec dphi_trks(fNTrks);  // dphi between reco track and sim/cmssw (computed during matching --> not 100% ideal)
  TBrRefVec dphi_trks_br(fNTrks);

  // Set branches for tracks
  for (auto j = 0U; j < fNTrks; j++)  // loop over trks index
  {
    const auto& trk = fTrks[j];
    auto& refmask_trk = refmask_trks[j];
    auto& refmask_trk_br = refmask_trks_br[j];
    auto& iTkMatches_trk = iTkMatches_trks[j];
    auto& iTkMatches_trk_br = iTkMatches_trks_br[j];
    auto& nHits_trk = nHits_trks[j];
    auto& nHits_trk_br = nHits_trks_br[j];
    auto& fracHits_trk = fracHits_trks[j];
    auto& fracHits_trk_br = fracHits_trks_br[j];
    auto& score_trk = score_trks[j];
    auto& score_trk_br = score_trks_br[j];
    auto& nLayers_ref_trk = nLayers_ref_trks[j];
    auto& nLayers_ref_trk_br = nLayers_ref_trks_br[j];
    auto& pt_ref_trk = pt_ref_trks[j];
    auto& pt_ref_trk_br = pt_ref_trks_br[j];
    auto& eta_ref_trk = eta_ref_trks[j];
    auto& eta_ref_trk_br = eta_ref_trks_br[j];
    auto& dphi_trk = dphi_trks[j];
    auto& dphi_trk_br = dphi_trks_br[j];

    // initialize masks, branches
    refmask_trk = 0;
    refmask_trk_br = 0;
    iTkMatches_trk = 0;
    iTkMatches_trk_br = 0;

    // initialize nHits, branches
    nHits_trk = 0;
    nHits_trk_br = 0;
    fracHits_trk = 0.f;
    fracHits_trk_br = 0;
    score_trk = 0;
    score_trk_br = 0;

    // initialize diff branches
    nLayers_ref_trk = 0;
    nLayers_ref_trk_br = 0;
    pt_ref_trk = 0.f;
    pt_ref_trk_br = 0;
    eta_ref_trk = 0.f;
    eta_ref_trk_br = 0;
    dphi_trk = 0.f;
    dphi_trk_br = 0;

    // Set Branches
    frtree->SetBranchAddress(fSRefMask + "mask_" + trk, &refmask_trk, &refmask_trk_br);
    frtree->SetBranchAddress("iTkMatches_" + trk, &iTkMatches_trk, &iTkMatches_trk_br);

    frtree->SetBranchAddress("nHits_" + trk, &nHits_trk, &nHits_trk_br);
    frtree->SetBranchAddress("fracHitsMatched_" + trk, &fracHits_trk, &fracHits_trk_br);
    frtree->SetBranchAddress("score_" + trk, &score_trk, &score_trk_br);

    frtree->SetBranchAddress("nLayers_" + fSRefVarTrk + "_" + trk, &nLayers_ref_trk, &nLayers_ref_trk_br);
    frtree->SetBranchAddress("pt_" + fSRefVarTrk + "_" + trk, &pt_ref_trk, &pt_ref_trk_br);
    frtree->SetBranchAddress("eta_" + fSRefVarTrk + "_" + trk, &eta_ref_trk, &eta_ref_trk_br);
    frtree->SetBranchAddress("dphi_" + trk, &dphi_trk, &dphi_trk_br);
  }

  ///////////////////////////////////////////////////
  // Fill histos, compute rates from tree branches //
  ///////////////////////////////////////////////////

  // loop over entries
  const UInt_t nentries = frtree->GetEntries();
  for (auto e = 0U; e < nentries; e++) {
    // get branches
    for (auto i = 0U; i < fNVars; i++) {
      auto& var_trks_br = vars_trks_br[i];
      for (auto j = 0U; j < fNTrks; j++) {
        auto& var_trk_br = var_trks_br[j];

        var_trk_br->GetEntry(e);
      }
    }
    for (auto j = 0U; j < fNTrks; j++) {
      auto& refmask_trk_br = refmask_trks_br[j];
      auto& iTkMatches_trk_br = iTkMatches_trks_br[j];
      auto& nHits_trk_br = nHits_trks_br[j];
      auto& fracHits_trk_br = fracHits_trks_br[j];
      auto& score_trk_br = score_trks_br[j];
      auto& nLayers_ref_trk_br = nLayers_ref_trks_br[j];
      auto& pt_ref_trk_br = pt_ref_trks_br[j];
      auto& eta_ref_trk_br = eta_ref_trks_br[j];
      auto& dphi_trk_br = dphi_trks_br[j];

      refmask_trk_br->GetEntry(e);
      iTkMatches_trk_br->GetEntry(e);

      nHits_trk_br->GetEntry(e);
      fracHits_trk_br->GetEntry(e);
      score_trk_br->GetEntry(e);

      nLayers_ref_trk_br->GetEntry(e);
      pt_ref_trk_br->GetEntry(e);
      eta_ref_trk_br->GetEntry(e);
      dphi_trk_br->GetEntry(e);
    }

    // loop over plot indices
    for (auto j = 0U; j < fNTrks; j++)  // loop over trks index
    {
      const auto pt_trk = vars_trks[0][j];
      const auto eta_trk = vars_trks[1][j];
      const auto phi_trk = vars_trks[2][j];

      const auto refmask_trk = refmask_trks[j];
      const auto iTkMatches_trk = iTkMatches_trks[j];
      const auto nHits_trk = nHits_trks[j];
      const auto fracHits_trk = fracHits_trks[j];
      const auto score_trk = score_trks[j];

      const auto nLayers_ref_trk = nLayers_ref_trks[j];
      const auto pt_ref_trk = pt_ref_trks[j];
      const auto eta_ref_trk = eta_ref_trks[j];
      const auto dphi_trk = dphi_trks[j];

      for (auto k = 0U; k < fNPtCuts; k++)  // loop over pt cuts
      {
        const auto ptcut = fPtCuts[k];

        if (pt_trk < ptcut)
          continue;  // cut on tracks with a low pt

        // fill rate plots
        for (auto i = 0U; i < fNVars; i++)  // loop over vars index
        {
          const auto var_trk = vars_trks[i][j];

          // plot key
          const TString plotkey = Form("%i_%i_%i", i, j, k);

          // can include masks of 1,0,2 to enter denominator
          if (refmask_trk >= 0)
            plots[plotkey]->Fill((refmask_trk == 0), var_trk);  // only completely unassociated reco tracks enter FR
        }                                                       // end loop over vars

        // base hist key
        const TString basekey = Form("%i_%i", j, k);  // hist key

        // key strings
        const TString nhitkey = Form("%s_0", basekey.Data());
        const TString frackey = Form("%s_1", basekey.Data());
        const TString scorekey = Form("%s_2", basekey.Data());

        const TString dnhitkey = Form("%s_d_0", basekey.Data());
        const TString dinvptkey = Form("%s_d_1", basekey.Data());
        const TString detakey = Form("%s_d_2", basekey.Data());
        const TString dphikey = Form("%s_d_3", basekey.Data());

        // all reco
        hists[Form("%s_0", nhitkey.Data())]->Fill(nHits_trk);
        hists[Form("%s_0", frackey.Data())]->Fill(fracHits_trk);
        hists[Form("%s_0", scorekey.Data())]->Fill(score_trk);

        if (refmask_trk == 0)  // all fakes
        {
          hists[Form("%s_1", nhitkey.Data())]->Fill(nHits_trk);
          hists[Form("%s_1", frackey.Data())]->Fill(fracHits_trk);
          hists[Form("%s_1", scorekey.Data())]->Fill(score_trk);
        } else if (refmask_trk == 1)  // all matches
        {
          hists[Form("%s_2", nhitkey.Data())]->Fill(nHits_trk);
          hists[Form("%s_2", frackey.Data())]->Fill(fracHits_trk);
          hists[Form("%s_2", scorekey.Data())]->Fill(score_trk);

          hists[Form("%s_2", dnhitkey.Data())]->Fill(nHits_trk - (Int_t)nLayers_ref_trk);
          hists[Form("%s_2", dinvptkey.Data())]->Fill(1.f / pt_trk - 1.f / pt_ref_trk);
          hists[Form("%s_2", detakey.Data())]->Fill(eta_trk - eta_ref_trk);
          hists[Form("%s_2", dphikey.Data())]->Fill(dphi_trk);

          if (iTkMatches_trk == 0)  // best matches only
          {
            hists[Form("%s_3", nhitkey.Data())]->Fill(nHits_trk);
            hists[Form("%s_3", frackey.Data())]->Fill(fracHits_trk);
            hists[Form("%s_3", scorekey.Data())]->Fill(score_trk);

            hists[Form("%s_3", dnhitkey.Data())]->Fill(nHits_trk - (Int_t)nLayers_ref_trk);
            hists[Form("%s_3", dinvptkey.Data())]->Fill(1.f / pt_trk - 1.f / pt_ref_trk);
            hists[Form("%s_3", detakey.Data())]->Fill(eta_trk - eta_ref_trk);
            hists[Form("%s_3", dphikey.Data())]->Fill(dphi_trk);
          }  // end check over best matches
        }    // end check over all matches
      }      // end loop over pt cuts
    }        // end loop over trks
  }          // end loop over entry in tree

  /////////////////
  // Make output //
  /////////////////

  // make subdirs
  TStrVec dirnames = {"fakerate", "quality", "kindiffs"};
  for (auto& dirname : dirnames)
    dirname += fSRefDir;
  const UInt_t ndirs = dirnames.size();

  TDirRefVec subdirs(ndirs);
  for (auto q = 0U; q < ndirs; q++)
    subdirs[q] = PlotValidation::MakeSubDirs(dirnames[q]);

  // Draw, divide, and save fake rate plots --> then delete!
  for (auto j = 0U; j < fNTrks; j++)  // loop over trks
  {
    for (auto k = 0U; k < fNPtCuts; k++)  // loop over pt cuts
    {
      // fake rate plots
      for (auto i = 0U; i < fNVars; i++)  // loop over vars
      {
        const Int_t diridx = 0;
        const TString plotkey = Form("%i_%i_%i", i, j, k);
        PlotValidation::DrawWriteSavePlot(plots[plotkey], subdirs[diridx], dirnames[diridx], "AP");
        delete plots[plotkey];
      }

      // track quality plots
      for (auto n = 0U; n < fNTrkQual; n++)  // loop over track quality vars
      {
        for (auto o = 0U; o < ncolls; o++)  // loop over collection of tracks
        {
          const Int_t diridx = 1;
          const TString histkey = Form("%i_%i_%i_%i", j, k, n, o);
          PlotValidation::DrawWriteSavePlot(hists[histkey], subdirs[diridx], dirnames[diridx], "");
          delete hists[histkey];
        }  // end loop over track collections
      }    // end loop over hit vars

      // kinematic diff plots
      for (auto p = 0U; p < ndvars; p++)  // loop over diff vars
      {
        for (auto o = 2U; o < ncolls; o++)  // loop over collection of tracks for only matched tracks
        {
          const Int_t diridx = 2;
          const TString histkey = Form("%i_%i_d_%i_%i", j, k, p, o);
          PlotValidation::DrawWriteSavePlot(hists[histkey], subdirs[diridx], dirnames[diridx], "");
          delete hists[histkey];
        }  // end loop over track collections
      }    // end loop over diff plots

    }  // end loop over pt cuts
  }    // end loop over tracks
}

void PlotValidation::PrintTotals(int algo) {
  ///////////////////////////////////////////////
  // Get number of events and number of tracks //
  ///////////////////////////////////////////////

  Int_t Nevents = 0;
  Int_t evtID = 0;
  TBranch* b_evtID = 0;
  efftree->SetBranchAddress("evtID", &evtID, &b_evtID);
  const UInt_t nentries = efftree->GetEntries();
  for (auto e = 0U; e < nentries; e++) {
    b_evtID->GetEntry(e);
    if (evtID > Nevents)
      Nevents = evtID;
  }

  const Int_t NtracksMC = efftree->GetEntries();
  const Float_t ntkspevMC = Float_t(NtracksMC) / Float_t(Nevents);
  const Int_t NtracksReco = frtree->GetEntries();
  const Float_t ntkspevReco = Float_t(NtracksReco) / Float_t(Nevents);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Print out totals of nHits, frac of Hits shared, track score, eff, FR, DR rate of seeds, build, fit //
  //                --> numer/denom plots for phi, know it will be in the bounds.                       //
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  const TStrVec rates = {"eff", "fr", "dr"};
  const TStrVec srates = {"Efficiency", "Fake Rate", "Duplicate Rate"};
  const TStrVec dirnames = {"efficiency", "fakerate", "duplicaterate"};
  const TStrVec types = (fCmsswComp ? TStrVec{"cmssw", "reco", "cmssw"}
                                    : TStrVec{"sim", "reco", "sim"});  // types will be same size as rates!
  const UInt_t nrates = rates.size();

  const TStrVec snumers = {
      fSRefTitle + " Tracks Matched", "Unmatched Reco Tracks", fSRefTitle + " Tracks Matched (nTimes>1)"};
  const TStrVec sdenoms = {
      "Eligible " + fSRefTitle + " Tracks", "Eligible Reco Tracks", "Eligible " + fSRefTitle + " Tracks"};

  TEffRefMap plots;
  for (auto j = 0U; j < fNTrks; j++) {
    const auto& trk = fTrks[j];

    for (auto k = 0U; k < fNPtCuts; k++) {
      const auto& hptcut = fHPtCuts[k];

      for (auto l = 0U; l < nrates; l++) {
        const auto& rate = rates[l];
        const auto& type = types[l];
        const auto& dirname = dirnames[l];

        const TString plotkey = Form("%i_%i_%i", j, k, l);
        const TString plotname = dirname + fSRefDir + "/" + rate + "_" + type + "_phi_" + trk + "_pt" + hptcut;
        plots[plotkey] = (TEfficiency*)fOutRoot->Get(plotname.Data());
      }
    }
  }

  // want nHits plots for (nearly) all types of tracks
  const TStrVec colls = {"allreco", "fake", "bestmatch"};
  const TStrVec scolls = {"All Reco", "Fake", "Best Match"};
  const UInt_t ncolls = colls.size();

  TH1FRefMap hists;
  for (auto j = 0U; j < fNTrks; j++) {
    const auto& trk = fTrks[j];

    for (auto k = 0U; k < fNPtCuts; k++) {
      const auto& hptcut = fHPtCuts[k];

      for (auto n = 0U; n < fNTrkQual; n++) {
        const auto& trkqual = fTrkQual[n];

        for (auto o = 0U; o < ncolls; o++) {
          const auto& coll = colls[o];

          const TString histkey = Form("%i_%i_%i_%i", j, k, n, o);
          const TString histname = "quality" + fSRefDir + "/h_" + trkqual + "_" + coll + "_" + trk + "_pt" + hptcut;
          hists[histkey] = (TH1F*)fOutRoot->Get(histname.Data());
        }
      }
    }
  }

  // setup output stream
  const TString outfilename = fOutName + "/totals_" + fOutName + fSRefOut + ".txt";
  std::ofstream totalsout(outfilename.Data());

  std::cout << "--------Track Reconstruction Summary--------" << std::endl;
  std::cout << "nEvents: " << Nevents << Form(" n%sTracks/evt: ", fSRefTitle.Data()) << ntkspevMC
            << " nRecoTracks/evt: " << ntkspevReco << std::endl;
  std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << std::endl;

  totalsout << "--------Track Reconstruction Summary--------" << std::endl;
  totalsout << "nEvents: " << Nevents << Form(" n%sTracks/evt: ", fSRefTitle.Data()) << ntkspevMC
            << " nRecoTracks/evt: " << ntkspevReco << std::endl;
  totalsout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  totalsout << std::endl;

  for (auto k = 0U; k < fNPtCuts; k++) {
    const auto& ptcut = fPtCuts[k];

    std::cout << Form("xxxxxxxxxx Track pT > %3.1f Cut xxxxxxxxxx", ptcut) << std::endl;
    std::cout << std::endl;

    totalsout << Form("xxxxxxxxxx Track pT > %3.1f Cut xxxxxxxxxx", ptcut) << std::endl;
    totalsout << std::endl;

    for (auto j = 0U; j < fNTrks; j++) {
      const auto& strk = fSTrks[j];

      std::cout << strk.Data() << " Tracks" << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++" << std::endl << std::endl;
      std::cout << "Quality Info for " << strk.Data() << " Track Collections" << std::endl;
      std::cout << "==========================================" << std::endl;

      totalsout << strk.Data() << " Tracks" << std::endl;
      totalsout << "++++++++++++++++++++++++++++++++++++++++++" << std::endl << std::endl;
      totalsout << "Quality Info for " << strk.Data() << " Track Collections" << std::endl;
      totalsout << "==========================================" << std::endl;
      for (auto o = 0U; o < ncolls; o++) {
        const auto& scoll = scolls[o];

        const Float_t nHits_mean = hists[Form("%i_%i_0_%i", j, k, o)]->GetMean(1);           // 1 is mean of x-axis
        const Float_t nHits_mean_unc = hists[Form("%i_%i_0_%i", j, k, o)]->GetMeanError(1);  // 1 is mean of x-axis
        const Float_t fracHits_mean = hists[Form("%i_%i_1_%i", j, k, o)]->GetMean(1);
        const Float_t fracHits_mean_unc = hists[Form("%i_%i_1_%i", j, k, o)]->GetMeanError(1);
        const Float_t score_mean = hists[Form("%i_%i_2_%i", j, k, o)]->GetMean(1);
        const Float_t score_mean_unc = hists[Form("%i_%i_2_%i", j, k, o)]->GetMeanError(1);

        std::cout << scoll.Data() << " Tracks" << std::endl;
        std::cout << "Mean nHits / Track = " << nHits_mean << " +/- " << nHits_mean_unc << std::endl;
        std::cout << "Mean Shared Hits / Track = " << fracHits_mean << " +/- " << fracHits_mean_unc << std::endl;
        std::cout << "Mean Track Score = " << score_mean << " +/- " << score_mean_unc << std::endl;
        std::cout << "------------------------------------------" << std::endl;

        totalsout << scoll.Data() << " Tracks" << std::endl;
        totalsout << "Mean nHits / Track = " << nHits_mean << " +/- " << nHits_mean_unc << std::endl;
        totalsout << "Mean Shared Hits / Track = " << fracHits_mean << " +/- " << fracHits_mean_unc << std::endl;
        totalsout << "Mean Track Score = " << score_mean << " +/- " << score_mean_unc << std::endl;
        totalsout << "------------------------------------------" << std::endl;
      }

      std::cout << std::endl << "Rates for " << strk.Data() << " Tracks" << std::endl;
      std::cout << "==========================================" << std::endl;

      totalsout << std::endl << "Rates for " << strk.Data() << " Tracks" << std::endl;
      totalsout << "==========================================" << std::endl;
      for (auto l = 0U; l < nrates; l++) {
        const auto& snumer = snumers[l];
        const auto& sdenom = sdenoms[l];
        const auto& srate = srates[l];

        EffStruct effs;
        PlotValidation::GetTotalEfficiency(plots[Form("%i_%i_%i", j, k, l)], effs);

        std::cout << snumer.Data() << ": " << effs.passed_ << std::endl;
        std::cout << sdenom.Data() << ": " << effs.total_ << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        std::cout << srate.Data() << ": " << effs.eff_ << ", -" << effs.elow_ << ", +" << effs.eup_ << std::endl;
        std::cout << "------------------------------------------" << std::endl;

        totalsout << snumer.Data() << ": " << effs.passed_ << std::endl;
        totalsout << sdenom.Data() << ": " << effs.total_ << std::endl;
        totalsout << "------------------------------------------" << std::endl;
        totalsout << srate.Data() << ": " << effs.eff_ << ", -" << effs.elow_ << ", +" << effs.eup_ << std::endl;
        totalsout << "------------------------------------------" << std::endl;
      }
      std::cout << std::endl << std::endl;
      totalsout << std::endl << std::endl;
    }
  }

  // delete everything
  for (auto& hist : hists)
    delete hist.second;
  for (auto& plot : plots)
    delete plot.second;
}

template <typename T>
void PlotValidation::DrawWriteSavePlot(T*& plot, TDirectory*& subdir, const TString& subdirname, const TString& option) {
  // cd into root subdir and save
  subdir->cd();
  plot->SetDirectory(subdir);
  plot->Write(plot->GetName(), TObject::kWriteDelete);

  // draw it
  if (fSaveAs) {
    auto canv = new TCanvas();
    canv->cd();
    plot->Draw(option.Data());

    // first save log
    canv->SetLogy(1);
    canv->SaveAs(Form("%s/%s/log/%s.%s", fOutName.Data(), subdirname.Data(), plot->GetName(), fOutType.Data()));

    // then lin
    canv->SetLogy(0);
    canv->SaveAs(Form("%s/%s/lin/%s.%s", fOutName.Data(), subdirname.Data(), plot->GetName(), fOutType.Data()));

    delete canv;
  }
}

void PlotValidation::GetTotalEfficiency(const TEfficiency* eff, EffStruct& effs) {
  effs.passed_ = eff->GetPassedHistogram()->Integral();
  effs.total_ = eff->GetTotalHistogram()->Integral();

  auto tmp_eff = new TEfficiency("tmp_eff", "tmp_eff", 1, 0, 1);
  tmp_eff->SetTotalEvents(1, effs.total_);
  tmp_eff->SetPassedEvents(1, effs.passed_);

  effs.eff_ = tmp_eff->GetEfficiency(1);
  effs.elow_ = tmp_eff->GetEfficiencyErrorLow(1);
  effs.eup_ = tmp_eff->GetEfficiencyErrorUp(1);

  delete tmp_eff;
}

void PlotValidation::MakeOutDir(const TString& outdirname) {
  // make output directory
  FileStat_t dummyFileStat;
  if (gSystem->GetPathInfo(outdirname.Data(), dummyFileStat) == 1) {
    const TString mkDir = "mkdir -p " + outdirname;
    gSystem->Exec(mkDir.Data());
  }
}

void PlotValidation::MoveInput() {
  const TString mvin = "mv " + fInName + " " + fOutName;
  gSystem->Exec(mvin.Data());
}

TDirectory* PlotValidation::MakeSubDirs(const TString& subdirname) {
  PlotValidation::MakeOutDir(fOutName + "/" + subdirname);
  PlotValidation::MakeOutDir(fOutName + "/" + subdirname + "/lin");
  PlotValidation::MakeOutDir(fOutName + "/" + subdirname + "/log");

  return fOutRoot->mkdir(subdirname.Data());
}

void PlotValidation::SetupStyle() {
  // General style
  gROOT->Reset();
  gStyle->SetOptStat("emou");
  gStyle->SetTitleFontSize(0.04);
  gStyle->SetOptFit(1011);
  gStyle->SetStatX(0.9);
  gStyle->SetStatW(0.1);
  gStyle->SetStatY(1.0);
  gStyle->SetStatH(0.08);
}

void PlotValidation::SetupBins() {
  // pt bins
  PlotValidation::SetupVariableBins(
      "0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.5 3 3.5 4 4.5 5 5 6 7 8 9 10 15 20 25 30 40 50 100 200 500 1000", fPtBins);

  // eta bins
  PlotValidation::SetupFixedBins(60, -3, 3, fEtaBins);

  // phi bins
  PlotValidation::SetupFixedBins(70, -3.5, 3.5, fPhiBins);

  // nLayers bins
  PlotValidation::SetupFixedBins(26, -0.5, 25.5, fNLayersBins);

  // nHits bins
  PlotValidation::SetupFixedBins(40, 0, 40, fNHitsBins);

  // fraction hits matched bins
  PlotValidation::SetupFixedBins(110, 0, 1.1, fFracHitsBins);

  // track score bins
  PlotValidation::SetupFixedBins(50, -500, 5000, fScoreBins);

  // dNhits
  PlotValidation::SetupFixedBins(40, -20, 20, fDNHitsBins);

  // dinvpt
  PlotValidation::SetupFixedBins(45, -1.0, 1.0, fDInvPtBins);

  // dphi
  PlotValidation::SetupFixedBins(45, -0.1, 0.1, fDPhiBins);

  // deta
  PlotValidation::SetupFixedBins(45, -0.1, 0.1, fDEtaBins);
}

void PlotValidation::SetupVariableBins(const std::string& s_bins, DblVec& bins) {
  std::stringstream ss(s_bins);
  Double_t boundary;
  while (ss >> boundary)
    bins.emplace_back(boundary);
}

void PlotValidation::SetupFixedBins(const UInt_t nBins, const Double_t low, const Double_t high, DblVec& bins) {
  const Double_t width = (high - low) / nBins;

  for (auto i = 0U; i <= nBins; i++)
    bins.emplace_back(i * width + low);
}

void PlotValidation::SetupCommonVars() {
  // common kinematic variables
  fVars = {"pt", "eta", "phi", "nLayers"};
  fSVars = {"p_{T}", "#eta", "#phi", "Number of layers"};  // svars --> labels for histograms for given variable
  fSUnits = {"GeV/c", "", "", ""};                         // units --> labels for histograms for given variable
  fNVars = fVars.size();

  fSVarPt = fSVars[0];
  fSUnitPt = fSUnits[0];

  // add square brackets around units
  for (auto& sunit : fSUnits) {
    if (!sunit.EqualTo("")) {
      sunit.Prepend(" [");
      sunit.Append("]");
    }
  }

  // get bins ready for rate variables
  fVarBins = {fPtBins, fEtaBins, fPhiBins, fNLayersBins};

  // which tracks to use
  fTrks = (fCmsswComp ? TStrVec{"build", "fit"} : TStrVec{"seed", "build", "fit"});
  fSTrks = (fCmsswComp ? TStrVec{"Build", "Fit"}
                       : TStrVec{"Seed", "Build", "Fit"});  // strk --> labels for histograms for given track type
  fNTrks = fTrks.size();

  // which pt cuts
  fPtCuts = {0.f, 0.9f, 2.f};
  for (const auto ptcut : fPtCuts) {
    fSPtCuts.emplace_back(Form("%3.1f", ptcut));
  }
  for (const auto& sptcut : fSPtCuts) {
    TString hptcut = sptcut;
    hptcut.ReplaceAll(".", "p");
    fHPtCuts.emplace_back(hptcut);
  }
  fNPtCuts = fPtCuts.size();

  // quality info
  fTrkQual = {"nHits", "fracHitsMatched", "score"};
  fSTrkQual = {"nHits / Track", "Highest Fraction of Matched Hits / Track", "Track Score"};
  fNTrkQual = fTrkQual.size();

  // reference related strings
  fSRefTitle = (fCmsswComp ? "CMSSW" : "Sim");
  fSRefVar = (fCmsswComp ? "cmssw" : "mc_gen");
  fSRefMask = (fCmsswComp ? "cmssw" : "mc");
  fSRefVarTrk = (fCmsswComp ? "cmssw" : "mc");
  fSRefDir = (fCmsswComp ? "_cmssw" : "");
  fSRefOut = (fCmsswComp ? "_cmssw" : "");
}
