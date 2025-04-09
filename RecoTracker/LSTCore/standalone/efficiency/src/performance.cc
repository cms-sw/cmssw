#include "performance.h"

enum { pT5 = 7, pT3 = 5, T5 = 4, pLS = 8 };

//__________________________________________________________________________________________________________________________________________________________________________
int main(int argc, char** argv) {
  // Parse arguments
  parseArguments(argc, argv);

  // Initialize input and output root files
  initializeInputsAndOutputs();

  // Set of pdgids
  std::vector<int> pdgids = {0, 11, 211, 13, 321};

  // Set of charges
  std::vector<int> charges = {0, 1, -1};

  // Set of extra selections for efficiency plots
  std::vector<TString> selnames = {
      "base",    // default baseline that is more inline with MTV
      "loweta",  // When the eta cut is restricted to 2.4
      "xtr",     // When the eta cut is restricted to transition regions
      "vtr"      // When the eta cut is vetoing transition regions
  };
  std::vector<std::function<bool(unsigned int)>> sels = {
      [&](unsigned int isim) { return 1.; },
      [&](unsigned int isim) { return abs(lstEff.sim_eta().at(isim)) < 2.4; },
      [&](unsigned int isim) { return abs(lstEff.sim_eta().at(isim)) > 1.1 and abs(lstEff.sim_eta().at(isim)) < 1.7; },
      [&](unsigned int isim) {
        return (abs(lstEff.sim_eta().at(isim)) < 1.1 or abs(lstEff.sim_eta().at(isim)) > 1.7) and
               abs(lstEff.sim_eta().at(isim)) < 2.4;
      }};
  pdgids.insert(pdgids.end(), ana.pdgids.begin(), ana.pdgids.end());

  std::vector<SimTrackSetDefinition> list_effSetDef;

  // creating a set of efficiency plots for each pdgids being considered
  for (auto& pdgid : pdgids) {
    for (auto& charge : charges) {
      for (unsigned int isel = 0; isel < sels.size(); ++isel) {
        list_effSetDef.push_back(
            SimTrackSetDefinition(/* name  */
                                  TString("TC_") + selnames[isel],
                                  /* pdgid */ pdgid,
                                  /* q     */ charge,
                                  /* pass  */ [&](unsigned int isim) { return lstEff.sim_TC_matched().at(isim) > 0; },
                                  /* sel   */ sels[isel]));
        list_effSetDef.push_back(SimTrackSetDefinition(
            /* name  */
            TString("pT5_") + selnames[isel],
            /* pdgid */ pdgid,
            /* q     */ charge,
            /* pass  */ [&](unsigned int isim) { return lstEff.sim_TC_matched_mask().at(isim) & (1 << pT5); },
            /* sel   */ sels[isel]));
        list_effSetDef.push_back(SimTrackSetDefinition(
            /* name  */
            TString("pT3_") + selnames[isel],
            /* pdgid */ pdgid,
            /* q     */ charge,
            /* pass  */ [&](unsigned int isim) { return lstEff.sim_TC_matched_mask().at(isim) & (1 << pT3); },
            /* sel   */ sels[isel]));
        list_effSetDef.push_back(SimTrackSetDefinition(
            /* name  */
            TString("T5_") + selnames[isel],
            /* pdgid */ pdgid,
            /* q     */ charge,
            /* pass  */ [&](unsigned int isim) { return lstEff.sim_TC_matched_mask().at(isim) & (1 << T5); },
            /* sel   */ sels[isel]));
        list_effSetDef.push_back(SimTrackSetDefinition(
            /* name  */
            TString("pLS_") + selnames[isel],
            /* pdgid */ pdgid,
            /* q     */ charge,
            /* pass  */ [&](unsigned int isim) { return lstEff.sim_TC_matched_mask().at(isim) & (1 << pLS); },
            /* sel   */ sels[isel]));

        if (ana.do_lower_level) {
          //lower objects - name will have pT5_lower_, T5_lower_, pT3_lower_
          list_effSetDef.push_back(SimTrackSetDefinition(
              /* name  */
              TString("pT5_lower_") + selnames[isel],
              /* pdgid */ pdgid,
              /* q     */ charge,
              /* pass  */ [&](unsigned int isim) { return lstEff.sim_pT5_matched().at(isim) > 0; },
              /* sel   */ sels[isel]));
          list_effSetDef.push_back(
              SimTrackSetDefinition(/* name  */
                                    TString("T5_lower_") + selnames[isel],
                                    /* pdgid */ pdgid,
                                    /* q     */ charge,
                                    /* pass  */ [&](unsigned int isim) { return lstEff.sim_T5_matched().at(isim) > 0; },
                                    /* sel   */ sels[isel]));
          list_effSetDef.push_back(SimTrackSetDefinition(
              /* name  */
              TString("pT3_lower_") + selnames[isel],
              /* pdgid */ pdgid,
              /* q     */ charge,
              /* pass  */ [&](unsigned int isim) { return lstEff.sim_pT3_matched().at(isim) > 0; },
              /* sel   */ sels[isel]));
        }
      }
    }
  }

  bookEfficiencySets(list_effSetDef);

  // creating a set of fake rate plots
  std::vector<RecoTrackSetDefinition> list_FRSetDef;
  list_FRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "TC",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isFake().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return 1; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));
  list_FRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "pT5",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isFake().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return lstEff.tc_type().at(itc) == pT5; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));
  list_FRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "pT3",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isFake().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return lstEff.tc_type().at(itc) == pT3; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));
  list_FRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "T5",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isFake().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return lstEff.tc_type().at(itc) == T5; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));
  list_FRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "pLS",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isFake().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return lstEff.tc_type().at(itc) == pLS; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));

  if (ana.do_lower_level) {
    list_FRSetDef.push_back(RecoTrackSetDefinition(
        /* name  */
        "pT5_lower",
        /* pass  */ [&](unsigned int ipT5) { return lstEff.pT5_isFake().at(ipT5) > 0; },
        /* sel   */ [&](unsigned int ipT5) { return 1; },
        /* pt    */ tas::pT5_pt,
        /* eta   */ tas::pT5_eta,
        /* phi   */ tas::pT5_phi,
        /* type  */ [&]() { return static_cast<const std::vector<int>>(std::vector<int>(tas::pT5_pt().size(), 1)); }));
    list_FRSetDef.push_back(RecoTrackSetDefinition(
        /* name  */
        "T5_lower",
        /* pass  */ [&](unsigned int it5) { return lstEff.t5_isFake().at(it5) > 0; },
        /* sel   */ [&](unsigned int it5) { return 1; },
        /* pt    */ tas::t5_pt,
        /* eta   */ tas::t5_eta,
        /* phi   */ tas::t5_phi,
        /* type  */ [&]() { return static_cast<const std::vector<int>>(std::vector<int>(tas::t5_pt().size(), 1)); }));
    list_FRSetDef.push_back(RecoTrackSetDefinition(
        /* name  */
        "pT3_lower",
        /* pass  */ [&](unsigned int ipT3) { return lstEff.pT3_isFake().at(ipT3) > 0; },
        /* sel   */ [&](unsigned int ipT3) { return 1; },
        /* pt    */ tas::pT3_pt,
        /* eta   */ tas::pT3_eta,
        /* phi   */ tas::pT3_phi,
        /* type  */ [&]() { return static_cast<const std::vector<int>>(std::vector<int>(tas::pT3_pt().size(), 1)); }));
  }

  bookFakeRateSets(list_FRSetDef);

  // creating a set of duplicate rate plots
  std::vector<RecoTrackSetDefinition> list_DRSetDef;
  list_DRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "TC",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isDuplicate().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return 1; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));
  list_DRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "pT5",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isDuplicate().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return lstEff.tc_type().at(itc) == pT5; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));
  list_DRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "pT3",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isDuplicate().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return lstEff.tc_type().at(itc) == pT3; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));
  list_DRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "T5",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isDuplicate().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return lstEff.tc_type().at(itc) == T5; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));
  list_DRSetDef.push_back(
      RecoTrackSetDefinition(/* name  */
                             "pLS",
                             /* pass  */ [&](unsigned int itc) { return lstEff.tc_isDuplicate().at(itc) > 0; },
                             /* sel   */ [&](unsigned int itc) { return lstEff.tc_type().at(itc) == pLS; },
                             /* pt    */ tas::tc_pt,
                             /* eta   */ tas::tc_eta,
                             /* phi   */ tas::tc_phi,
                             /* type  */ tas::tc_type));

  if (ana.do_lower_level) {
    list_DRSetDef.push_back(RecoTrackSetDefinition(
        /* name  */
        "pT5_lower",
        /* pass  */ [&](unsigned int ipT5) { return lstEff.pT5_isDuplicate().at(ipT5) > 0; },
        /* sel   */ [&](unsigned int ipT5) { return 1; },
        /* pt    */ tas::pT5_pt,
        /* eta   */ tas::pT5_eta,
        /* phi   */ tas::pT5_phi,
        /* type  */ [&]() { return static_cast<const std::vector<int>>(std::vector<int>(tas::pT5_pt().size(), 1)); }));
    list_DRSetDef.push_back(RecoTrackSetDefinition(
        /* name  */
        "T5_lower",
        /* pass  */ [&](unsigned int it5) { return lstEff.t5_isDuplicate().at(it5) > 0; },
        /* sel   */ [&](unsigned int it5) { return 1; },
        /* pt    */ tas::t5_pt,
        /* eta   */ tas::t5_eta,
        /* phi   */ tas::t5_phi,
        /* type  */ [&]() { return static_cast<const std::vector<int>>(std::vector<int>(tas::t5_pt().size(), 1)); }));
    list_DRSetDef.push_back(RecoTrackSetDefinition(
        /* name  */
        "pT3_lower",
        /* pass  */ [&](unsigned int ipT3) { return lstEff.pT3_isDuplicate().at(ipT3) > 0; },
        /* sel   */ [&](unsigned int ipT3) { return 1; },
        /* pt    */ tas::pT3_pt,
        /* eta   */ tas::pT3_eta,
        /* phi   */ tas::pT3_phi,
        /* type  */ [&]() { return static_cast<const std::vector<int>>(std::vector<int>(tas::pT3_pt().size(), 1)); }));
  }

  bookDuplicateRateSets(list_DRSetDef);

  // Book Histograms
  ana.cutflow.bookHistograms(ana.histograms);  // if just want to book everywhere

  int nevts = 0;

  // Looping input file
  while (ana.looper.nextEvent()) {
    // If splitting jobs are requested then determine whether to process the event or not based on remainder
    if (ana.job_index != -1 and ana.nsplit_jobs != -1) {
      if (ana.looper.getNEventsProcessed() % ana.nsplit_jobs != (unsigned int)ana.job_index)
        continue;
    }

    // Reset all temporary variables necessary for histogramming
    ana.tx.clear();

    // Compute all temporary variables and pack the vector quantities that will get filled to the histograms
    fillEfficiencySets(list_effSetDef);
    fillFakeRateSets(list_FRSetDef);
    fillDuplicateRateSets(list_DRSetDef);

    // Reset all temporary variables necessary for histogramming
    ana.cutflow.fill();

    // Counting number of events processed
    nevts++;
  }

  // Write number of events processed
  ana.output_tfile->cd();
  TH1F* h_nevts = new TH1F("nevts", "nevts", 1, 0, 1);
  h_nevts->SetBinContent(1, nevts);
  h_nevts->Write();

  // Writing output file
  ana.cutflow.saveOutput();

  // The below can be sometimes crucial
  delete ana.output_tfile;

  // Done
  return 0;
}

// ---------------------------------------------------------=============================================-------------------------------------------------------------------
// ---------------------------------------------------------=============================================-------------------------------------------------------------------
// ---------------------------------------------------------=============================================-------------------------------------------------------------------
// ---------------------------------------------------------=============================================-------------------------------------------------------------------
// ---------------------------------------------------------=============================================-------------------------------------------------------------------

//__________________________________________________________________________________________________________________________________________________________________________
void bookEfficiencySets(std::vector<SimTrackSetDefinition>& effsets) {
  for (auto& effset : effsets)
    bookEfficiencySet(effset);
}

//__________________________________________________________________________________________________________________________________________________________________________
void bookEfficiencySet(SimTrackSetDefinition& effset) {
  TString category_name = TString::Format("%s_%d_%d", effset.set_name.Data(), effset.pdgid, effset.q);

  // Added by Kasia -----------------------------------------------------------------------------------------
    // Lines for etadiffs
    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_etadiffs");
    ana.histograms.addVecHistogram(category_name + "_ef_denom_etadiffs", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_etadiffs");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_etadiffs");
    ana.histograms.addVecHistogram(category_name + "_ef_numer_etadiffs", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_etadiffs");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_etadiffs");
    ana.histograms.addVecHistogram(category_name + "_ie_numer_etadiffs", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_etadiffs");} );

    // Lines for phidiffs
    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_phidiffs");
    ana.histograms.addVecHistogram(category_name + "_ef_denom_phidiffs", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_phidiffs");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_phidiffs");
    ana.histograms.addVecHistogram(category_name + "_ef_numer_phidiffs", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_phidiffs");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_phidiffs");
    ana.histograms.addVecHistogram(category_name + "_ie_numer_phidiffs", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_phidiffs");} );

    // Lines for rjet
    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_rjet");
    ana.histograms.addVecHistogram(category_name + "_ef_denom_rjet", 50 , 0  , 0.1  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_rjet");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_rjet");
    ana.histograms.addVecHistogram(category_name + "_ef_numer_rjet", 50 , 0  , 0.1  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_rjet");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_rjet");
    ana.histograms.addVecHistogram(category_name + "_ie_numer_rjet", 50 , 0  , 0.1  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_rjet");} );

    // Lines for jet_eta
    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_jet_eta");
    ana.histograms.addVecHistogram(category_name + "_ef_denom_jet_eta", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_jet_eta");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_jet_eta");
    ana.histograms.addVecHistogram(category_name + "_ef_numer_jet_eta", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_jet_eta");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_jet_eta");
    ana.histograms.addVecHistogram(category_name + "_ie_numer_jet_eta", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_jet_eta");} );

    // Lines for jet_phi
    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_jet_phi");
    ana.histograms.addVecHistogram(category_name + "_ef_denom_jet_phi", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_jet_phi");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_jet_phi");
    ana.histograms.addVecHistogram(category_name + "_ef_numer_jet_phi", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_jet_phi");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_jet_phi");
    ana.histograms.addVecHistogram(category_name + "_ie_numer_jet_phi", 180 , -4.5  , 4.5  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_jet_phi");} );

    // Lines for jet_pt
    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_jet_pt");
    ana.histograms.addVecHistogram(category_name + "_ef_denom_jet_pt", 50 , 50  , 1000  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_jet_pt");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_jet_pt");
    ana.histograms.addVecHistogram(category_name + "_ef_numer_jet_pt", 50 , 50  , 1000  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_jet_pt");} );

    ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_jet_pt");
    ana.histograms.addVecHistogram(category_name + "_ie_numer_jet_pt", 50 , 50  , 1000  , [&, category_name]() 
    {return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_jet_pt");} );

    // Moving the standard pT code up here for convenience
    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_pt");
    ana.histograms.addVecHistogram(category_name + "_ef_denom_pt", 40 , 0.9  , 2000  , [&, category_name]() { // getPtBounds(0), [&, category_name]() {
      return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_pt");
    });

    ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_pt");
    ana.histograms.addVecHistogram(category_name + "_ef_numer_pt", 40 , 0.9  , 2000  , [&, category_name]() { //
      return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_pt");
    });

    ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_pt");
    ana.histograms.addVecHistogram(category_name + "_ie_numer_pt", 40 , 0.9  , 2000  , [&, category_name]() { //
      return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_pt");
    });
    // --------------------------------------------------------------------------------------------------------

  // Denominator tracks' quantities
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_eta");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_dxy");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_vxy");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_dz");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_denom_phi");

  // Numerator tracks' quantities
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_eta");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_dxy");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_vxy");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_dz");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ef_numer_phi");

  // Inefficiencies
  ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_eta");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_dxy");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_vxy");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_dz");
  ana.tx.createBranch<std::vector<float>>(category_name + "_ie_numer_phi");


  ana.histograms.addVecHistogram(category_name + "_ef_denom_ptlow", getPtBounds(4), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_denom_ptmtv", getPtBounds(9), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_denom_ptflatbin", 180, 0., 100, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_denom_eta", 180, -4.5, 4.5, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_eta");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_denom_dxy", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_dxy");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_denom_vxy", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_vxy");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_denom_dz", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_dz");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_denom_phi", 180, -M_PI, M_PI, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_denom_phi");
  });

  ana.histograms.addVecHistogram(category_name + "_ef_numer_ptlow", getPtBounds(4), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_numer_ptmtv", getPtBounds(9), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_numer_ptflatbin", 180, 0., 100, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_numer_eta", 180, -4.5, 4.5, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_eta");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_numer_dxy", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_dxy");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_numer_vxy", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_vxy");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_numer_dz", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_dz");
  });
  ana.histograms.addVecHistogram(category_name + "_ef_numer_phi", 180, -M_PI, M_PI, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ef_numer_phi");
  });

  ana.histograms.addVecHistogram(category_name + "_ie_numer_ptlow", getPtBounds(4), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ie_numer_ptmtv", getPtBounds(9), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ie_numer_ptflatbin", 180, 0., 100, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_ie_numer_eta", 180, -4.5, 4.5, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_eta");
  });
  ana.histograms.addVecHistogram(category_name + "_ie_numer_dxy", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_dxy");
  });
  ana.histograms.addVecHistogram(category_name + "_ie_numer_vxy", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_vxy");
  });
  ana.histograms.addVecHistogram(category_name + "_ie_numer_dz", 180, -30., 30., [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_dz");
  });
  ana.histograms.addVecHistogram(category_name + "_ie_numer_phi", 180, -M_PI, M_PI, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_ie_numer_phi");
  });
}

//__________________________________________________________________________________________________________________________________________________________________________
void bookDuplicateRateSets(std::vector<RecoTrackSetDefinition>& DRsets) {
  for (auto& DRset : DRsets) {
    bookDuplicateRateSet(DRset);
  }
}

//__________________________________________________________________________________________________________________________________________________________________________
void bookDuplicateRateSet(RecoTrackSetDefinition& DRset) {
  TString category_name = DRset.set_name;

  // Denominator tracks' quantities
  ana.tx.createBranch<std::vector<float>>(category_name + "_dr_denom_pt");
  ana.tx.createBranch<std::vector<float>>(category_name + "_dr_denom_eta");
  ana.tx.createBranch<std::vector<float>>(category_name + "_dr_denom_phi");

  // Numerator tracks' quantities
  ana.tx.createBranch<std::vector<float>>(category_name + "_dr_numer_pt");
  ana.tx.createBranch<std::vector<float>>(category_name + "_dr_numer_eta");
  ana.tx.createBranch<std::vector<float>>(category_name + "_dr_numer_phi");

  // Histogram utility object that is used to define the histograms
  ana.histograms.addVecHistogram(category_name + "_dr_denom_pt", getPtBounds(0), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_denom_ptlow", getPtBounds(4), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_denom_ptmtv", getPtBounds(9), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_denom_eta", 180, -4.5, 4.5, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_denom_eta");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_denom_phi", 180, -M_PI, M_PI, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_denom_phi");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_numer_pt", getPtBounds(0), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_numer_ptlow", getPtBounds(4), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_numer_ptmtv", getPtBounds(9), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_numer_eta", 180, -4.5, 4.5, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_numer_eta");
  });
  ana.histograms.addVecHistogram(category_name + "_dr_numer_phi", 180, -M_PI, M_PI, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_dr_numer_phi");
  });
}

//__________________________________________________________________________________________________________________________________________________________________________
void bookFakeRateSets(std::vector<RecoTrackSetDefinition>& FRsets) {
  for (auto& FRset : FRsets) {
    bookFakeRateSet(FRset);
  }
}

//__________________________________________________________________________________________________________________________________________________________________________
void bookFakeRateSet(RecoTrackSetDefinition& FRset) {
  TString category_name = FRset.set_name;

  // Denominator tracks' quantities
  ana.tx.createBranch<std::vector<float>>(category_name + "_fr_denom_pt");
  ana.tx.createBranch<std::vector<float>>(category_name + "_fr_denom_eta");
  ana.tx.createBranch<std::vector<float>>(category_name + "_fr_denom_phi");

  // Numerator tracks' quantities
  ana.tx.createBranch<std::vector<float>>(category_name + "_fr_numer_pt");
  ana.tx.createBranch<std::vector<float>>(category_name + "_fr_numer_eta");
  ana.tx.createBranch<std::vector<float>>(category_name + "_fr_numer_phi");

  // Histogram utility object that is used to define the histograms
  ana.histograms.addVecHistogram(category_name + "_fr_denom_pt", getPtBounds(0), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_denom_ptlow", getPtBounds(4), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_denom_ptmtv", getPtBounds(9), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_denom_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_denom_eta", 180, -4.5, 4.5, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_denom_eta");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_numer_phi", 180, -M_PI, M_PI, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_numer_phi");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_numer_pt", getPtBounds(0), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_numer_ptlow", getPtBounds(4), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_numer_ptmtv", getPtBounds(9), [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_numer_pt");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_numer_eta", 180, -4.5, 4.5, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_numer_eta");
  });
  ana.histograms.addVecHistogram(category_name + "_fr_denom_phi", 180, -M_PI, M_PI, [&, category_name]() {
    return ana.tx.getBranchLazy<std::vector<float>>(category_name + "_fr_denom_phi");
  });
}

// ---------------------------------------------------------=============================================-------------------------------------------------------------------
// ---------------------------------------------------------=============================================-------------------------------------------------------------------
// ---------------------------------------------------------=============================================-------------------------------------------------------------------
// ---------------------------------------------------------=============================================-------------------------------------------------------------------
// ---------------------------------------------------------=============================================-------------------------------------------------------------------

//__________________________________________________________________________________________________________________________________________________________________________
void fillEfficiencySets(std::vector<SimTrackSetDefinition>& effsets) {
  for (auto& effset : effsets) {
    for (unsigned int isimtrk = 0; isimtrk < lstEff.sim_pt().size(); ++isimtrk) {
      fillEfficiencySet(isimtrk, effset);
    }
  }
}

//__________________________________________________________________________________________________________________________________________________________________________
void fillEfficiencySet(int isimtrk, SimTrackSetDefinition& effset) {
  //=========================================================
  // NOTE: The following is not applied as the LSTNtuple no longer writes this.
  // const int &bunch = lstEff.sim_bunchCrossing()[isimtrk];
  // const int &event = lstEff.sim_event()[isimtrk];
  // if (bunch != 0)
  //     return;
  // if (event != 0)
  //     return;
  //=========================================================


  // Added by Kasia
  const float& etadiffs = lstEff.sim_etadiffs()[isimtrk];
  const float& phidiffs = lstEff.sim_phidiffs()[isimtrk];
  const float& rjet = lstEff.sim_rjet()[isimtrk];
  const float& jet_eta = lstEff.sim_jet_eta()[isimtrk];
  const float& jet_phi = lstEff.sim_jet_phi()[isimtrk];
  const float& jet_pt = lstEff.sim_jet_pt()[isimtrk];

  const float& pt = lstEff.sim_pt()[isimtrk];
  const float& eta = lstEff.sim_eta()[isimtrk];
  const float& dz = lstEff.sim_pca_dz()[isimtrk];
  const float& dxy = lstEff.sim_pca_dxy()[isimtrk];
  const float& phi = lstEff.sim_phi()[isimtrk];
  const int& pdgidtrk = lstEff.sim_pdgId()[isimtrk];
  const int& q = lstEff.sim_q()[isimtrk];
  const float& vtx_x = lstEff.sim_vx()[isimtrk];
  const float& vtx_y = lstEff.sim_vy()[isimtrk];
  const float& vtx_z = lstEff.sim_vz()[isimtrk];
  const float& vtx_perp = sqrt(vtx_x * vtx_x + vtx_y * vtx_y);
  bool pass = effset.pass(isimtrk);
  bool sel = effset.sel(isimtrk);

  if (effset.pdgid != 0) {
    if (abs(pdgidtrk) != abs(effset.pdgid))
      return;
  }

  if (effset.q != 0) {
    if (q != effset.q)
      return;
  }

  if (effset.pdgid == 0 and q == 0)
    return;

  if (not sel)
    return;

  TString category_name = TString::Format("%s_%d_%d", effset.set_name.Data(), effset.pdgid, effset.q);

  // https://github.com/cms-sw/cmssw/blob/7cbdb18ec6d11d5fd17ca66c1153f0f4e869b6b0/SimTracker/Common/python/trackingParticleSelector_cfi.py
  // https://github.com/cms-sw/cmssw/blob/7cbdb18ec6d11d5fd17ca66c1153f0f4e869b6b0/SimTracker/Common/interface/TrackingParticleSelector.h#L122-L124
  const float vtx_z_thresh = 30;
  const float vtx_perp_thresh = 10;

  // jet cuts added by Kasia
  // && (sqrt(pow(vtx_x,2) + pow(vtx_y,2)) < 100)
  if(pt>0 && jet_eta<140 && jet_eta>-140 && (jet_eta>-999 && etadiffs>-999)){
    // N minus eta cut
    if (pt > ana.pt_cut and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh) {
      // vs. eta plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_eta", eta);
      if (pass)
        ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_eta", eta);
      else
        ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_eta", eta);
    }

    // N minus pt cut
    if (abs(eta) < ana.eta_cut and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh) {
      // vs. pt plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_pt", pt);
      if (pass)
        ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_pt", pt);
      else
        ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_pt", pt);
    }

    // N minus dxy cut
    if (abs(eta) < ana.eta_cut and pt > ana.pt_cut and abs(vtx_z) < vtx_z_thresh) {
      // vs. dxy plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_dxy", dxy);
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_vxy", vtx_perp);
      if (pass) {
        ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_dxy", dxy);
        ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_vxy", vtx_perp);
      } else {
        ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_dxy", dxy);
        ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_vxy", vtx_perp);
      }
    }

    // N minus dz cut
    if (abs(eta) < ana.eta_cut and pt > ana.pt_cut and abs(vtx_perp) < vtx_perp_thresh) {
      // vs. dz plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_dz", dz);
      if (pass)
        ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_dz", dz);
      else
        ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_dz", dz);
    }

    // All phase-space cuts
    if (abs(eta) < ana.eta_cut and pt > ana.pt_cut and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh) {
      // vs. Phi plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_phi", phi);
      if (pass)
        ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_phi", phi);
      else
        ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_phi", phi);
    

    // Added by Kasia ---------------------------------------------------------------------------------
      // vs. etadiffs plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_etadiffs", etadiffs);
      if (pass)
          ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_etadiffs", etadiffs);
      else
          ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_etadiffs", etadiffs);

      // vs. phidiffs plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_phidiffs", phidiffs);
      if (pass)
          ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_phidiffs", phidiffs);
      else
          ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_phidiffs", phidiffs);

      // vs. rjet plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_rjet", rjet);
      if (pass)
          ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_rjet", rjet);
      else
          ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_rjet", rjet);

      // vs. jet_eta plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_jet_eta", jet_eta);
      if (pass)
          ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_jet_eta", jet_eta);
      else
          ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_jet_eta", jet_eta);

      // vs. jet_phi plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_jet_phi", jet_phi);
      if (pass)
          ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_jet_phi", jet_phi);
      else
          ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_jet_phi", jet_phi);

      // vs. jet_pt plot
      ana.tx.pushbackToBranch<float>(category_name + "_ef_denom_jet_pt", jet_pt);
      if (pass)
          ana.tx.pushbackToBranch<float>(category_name + "_ef_numer_jet_pt", jet_pt);
      else
          ana.tx.pushbackToBranch<float>(category_name + "_ie_numer_jet_pt", jet_pt);
  //---------------------------------------------------------------------------------------------------
    }
  }
}

//__________________________________________________________________________________________________________________________________________________________________________
void fillFakeRateSets(std::vector<RecoTrackSetDefinition>& FRsets) {
  for (auto& FRset : FRsets) {
    for (unsigned int itc = 0; itc < FRset.pt().size(); ++itc) {
      fillFakeRateSet(itc, FRset);
    }
  }
}

//__________________________________________________________________________________________________________________________________________________________________________
void fillFakeRateSet(int itc, RecoTrackSetDefinition& FRset) {
  float pt = FRset.pt().at(itc);
  float eta = FRset.eta().at(itc);
  float phi = FRset.phi().at(itc);
  TString category_name = FRset.set_name;
  bool pass = FRset.pass(itc);
  bool sel = FRset.sel(itc);

  if (not sel)
    return;

  if (pt > ana.pt_cut) {
    ana.tx.pushbackToBranch<float>(category_name + "_fr_denom_eta", eta);
    if (pass)
      ana.tx.pushbackToBranch<float>(category_name + "_fr_numer_eta", eta);
  }
  if (abs(eta) < ana.eta_cut) {
    ana.tx.pushbackToBranch<float>(category_name + "_fr_denom_pt", pt);
    if (pass)
      ana.tx.pushbackToBranch<float>(category_name + "_fr_numer_pt", pt);
  }
  if (abs(eta) < ana.eta_cut and pt > ana.pt_cut) {
    ana.tx.pushbackToBranch<float>(category_name + "_fr_denom_phi", phi);
    if (pass)
      ana.tx.pushbackToBranch<float>(category_name + "_fr_numer_phi", phi);
  }
}

//__________________________________________________________________________________________________________________________________________________________________________
void fillDuplicateRateSets(std::vector<RecoTrackSetDefinition>& DRsets) {
  for (auto& DRset : DRsets) {
    for (unsigned int itc = 0; itc < DRset.pt().size(); ++itc) {
      fillDuplicateRateSet(itc, DRset);
    }
  }
}

//__________________________________________________________________________________________________________________________________________________________________________
void fillDuplicateRateSet(int itc, RecoTrackSetDefinition& DRset) {
  float pt = DRset.pt().at(itc);
  float eta = DRset.eta().at(itc);
  float phi = DRset.phi().at(itc);
  TString category_name = DRset.set_name;
  bool pass = DRset.pass(itc);
  bool sel = DRset.sel(itc);

  if (not sel)
    return;

  if (pt > ana.pt_cut) {
    ana.tx.pushbackToBranch<float>(category_name + "_dr_denom_eta", eta);
    if (pass)
      ana.tx.pushbackToBranch<float>(category_name + "_dr_numer_eta", eta);
  }
  if (abs(eta) < ana.eta_cut) {
    ana.tx.pushbackToBranch<float>(category_name + "_dr_denom_pt", pt);
    if (pass)
      ana.tx.pushbackToBranch<float>(category_name + "_dr_numer_pt", pt);
  }
  if (abs(eta) < ana.eta_cut and pt > ana.pt_cut) {
    ana.tx.pushbackToBranch<float>(category_name + "_dr_denom_phi", phi);
    if (pass)
      ana.tx.pushbackToBranch<float>(category_name + "_dr_numer_phi", phi);
  }
}
