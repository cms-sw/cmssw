void reparam(int ncat=4, int crop=0) {
    if (gFile == 0) return;
    TFile *fIn = gFile;

    // Now we do the re-paratemtrization trick.

    // (1) import everything
    RooWorkspace *wAll = new RooWorkspace("w_all","w_all");
    RooWorkspace *ws[4];
    for (int c = 0; c < ncat; ++c) {
        ws[c] = (RooWorkspace*) fIn->Get(TString::Format("w_cat%d",c));
        wAll->import(*ws[c]->pdf(TString::Format("MggBkg_cat%d",c)));
        wAll->import(*ws[c]->pdf(TString::Format("MggSig_cat%d",c)));
        wAll->import(*ws[c]->data(TString::Format("data_obs_cat%d",c)));
    }
    wAll->Print("V");

    // (2) create new masses
    wAll->factory("CMS_hgg_sig_m0_absShiftEBEB[0]");
    wAll->factory("CMS_hgg_sig_m0_absShiftEEEX[0]");
    wAll->factory("CMS_hgg_sig_m0_absShiftBadR9[0]");
    wAll->factory("sum::CMS_hgg_sig_m0_cat0(mgg_sig_m0_cat0, CMS_hgg_sig_m0_absShiftEBEB)");
    wAll->factory("sum::CMS_hgg_sig_m0_cat1(mgg_sig_m0_cat1, CMS_hgg_sig_m0_absShiftEBEB, CMS_hgg_sig_m0_absShiftBadR9)");
    wAll->factory("sum::CMS_hgg_sig_m0_cat2(mgg_sig_m0_cat2, CMS_hgg_sig_m0_absShiftEEEX)");
    wAll->factory("sum::CMS_hgg_sig_m0_cat3(mgg_sig_m0_cat3, CMS_hgg_sig_m0_absShiftEEEX, CMS_hgg_sig_m0_absShiftBadR9)");
    
    // (2) create new sigmas
    wAll->factory("CMS_hgg_sig_sigmaScaleEBEB[1]");
    wAll->factory("CMS_hgg_sig_sigmaScaleEEEX[1]");
    wAll->factory("CMS_hgg_sig_sigmaScaleBadR9[1]");
    wAll->factory("prod::CMS_hgg_sig_sigma_cat0(mgg_sig_sigma_cat0, CMS_hgg_sig_sigmaScaleEBEB)");
    wAll->factory("prod::CMS_hgg_sig_sigma_cat1(mgg_sig_sigma_cat1, CMS_hgg_sig_sigmaScaleEBEB, CMS_hgg_sig_sigmaScaleBadR9)");
    wAll->factory("prod::CMS_hgg_sig_sigma_cat2(mgg_sig_sigma_cat2, CMS_hgg_sig_sigmaScaleEEEX)");
    wAll->factory("prod::CMS_hgg_sig_sigma_cat3(mgg_sig_sigma_cat3, CMS_hgg_sig_sigmaScaleEEEX, CMS_hgg_sig_sigmaScaleBadR9)");
    wAll->factory("prod::CMS_hgg_sig_gsigma_cat0(mgg_sig_gsigma_cat0, CMS_hgg_sig_sigmaScaleEBEB)");
    wAll->factory("prod::CMS_hgg_sig_gsigma_cat1(mgg_sig_gsigma_cat1, CMS_hgg_sig_sigmaScaleEBEB, CMS_hgg_sig_sigmaScaleBadR9)");
    wAll->factory("prod::CMS_hgg_sig_gsigma_cat2(mgg_sig_gsigma_cat2, CMS_hgg_sig_sigmaScaleEEEX)");
    wAll->factory("prod::CMS_hgg_sig_gsigma_cat3(mgg_sig_gsigma_cat3, CMS_hgg_sig_sigmaScaleEEEX, CMS_hgg_sig_sigmaScaleBadR9)");
    
    for (int c = 0; c < ncat; ++c) {
        wAll->factory(TString::Format("CMS_hgg_bkg_slope_cat%d[%g]", c, wAll->var(TString::Format("mgg_bkg_slope_cat%d",c))->getVal()));
    }

    // (3) do reparametrization of signal
    for (int c = 0; c < ncat; ++c) {
        wAll->factory(
            TString::Format("EDIT::CMS_hgg_sig_cat%d(MggSig_cat%d,",c,c) +
            TString::Format(" mgg_sig_m0_cat%d=CMS_hgg_sig_m0_cat%d, ", c,c) +
            TString::Format(" mgg_sig_sigma_cat%d=CMS_hgg_sig_sigma_cat%d, ", c,c) +
            TString::Format(" mgg_sig_gsigma_cat%d=CMS_hgg_sig_gsigma_cat%d)", c,c)
        );
        wAll->factory(
            TString::Format("EDIT::CMS_hgg_bkg_cat%d(MggBkg_cat%d,",c,c) +
            TString::Format(" mgg_bkg_slope_cat%d=CMS_hgg_bkg_slope_cat%d)", c,c)
        );

    }

    TString name = fIn->GetName();
    name.ReplaceAll(".root",".reparam.root");
    wAll->writeToFile(name);

  
    std::cout << "observation ";
    for (int c = 0; c < ncat; ++c) {
        std::cout << "  " << wAll->data(TString::Format("data_obs_cat%d",c))->sumEntries();
    }
    std::cout << std::endl;

    for (int c = 0; c < ncat; ++c) {
        printf("CMS_hgg_bkg_cat%d  param  %.4f  %.3f   # Mean and absolute uncertainty on background slope\n", 
                c, wAll->var(TString::Format("CMS_hgg_bkg_slope_cat%d",c))->getVal(), 0.003);
    }

}
