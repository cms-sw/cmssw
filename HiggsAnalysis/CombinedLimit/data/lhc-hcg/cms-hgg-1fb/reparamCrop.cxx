/*
 * Usage: 
 *     root.exe -b -l -q hgg140-shapes-combined-Unbinned.root HGG_SM_WITHSYS_140_4cat.root reparamCrop.cxx
 * Then copy-paste the lines from the output into the datacard 
 */
void setConstPars(RooAbsPdf *pdf, RooAbsData *data, bool setConst) {
    RooArgSet *params = pdf->getParameters(*data);
    TIterator *iter = params->createIterator();
    for (TObject *a = iter->Next(); a != 0; a = iter->Next()) {
        if (!a->InheritsFrom("RooRealVar")) continue; // dynamic_cast in CINT doesn't work well
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv) { rrv->setConstant(setConst); }
    }
    delete iter;
    delete params;
}

void reparamCrop(double low=130, double high=150, double veryLow=100, double veryHigh=180, int ncat=4) {
    using namespace RooFit;
    if (gFile == 0) return;
    TFile *fSerguei = _file0;
    TFile *fMarco   = _file1;
    TCanvas *c1 = new TCanvas("c1","c1");
    RooRandom::randomGenerator()->SetSeed(114);
    // Now we do the re-paratemtrization trick.

    // (1) import everything
    RooWorkspace *wAll = new RooWorkspace("w_all","w_all");
    RooWorkspace *ws[4];
    wAll->factory(TString::Format("CMS_mgg[%g,%g]", veryLow, veryHigh));
    RooRealVar *mgg = wAll->var("CMS_mgg");
    mgg->setRange("sidebandLo", veryLow, low);
    mgg->setRange("signal",     low,     high);
    mgg->setRange("sidebandHi", high,    veryHigh);
    for (int c = 0; c < ncat; ++c) {
        TString postfix = TString::Format("_cat%d",c);
        ws[c] = (RooWorkspace*) fSerguei->Get("w"+postfix);
        TH1* m_sig = fMarco->Get("sig"+postfix);
        TH1* m_bkg = fMarco->Get("bkg"+postfix);
        TH1* m_dat = fMarco->Get("datmc"+postfix);
        RooArgList hvars(*wAll->var("CMS_mgg"));
        RooDataHist *m_hsig = new RooDataHist("sighist"+postfix, "", hvars, m_sig);
        RooDataHist *m_hbkg = new RooDataHist("bkghist"+postfix, "", hvars, m_bkg);
        RooDataHist *m_hdat = new RooDataHist("dathist"+postfix, "", hvars, m_dat);
        wAll->factory(TString::Format("nS_cat%d[%g]", c, m_sig->Integral()));
        wAll->factory(TString::Format("nBall_cat%d[%g]", c, m_bkg->Integral()));
        wAll->import(*ws[c]->pdf("MggBkg"+postfix), RenameAllVariablesExcept("CMS","mass"), RenameAllNodes("CMS"), RenameVariable("mass","CMS_mgg"));
        wAll->import(*ws[c]->pdf("MggSig"+postfix), RenameAllVariablesExcept("CMS","mass"), RenameAllNodes("CMS"), RenameVariable("mass","CMS_mgg"));
        // fit parametric forms for signal to input histograms
        setConstPars(wAll->pdf("MggSig"+postfix+"_CMS"), m_hsig, false); // unfloat all parameters
        wAll->pdf("MggSig"+postfix+"_CMS")->fitTo(*m_hsig, Range("signal"));
        setConstPars(wAll->pdf("MggSig"+postfix+"_CMS"), m_hsig, true); // then fix them again
        RooPlot *sframe = wAll->var("CMS_mgg")->frame(Range("signal"));
        m_hsig->plotOn(sframe);
        wAll->pdf("MggSig"+postfix+"_CMS")->plotOn(sframe, LineColor(kRed),  Range("signal"));
        sframe->Draw(); c1->Print(TString::Format("fit_sig_cat%d.png",c));
        // now fit background on data, get resulting uncertainties on nB, slope
        wAll->factory(TString::Format("nB_cat%d[0,%g]", c, 2*m_bkg->Integral()));
        setConstPars(wAll->pdf("MggBkg"+postfix+"_CMS"), m_hbkg, false); // unfloat all parameters
        RooAddPdf bgModel("bgModel","bgModel", RooArgList(*wAll->pdf("MggBkg"+postfix+"_CMS")), RooArgList(*wAll->var("nB"+postfix)));
        bgModel.fitTo(*m_hdat, Range("sidebandLo,sidebandHi"), SumCoefRange("signal"));
        RooPlot *frame = wAll->var("CMS_mgg")->frame();
        m_hdat->plotOn(frame);
        bgModel.plotOn(frame, LineColor(kBlue), Range("sidebandLo,sidebandHi"));
        bgModel.plotOn(frame, LineColor(kRed),  Range("signal"));
        frame->Draw(); c1->Print(TString::Format("fit_cat%d.png",c));
    }
    //wAll->Print("V");

    // (2) create new masses
    wAll->factory("CMS_hgg_sig_m0_absShiftEBEB[0]");
    wAll->factory("CMS_hgg_sig_m0_absShiftEEEX[0]");
    wAll->factory("CMS_hgg_sig_m0_absShiftBadR9[0]");
    wAll->factory("sum::CMS_hgg_sig_m0_cat0(mgg_sig_m0_cat0_CMS, CMS_hgg_sig_m0_absShiftEBEB)");
    wAll->factory("sum::CMS_hgg_sig_m0_cat1(mgg_sig_m0_cat1_CMS, CMS_hgg_sig_m0_absShiftEBEB, CMS_hgg_sig_m0_absShiftBadR9)");
    wAll->factory("sum::CMS_hgg_sig_m0_cat2(mgg_sig_m0_cat2_CMS, CMS_hgg_sig_m0_absShiftEEEX)");
    wAll->factory("sum::CMS_hgg_sig_m0_cat3(mgg_sig_m0_cat3_CMS, CMS_hgg_sig_m0_absShiftEEEX, CMS_hgg_sig_m0_absShiftBadR9)");
    
    // (2) create new sigmas
    wAll->factory("CMS_hgg_sig_sigmaScaleEBEB[1]");
    wAll->factory("CMS_hgg_sig_sigmaScaleEEEX[1]");
    wAll->factory("CMS_hgg_sig_sigmaScaleBadR9[1]");
    wAll->factory("prod::CMS_hgg_sig_sigma_cat0(mgg_sig_sigma_cat0_CMS, CMS_hgg_sig_sigmaScaleEBEB)");
    wAll->factory("prod::CMS_hgg_sig_sigma_cat1(mgg_sig_sigma_cat1_CMS, CMS_hgg_sig_sigmaScaleEBEB, CMS_hgg_sig_sigmaScaleBadR9)");
    wAll->factory("prod::CMS_hgg_sig_sigma_cat2(mgg_sig_sigma_cat2_CMS, CMS_hgg_sig_sigmaScaleEEEX)");
    wAll->factory("prod::CMS_hgg_sig_sigma_cat3(mgg_sig_sigma_cat3_CMS, CMS_hgg_sig_sigmaScaleEEEX, CMS_hgg_sig_sigmaScaleBadR9)");
    wAll->factory("prod::CMS_hgg_sig_gsigma_cat0(mgg_sig_gsigma_cat0_CMS, CMS_hgg_sig_sigmaScaleEBEB)");
    wAll->factory("prod::CMS_hgg_sig_gsigma_cat1(mgg_sig_gsigma_cat1_CMS, CMS_hgg_sig_sigmaScaleEBEB, CMS_hgg_sig_sigmaScaleBadR9)");
    wAll->factory("prod::CMS_hgg_sig_gsigma_cat2(mgg_sig_gsigma_cat2_CMS, CMS_hgg_sig_sigmaScaleEEEX)");
    wAll->factory("prod::CMS_hgg_sig_gsigma_cat3(mgg_sig_gsigma_cat3_CMS, CMS_hgg_sig_sigmaScaleEEEX, CMS_hgg_sig_sigmaScaleBadR9)");
    
    for (int c = 0; c < ncat; ++c) {
        wAll->factory(TString::Format("CMS_hgg_bkg_slope_cat%d[0]", c));
        wAll->var(TString::Format("CMS_hgg_bkg_slope_cat%d",c))->setVal(wAll->var(TString::Format("mgg_bkg_slope_cat%d_CMS",c))->getVal());
        wAll->var(TString::Format("CMS_hgg_bkg_slope_cat%d",c))->setError(wAll->var(TString::Format("mgg_bkg_slope_cat%d_CMS",c))->getError());
    }

    // (3) do reparametrization of signal
    for (int c = 0; c < ncat; ++c) {
        wAll->factory(
            TString::Format("EDIT::CMS_hgg_sig_cat%d(MggSig_cat%d_CMS,",c,c) +
            TString::Format(" mgg_sig_m0_cat%d_CMS=CMS_hgg_sig_m0_cat%d, ", c,c) +
            TString::Format(" mgg_sig_sigma_cat%d_CMS=CMS_hgg_sig_sigma_cat%d, ", c,c) +
            TString::Format(" mgg_sig_gsigma_cat%d_CMS=CMS_hgg_sig_gsigma_cat%d)", c,c)
        );
        wAll->factory(
            TString::Format("EDIT::CMS_hgg_bkg_cat%d(MggBkg_cat%d_CMS,",c,c) +
            TString::Format(" mgg_bkg_slope_cat%d_CMS=CMS_hgg_bkg_slope_cat%d)", c,c)
        );

    }
    wAll->Print("V");

    // now crop ranges
    mgg->setMin(low); mgg->setMax(high); 
    RooArgSet obs(*mgg);
    wAll->factory("sFrac[0.1,0,1]");
    for (int c = 0; c < ncat; ++c) {
        TString postfix = TString::Format("_cat%d",c);
        RooAbsData *data_cat = wAll->pdf("CMS_hgg_bkg"+postfix)->generate(obs, wAll->var("nB"+postfix)->getVal());
        data_cat->SetName("data_obs"+postfix);
        wAll->import(*data_cat);
        RooAddPdf bgModel("bgModel","bgModel", *wAll->pdf("CMS_hgg_sig"+postfix), *wAll->pdf("CMS_hgg_bkg"+postfix), *wAll->var("sFrac"));
        bgModel.fitTo(*data_cat);
        RooPlot *frame = wAll->var("CMS_mgg")->frame();
        data_cat->plotOn(frame);
        bgModel.plotOn(frame,LineColor(kRed));
        bgModel.plotOn(frame,LineColor(kRed),  Components("CMS_hgg_sig"+postfix));
        bgModel.plotOn(frame,LineColor(kBlue), Components("CMS_hgg_bkg"+postfix));
        frame->Draw(); c1->Print(TString::Format("fit_data_cat%d.png",c));
    }

    TString name = _file0->GetName();
    name.ReplaceAll(".root",".crop.root");
    wAll->writeToFile(name);

 
    std::cout << std::endl;
    std::cout << "==================================================================================================" << std::endl;
    std::cout << "=== LINES TO PUT IN THE DATACARD =================================================================" << std::endl;
    std::cout << "==================================================================================================" << std::endl;
    std::cout << "observation   ";
    for (int c = 0; c < ncat; ++c) {
        std::cout << wAll->data(TString::Format("data_obs_cat%d",c))->sumEntries() << "    ";
    }
    std::cout << std::endl;

    std::cout << "rate                     ";
    for (int c = 0; c < ncat; ++c) {
        TString postfix = TString::Format("_cat%d",c);
        printf("%4.2f      %5.1f      ", 
            wAll->var("nS"+postfix)->getVal(),
            wAll->var("nB"+postfix)->getVal());
    }
    std::cout << std::endl;

    for (int c = 0; c < ncat; ++c) {
        TString postfix = TString::Format("_cat%d",c);
        printf("CMS_hgg_bg_cat%d lnN       ",c);
        for (int k = 0; k < ncat; ++k) {
            if (k == c) {
                printf("-         %5.3f      ", 1.0 + wAll->var("nB"+postfix)->getError()/wAll->var("nB"+postfix)->getVal());
            } else {
                printf("-          -         ");
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int c = 0; c < ncat; ++c) {
        printf("CMS_hgg_bkg_slope_cat%d  param  %.4f  %.4f   # Mean and absolute uncertainty on background slope\n", c,
                wAll->var(TString::Format("CMS_hgg_bkg_slope_cat%d",c))->getVal(),
                wAll->var(TString::Format("CMS_hgg_bkg_slope_cat%d",c))->getError());
    }

}
