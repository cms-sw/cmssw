
void cccPlot(const char *poi = "r", double rMax=4, const char *filename="ccc.pdf") {
    TCanvas *c1 = new TCanvas("c1");
    c1->SetLeftMargin(0.4);
    c1->SetGridx(1);

    if (gFile == 0) { std::cout << "No input file open" << std::endl; return; }
    RooFitResult *fit_nominal   = (RooFitResult *) gFile->Get("fit_nominal");
    RooFitResult *fit_alternate = (RooFitResult *) gFile->Get("fit_alternate");
    if (fit_nominal == 0 || fit_alternate == 0) { std::cout << "Input file " << gFile->GetName() << " does not contain fit_nominal or fit_alternate" << std::endl; return; }
    RooRealVar *rFit = (RooRealVar *) fit_nominal->floatParsFinal().find(poi);
    if (rFit == 0)  { std::cout << "Nominal fit does not contain parameter " << poi << std::endl; return; }

    TString prefix =  TString::Format("_ChannelCompatibilityCheck_%s_", poi);


    int nChann = 0;
    TIterator *iter = fit_alternate->floatParsFinal().createIterator();
    for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
        if (TString(a->GetName()).Index(prefix) == 0) nChann++;
    }
    TH2F frame("frame",";best fit #sigma/#sigma_{SM};",1,rFit->getMin(),TMath::Min(rFit->getMax(),rMax),nChann,0,nChann);

    iter->Reset(); int iChann = 0; TGraphAsymmErrors points(nChann);
    for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
        if (TString(a->GetName()).Index(prefix) == 0) {
            RooRealVar *ri = (RooRealVar *) a;
            TString channel = a->GetName(); channel.ReplaceAll(prefix,"");
            points.SetPoint(iChann,       ri->getVal(), iChann+0.5);
            points.SetPointError(iChann, -ri->getAsymErrorLo(), ri->getAsymErrorHi(), 0, 0);
            iChann++;
            frame.GetYaxis()->SetBinLabel(iChann, channel);
        }
    }
    points.SetLineColor(kRed);
    points.SetLineWidth(3);
    points.SetMarkerStyle(21);
    //frame.GetXaxis()->SetNdivisions(505);
    frame.GetXaxis()->SetTitleSize(0.05);
    frame.GetXaxis()->SetLabelSize(0.04);
    frame.GetYaxis()->SetLabelSize(0.06);
    frame.Draw(); gStyle->SetOptStat(0);
    TBox globalFitBand(rFit->getVal()+rFit->getAsymErrorLo(), 0, rFit->getVal()+rFit->getAsymErrorHi(), nChann);
    globalFitBand.SetFillStyle(3013);
    globalFitBand.SetFillColor(65);
    globalFitBand.SetLineStyle(0);
    globalFitBand.DrawClone();
    TLine globalFitLine(rFit->getVal(), 0, rFit->getVal(), nChann);
    globalFitLine.SetLineWidth(4);
    globalFitLine.SetLineColor(214);
    globalFitLine.DrawClone();
    points.Draw("P SAME");
    c1->Print(filename);
}
