void make_simple_shapes(int nS=10, int nB=100, int seed=37) {
    using namespace RooFit;
    RooRandom::randomGenerator()->SetSeed(seed); 
    TCanvas *c1 = new TCanvas("c1","c1");

    RooWorkspace *w = new RooWorkspace();
    w->factory("x[0,10]");
    w->var("x")->setBins(10);
    w->factory(TString::Format("nS[%d]",nS));
    w->factory(TString::Format("nB[%d]",nB));
    w->factory("Exponential::background(x,alpha[-0.3])");
    w->factory("Gaussian::signal(x,6,sigma[1])");
    w->factory("SUM::model_s(nB*background, nS*signal)");
    w->factory("SUM::model_b(nB*background)");

    RooArgSet obs(*w->var("x"));
    RooDataSet *data_s = w->pdf("model_s")->generate(obs,Extended());
    RooDataSet *data_b = w->pdf("model_b")->generate(obs,Extended());
    
    RooPlot *frame = w->var("x")->frame();
    data_s->plotOn(frame);
    w->pdf("model_s")->plotOn(frame, LineColor(kRed));
    w->pdf("model_s")->plotOn(frame, Components("background"));
    frame->Draw();
    c1->Print("data_s.png");
    frame = w->var("x")->frame();
    data_b->plotOn(frame);
    w->pdf("model_b")->plotOn(frame);
    frame->Draw();
    c1->Print("data_b.png");


    RooDataHist *bdata_b = new RooDataHist("data_obs", "", obs, *data_b);
    RooDataHist *bdata_s = new RooDataHist("data_sig", "", obs, *data_s);

    // ------------ Make histograms ---------------------------
    TFile *allHistFile = new TFile("simple-shapes-TH1.root", "RECREATE");
    // Signal model
    TH1 *signal_nominal = w->pdf("signal")->createHistogram("x"); 
    signal_nominal->SetName("signal"); signal_nominal->Scale(nS/signal_nominal->Integral());
    w->var("sigma")->setVal(1.6);
    TH1 *signal_sigmaUp = w->pdf("signal")->createHistogram("x");  
    signal_sigmaUp->SetName("signal_sigmaUp"); signal_sigmaUp->Scale(nS/signal_sigmaUp->Integral());
    w->var("sigma")->setVal(0.7);
    TH1 *signal_sigmaDown = w->pdf("signal")->createHistogram("x");  
    signal_sigmaDown->SetName("signal_sigmaDown"); signal_sigmaDown->Scale(nS/signal_sigmaDown->Integral());
    w->var("sigma")->setVal(1.0);
    c1->Clear();
    signal_sigmaDown->Draw("H"); signal_sigmaDown->SetLineColor(kBlue); signal_sigmaDown->SetLineWidth(2);
    signal_sigmaUp->Draw("H SAME"); signal_sigmaUp->SetLineColor(kRed); signal_sigmaUp->SetLineWidth(2);
    signal_nominal->Draw("H SAME"); signal_nominal->SetLineColor(kBlack); signal_nominal->SetLineWidth(3);
    c1->Print("signal_model_binned.png");
    
    frame = w->var("x")->frame();
    w->pdf("signal")->plotOn(frame, LineColor(kBlack), LineWidth(3));
    w->var("sigma")->setVal(1.6);
    w->pdf("signal")->plotOn(frame, LineColor(kBlue), LineWidth(2));
    w->var("sigma")->setVal(0.7);
    w->pdf("signal")->plotOn(frame, LineColor(kRed), LineWidth(2));
    frame->Draw();
    c1->Print("signal_model_unbinned.png");
    // background model
    frame = w->var("x")->frame();
    TH1 *background_nominal = w->pdf("background")->createHistogram("x"); 
    background_nominal->SetName("background"); background_nominal->Scale(nB/background_nominal->Integral());
    w->var("alpha")->setVal(-0.2);
    TH1 *background_alphaUp = w->pdf("background")->createHistogram("x");  
    background_alphaUp->SetName("background_alphaUp"); background_alphaUp->Scale(nB*1.15/background_alphaUp->Integral());
    w->pdf("background")->plotOn(frame, LineColor(kRed), LineWidth(2), Normalization(1.15));
    w->var("alpha")->setVal(-0.4);
    TH1 *background_alphaDown = w->pdf("background")->createHistogram("x");  
    background_alphaDown->SetName("background_alphaDown"); background_alphaDown->Scale(nB*0.90/background_alphaDown->Integral());
    w->pdf("background")->plotOn(frame, LineColor(kBlue), LineWidth(2), Normalization(0.90));
    w->var("alpha")->setVal(-0.3);
    w->pdf("background")->plotOn(frame, LineColor(kBlack), LineWidth(3), Normalization(1.0));
    frame->Draw(); c1->Print("background_model_unbinned.png");
    background_alphaDown->Draw("H"); background_alphaDown->SetLineColor(kBlue); background_alphaDown->SetLineWidth(2);
    background_alphaUp->Draw("H SAME"); background_alphaUp->SetLineColor(kRed); background_alphaUp->SetLineWidth(2);
    background_nominal->Draw("H SAME"); background_nominal->SetLineColor(kBlack); background_nominal->SetLineWidth(3);
    c1->Print("background_model_binned.png");
    // data
    TH1 *hdata_b = bdata_b->createHistogram("x"); hdata_b->SetName("data_obs");
    TH1 *hdata_s = bdata_s->createHistogram("x"); hdata_s->SetName("data_sig");
    // write to file
    allHistFile->WriteTObject(signal_nominal); allHistFile->WriteTObject(signal_sigmaUp); allHistFile->WriteTObject(signal_sigmaDown);
    allHistFile->WriteTObject(background_nominal); allHistFile->WriteTObject(background_alphaUp); allHistFile->WriteTObject(background_alphaDown);
    allHistFile->WriteTObject(hdata_b);
    allHistFile->WriteTObject(hdata_s);

    // ------------ Make RooFit histograms ----------------------------------
    RooWorkspace *wB = new RooWorkspace("w","w");
    RooArgList hobs(*w->var("x"));
    wB->import(*bdata_b);
    wB->import(*bdata_s);
    RooDataHist *hsignal = new RooDataHist("hsignal","",hobs,signal_nominal);
    RooDataHist *hsignal_sigmaUp = new RooDataHist("hsignal_sigmaUp","",hobs,signal_sigmaUp);
    RooDataHist *hsignal_sigmaDown = new RooDataHist("hsignal_sigmaDown","",hobs,signal_sigmaDown);
    RooDataHist *hbackground = new RooDataHist("hbackground","",hobs,background_nominal);
    RooDataHist *hbackground_alphaUp = new RooDataHist("hbackground_alphaUp","",hobs,background_alphaUp);
    RooDataHist *hbackground_alphaDown = new RooDataHist("hbackground_alphaDown","",hobs,background_alphaDown);
    wB->import(*(new RooHistPdf("signal","",obs,*hsignal)));
    wB->import(*(new RooHistPdf("signal_sigmaUp","",obs,*hsignal_sigmaUp)));
    wB->import(*(new RooHistPdf("signal_sigmaDown","",obs,*hsignal_sigmaDown)));
    wB->import(*(new RooHistPdf("background","",obs,*hbackground)));
    wB->import(*(new RooHistPdf("background_alphaUp","",obs,*hbackground_alphaUp)));
    wB->import(*(new RooHistPdf("background_alphaDown","",obs,*hbackground_alphaDown)));
    wB->writeToFile("simple-shapes-RooDataHist.root");

    RooWorkspace *wBP = new RooWorkspace("w","w");
    wBP->import(*bdata_b);
    wBP->import(*bdata_s);
    wBP->import(*w->pdf("signal"));
    wBP->import(*w->pdf("background"));
    wBP->writeToFile("simple-shapes-BinnedParam.root");

    RooWorkspace *wUP = new RooWorkspace("w","w");
    wUP->var("x[0,10]");
    wUP->import(*data_b, Rename("data_obs"));
    wUP->import(*data_s, Rename("data_sig"));
    wUP->import(*w->pdf("signal"));
    wUP->import(*w->pdf("background"));
    wUP->writeToFile("simple-shapes-UnbinnedParam.root");

    // now we make a version in which the alpha is function of a unit gaussian, 
    // so that we can do normalization and parametric morphing together
    RooWorkspace *wUPN = new RooWorkspace("w","w");
    wUPN->var("x[0,10]");
    wUPN->import(*data_b, Rename("data_obs"));
    wUPN->import(*data_s, Rename("data_sig"));
    wUPN->import(*w->pdf("signal"));
    RooAbsPdf *bgpdf = (RooAbsPdf *) *w->pdf("background")->clone("background_norm");
    wUPN->import(*bgpdf);
    wUPN->factory("sum::param_alpha(-0.3,prod(alphaNorm[0],0.1))"); // param_alpha = -0.3 + 0.1*alphaNorm, so Gauss(-0.3,1)
    wUPN->factory("EDIT::background(background_norm, alpha=param_alpha)");
    wUPN->Print("V");
    wUPN->writeToFile("simple-shapes-UnbinnedParamNorm.root");
    
}
