void make_simple_shapes(int nS=10, int nB=100, int seed=42) {
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
    w->pdf("model_s")->plotOn(frame);
    frame->Draw();

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
    // background model
    TH1 *background_nominal = w->pdf("background")->createHistogram("x"); 
    background_nominal->SetName("background"); background_nominal->Scale(nB/background_nominal->Integral());
    w->var("alpha")->setVal(-0.25);
    TH1 *background_alphaUp = w->pdf("background")->createHistogram("x");  
    background_alphaUp->SetName("background_alphaUp"); background_alphaUp->Scale(nB*1.15/background_alphaUp->Integral());
    w->var("alpha")->setVal(-0.35);
    TH1 *background_alphaDown = w->pdf("background")->createHistogram("x");  
    background_alphaDown->SetName("background_alphaDown"); background_alphaDown->Scale(nB*0.90/background_alphaDown->Integral());
    w->var("alpha")->setVal(-0.3);
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
}
