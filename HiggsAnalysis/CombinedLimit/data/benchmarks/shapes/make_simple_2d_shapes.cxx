void make_simple_2d_shapes(int nS=20, int nB=400, int seed=37) {
    using namespace RooFit;
    RooRandom::randomGenerator()->SetSeed(seed); 
    TCanvas *c1 = new TCanvas("c1","c1");

    RooWorkspace *w = new RooWorkspace();
    w->factory("x[0,10]");
    w->factory("y[0,10]");
    w->var("x")->setBins(10);
    w->var("y")->setBins( 5);
    w->factory(TString::Format("nS[%d]",nS));
    w->factory(TString::Format("nB[%d]",nB));
    w->factory("PROD::background(Exponential::background_x(x,alpha[-0.3]), Gaussian::background_y(y,5,6))");
    w->factory("PROD::signal(Gaussian::signal_x(x,6,sigma[1]), Exponential::signal_y(y,-0.5))");
    w->factory("SUM::model_s(nB*background, nS*signal)");
    w->factory("SUM::model_b(nB*background)");

    RooArgSet obs(*w->var("x"), *w->var("y"));
    RooDataSet *data_s = w->pdf("model_s")->generate(obs,Extended());
    RooDataSet *data_b = w->pdf("model_b")->generate(obs,Extended());
    
    RooPlot *frameX = w->var("x")->frame();
    data_s->plotOn(frameX);
    w->pdf("model_s")->plotOn(frameX, LineColor(kRed));
    w->pdf("model_s")->plotOn(frameX, Components("background"));
    frameX->Draw();
    c1->Print("data_s_X.png");
    RooPlot *frameY = w->var("y")->frame();
    data_s->plotOn(frameY);
    w->pdf("model_s")->plotOn(frameY, LineColor(kRed));
    w->pdf("model_s")->plotOn(frameY, Components("background"));
    frameY->Draw();
    c1->Print("data_s_Y.png");

    frameX = w->var("x")->frame();
    data_b->plotOn(frameX);
    w->pdf("model_s")->plotOn(frameX, LineColor(kRed));
    w->pdf("model_s")->plotOn(frameX, Components("background"));
    frameX->Draw();
    c1->Print("data_b_X.png");
    frameY = w->var("y")->frame();
    data_b->plotOn(frameY);
    w->pdf("model_s")->plotOn(frameY, LineColor(kRed));
    w->pdf("model_s")->plotOn(frameY, Components("background"));
    frameY->Draw();
    c1->Print("data_b_Y.png");


    RooWorkspace *wUP = new RooWorkspace("w","w");
    wUP->import(*data_b, Rename("data_obs"));
    wUP->import(*data_s, Rename("data_sig"));
    wUP->import(*w->pdf("signal"));
    wUP->import(*w->pdf("background"));
    wUP->writeToFile("simple2d-shapes-param.root");
    FILE *fUP = fopen("simple2d-shapes-param.txt", "w");
    fprintf(fUP, "imax 1 channels\n");
    fprintf(fUP, "jmax 1 backgrounds\n");
    fprintf(fUP, "kmax * systematics\n");
    fprintf(fUP, "-------------------------------\n");
    fprintf(fUP, "shapes * * simple2d-shapes-param.root w:$PROCESS\n");
    fprintf(fUP, "-------------------------------\n");
    fprintf(fUP, "bin           A\n");
    fprintf(fUP, "observation  -1\n");
    fprintf(fUP, "-------------------------------\n");
    fprintf(fUP, "bin           A          A\n");
    fprintf(fUP, "process       0          1\n");
    fprintf(fUP, "process     signal  background\n");
    fprintf(fUP, "rate         %2d       %4d\n", nS, nB);
    fprintf(fUP, "-------------------------------\n");
    fprintf(fUP, "dB    lnN     -         1.3\n");
    fprintf(fUP, "alpha param   -0.3 0.05 \n");
    fprintf(fUP, "sigma param   1    0.2  \n");
    fclose(fUP);
}
