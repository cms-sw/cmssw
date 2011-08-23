TCanvas *c1 = new TCanvas("c1","c1");


void makeNplot(RooWorkspace *w, TString pdf, TString var, double refval, double newval, double x, TString algo) {
    w->var(var)->setVal(refval); TH1 *h_nominal = w->pdf(pdf)->createHistogram("x");
    w->var(var)->setVal(newval); TH1 *h_shift = w->pdf(pdf)->createHistogram("x");
    w->var(var)->setVal(x*newval+(1-x)*refval); TH1 *h_mid = w->pdf(pdf)->createHistogram("x");
    w->var(var)->setVal(refval);

    RooArgList hobs(*w->var("x"));
    RooArgSet obs(*w->var("x"));
    RooHistPdf *p_nominal = new RooHistPdf("p_nominal", "", obs, *(new RooDataHist("d_nominal","",hobs,h_nominal)));
    RooHistPdf *p_mid     = new RooHistPdf("p_mid", "",     obs, *(new RooDataHist("d_mid","",hobs,h_mid)));
    RooHistPdf *p_shift   = new RooHistPdf("p_shift", "",   obs, *(new RooDataHist("d_shift","",hobs,h_shift)));

    w->var("a")->setVal(x);
    RooArgList pdfs, coeffs(*w->var("a"));
    pdfs.add(*p_nominal);
    pdfs.add(*p_shift);
    pdfs.add(*p_shift);

    RooAbsPdf *morph;
#ifdef OLD
    if (algo == "vert_linear") {
        morph = new VerticalInterpPdf("m","m",pdfs,coeffs,0.,0);
    } else if (algo == "vert_linear_numint") {
        morph = new VerticalInterpPdf("m","m",pdfs,coeffs,0.,100);
    } else if (algo == "vert_log") {
        morph = new VerticalInterpPdf("m","m",pdfs,coeffs,0.,-1);
#else
    if (algo == "vert_linear") {
        morph = new VerticalInterpHistPdf("m","m",*w->var("x"),pdfs,coeffs,1.0,1);
    } else if (algo == "vert_log") {
        morph = new VerticalInterpHistPdf("m","m",*w->var("x"),pdfs,coeffs,1.0,-1);
#endif
    } else if (algo == "horizontal") {
        if (x >= 1 || x <= 0) return;
        w->var("a")->setVal(x*w->var("a")->getMax());
        morph = new RooIntegralMorph("m","m",*p_nominal,*p_shift,*w->var("x"),*w->var("a"));
    } else {
        std::cerr << "unknown algo: " << algo <<std::endl;
        return;
    }
    
    RooPlot *frame = w->var("x")->frame();
    p_nominal->plotOn(frame,LineColor(kBlue),LineWidth(3));
    w->var(var)->setVal(refval); w->pdf(pdf)->plotOn(frame,LineColor(kBlue),LineWidth(2));
    p_shift->plotOn(frame,LineColor(kRed),LineWidth(3));
    w->var(var)->setVal(newval); w->pdf(pdf)->plotOn(frame,LineColor(kRed),LineWidth(2));
    p_mid->plotOn(frame,LineColor(kBlack),LineWidth(3),LineStyle(2));
    w->var(var)->setVal(x*newval+(1-x)*refval); w->pdf(pdf)->plotOn(frame,LineColor(kBlack),LineWidth(3),LineStyle(2));
    morph->plotOn(frame,LineColor(209),LineWidth(3));
    frame->Draw();
    c1->Print(TString::Format("mplots/%s_%s_ref%.1f_shift%.1f_x%.1f_%s.png", pdf.Data(), var.Data(), refval, newval, x, algo.Data()));
    delete morph;
    delete p_nominal;
    delete p_shift;
    delete p_mid;
}
void make3plot(RooWorkspace *w, TString pdf, TString var, double refval, double newval, double x) {
    makeNplot(w,pdf,var,refval,newval,x,"vert_linear");
#ifdef OLD
    makeNplot(w,pdf,var,refval,newval,x,"vert_linear_numint");
#endif
    makeNplot(w,pdf,var,refval,newval,x,"vert_log");
    //if (x >= 0 && x <= 1) makeNplot(w,pdf,var,refval,newval,x,"horizontal");
}

void morphing_tests(int nS=10, int nB=100, int seed=37) {
    using namespace RooFit;
    gSystem->Load("libHiggsAnalysisCombinedLimit.so");

    RooRandom::randomGenerator()->SetSeed(seed); 

    RooWorkspace *w = new RooWorkspace();
    w->factory("x[0,10]");
    w->factory("a[0,5]");
    w->var("x")->setBins(20);
    w->factory("Exponential::expo(x,alpha[-0.3])");
    w->factory("Gaussian::gauss(x,mu[6],sigma[1])");
    w->factory("CBShape::cb(x,cbm[1.7],cbs[0.5],cba[-1],cbn[2])");
    make3plot(w,"expo","alpha",-0.3,-0.5,0.5);
    make3plot(w,"expo","alpha",-0.3,-0.5,1.5);
    make3plot(w,"expo","alpha",-0.3,-0.5,3.0);
    make3plot(w,"expo","alpha",-0.4,-0.2,0.5);
    make3plot(w,"expo","alpha",-0.4,-0.2,1.5);
    make3plot(w,"expo","alpha",-0.4,-0.2,3.0);
    make3plot(w,"expo","alpha",-0.6,-0.1,0.333);
    make3plot(w,"expo","alpha",-0.1,-0.6,0.333);

    w->var("x")->setMax(1.0);
    make3plot(w,"expo","alpha",-0.1,-0.2,0.5);
    make3plot(w,"expo","alpha",-0.1,-0.2,1.5);
    make3plot(w,"expo","alpha",-0.1,-0.2,3.0);
    make3plot(w,"expo","alpha",-0.2,-0.1,1.5);
    make3plot(w,"expo","alpha",-0.2,-0.1,3.0);

    w->var("x")->setMax(10.0);
    make3plot(w,"gauss","mu",3.5,5.0,0.5);
    make3plot(w,"gauss","mu",3.5,5.0,1.5);
    make3plot(w,"gauss","mu",3.5,5.0,3.0);
    make3plot(w,"gauss","mu",3.5,4.0,0.5);
    make3plot(w,"gauss","mu",3.5,4.0,1.5);
    make3plot(w,"gauss","mu",3.5,4.0,3.0);
    make3plot(w,"gauss","mu",3.0,5.0,0.333);
    w->var("sigma")->setVal(2.5);
    make3plot(w,"gauss","mu",3.2,5.7,0.2);

    w->var("x")->setMax(10.0);
    make3plot(w,"gauss","sigma",1.0,0.7,0.5);
    make3plot(w,"gauss","sigma",1.0,0.7,1.5);
    make3plot(w,"gauss","sigma",1.0,0.8,3.0);
    make3plot(w,"gauss","sigma",1.0,0.4,0.3333);
    make3plot(w,"gauss","sigma",1.0,1.3,0.5);
    make3plot(w,"gauss","sigma",1.0,1.3,1.5);
    make3plot(w,"gauss","sigma",1.0,1.2,3.0);
    make3plot(w,"gauss","sigma",1.0,1.6,0.3333);

    make3plot(w,"cb","cba",-1,-1.5,0.5);
    make3plot(w,"cb","cba",-1,-0.5,0.5);
    make3plot(w,"cb","cba",-1,-1.2,1.5);
    make3plot(w,"cb","cba",-1,-0.8,1.5);
    make3plot(w,"cb","cba",-1,-1.2,3.0);
    make3plot(w,"cb","cba",-1,-0.8,3.0);

}
