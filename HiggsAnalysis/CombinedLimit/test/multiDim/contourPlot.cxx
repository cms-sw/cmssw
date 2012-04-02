TGraph* bestFit(TTree *t, TString x, TString y) {
    t->Draw(y+":"+x, "quantileExpected == 1");
    TGraph *gr0 = (TGraph*) gROOT->FindObject("Graph")->Clone();
    gr0->SetMarkerStyle(34); gr0->SetMarkerSize(2.0);
    return gr0;
}

TGraph* smValue(double x0 = 1.0, double y0 = 1.0) {
    TGraph* ret = new TGraph(1);
    ret->SetPoint(0, x0, y0);
    ret->SetMarkerStyle(29); ret->SetMarkerSize(4.0);
    ret->SetMarkerColor(4);
    return ret;
}

TGraph* contourPlot(TTree *t, TString x, TString y, double pmin, double pmax, TGraph *bestFit) {
    int n = t->Draw(y+":"+x, Form("%f <= quantileExpected && quantileExpected <= %f && quantileExpected != 1",pmin,pmax));
    std::cout << "Drawing for " << Form("%f <= quantileExpected && quantileExpected <= %f && quantileExpected != 1",pmin,pmax) << " yielded " << n << " points." << std::endl;
    TGraph *gr = (TGraph*) gROOT->FindObject("Graph")->Clone();

    Double_t x0 = bestFit->GetX()[0], y0 = bestFit->GetY()[0];
    Double_t *xi = gr->GetX(), *yi = gr->GetY();
    int n = gr->GetN();
    for (int i = 0; i < n; ++i) { xi[i] -= x0; yi[i] -= y0; }
    gr->Sort(&TGraph::CompareArg);
    for (int i = 0; i < n; ++i) { xi[i] += x0; yi[i] += y0; }
    return gr;
}

void threeContours(TString name, TString x, TString y, TH2 *frame) {
    TTree *t68 = (TTree*) ((TFile*)gROOT->GetListOfFiles()->At(0))->Get("limit");
    TTree *t95 = (TTree*) ((TFile*)gROOT->GetListOfFiles()->At(1))->Get("limit");
    TTree *t99 = (TTree*) ((TFile*)gROOT->GetListOfFiles()->At(2))->Get("limit");
    TGraph *gr0 = bestFit(t68,x,y), *sm = smValue();
    TGraph *gr68 = contourPlot(t68,x,y,0.310,1, gr0);
    TGraph *gr95 = contourPlot(t95,x,y,0.049,1, gr0);
    TGraph *gr99 = contourPlot(t99,x,y,0.009,1, gr0);
    gr68->SetLineWidth(1); gr68->SetLineStyle(1); gr68->SetLineColor(1); gr68->SetFillStyle(1001); gr68->SetFillColor(82);  
    gr95->SetLineWidth(1); gr95->SetLineStyle(7); gr95->SetLineColor(1); gr95->SetFillStyle(1001); gr95->SetFillColor(89);
    gr99->SetLineWidth(1); gr99->SetLineStyle(3); gr99->SetLineColor(1); gr99->SetFillStyle(1001); gr99->SetFillColor(93);
    frame->Draw();
    gr99->Draw("LF SAME");
    gr95->Draw("LF SAME");
    gr68->Draw("LF SAME");
    frame->Draw("AXIS SAME");
    sm->Draw("P SAME");
    gr0->Draw("P SAME");
    gr0->SetName(name+"_best");  gFile->WriteTObject(gr0);
    gr68->SetName(name+"_cl68"); gFile->WriteTObject(gr68);
    gr95->SetName(name+"_cl95"); gFile->WriteTObject(gr95);
    gr99->SetName(name+"_cl99"); gFile->WriteTObject(gr99);
    sm->SetName(name+"_sm");     gFile->WriteTObject(sm);
}

void contourPlot(const char *name, const char *title, double qqHmax=10., double ggHmax=4.) {
    TFile *fOut = TFile::Open(Form("plots/%s.root", name), "RECREATE");
    TH2F *frame = new TH2F(name, Form("%s;#sigma/#sigma_{SM} pp #rightarrow qqH;#sigma/#sigma_{SM} gg #rightarrow H;",title),
                            50, 0, qqHmax, 50, 0., ggHmax);
    TCanvas *c1 = new TCanvas("c1","c1");
    gStyle->SetOptStat(0);
    threeContours(name, "r_qqH", "r_ggH", frame);
    c1->Print(Form("plots/%s.png", name));
    fOut->Close(); 
}
