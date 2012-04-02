TGraph* smValue(double x0 = 1.0, double y0 = 1.0) {
    TGraph* ret = new TGraph(1);
    ret->SetPoint(0, x0, y0);
    ret->SetMarkerStyle(29); ret->SetMarkerSize(4.0);
    ret->SetMarkerColor(1);
    return ret;
}

void bayesPosterior2D(TH2F *frame, TString x, TString y) { 
    TDirectory *toyDir = gFile->GetDirectory("toys");
    if (toyDir == 0) {
        std::cerr << "Error in file " << gROOT->GetListOfFiles()->At(i)->GetName() << ": directory /toys not found" << std::endl;
        continue;
    }
    TString prefix = "MarkovChain_";
    TIter next(toyDir->GetListOfKeys()); TKey *k;
    while ((k = (TKey *) next()) != 0) {
        if (TString(k->GetName()).Index(prefix) != 0) continue;
        RooStats::MarkovChain *chain = dynamic_cast<RooStats::MarkovChain *>(toyDir->Get(k->GetName()));
        const RooDataSet *rds = chain->GetAsConstDataSet();
        int entries = rds->numEntries();
        std::cout << "Got chain " << chain->GetName() << " with " << entries << " entries." << std::endl;
        int burnIn = entries/2;
        for (int i = burnIn; i < entries; ++i) {
            const RooArgSet *point = rds->get(i);
            frame->Fill(point->getRealValue(x), point->getRealValue(y), rds->weight());
            //std::cout << i<<"/"<<entries<<": " 
            //    << x << " = " << point->getRealValue(x) << ", " << y << " = " << point->getRealValue(y) <<
            //    " (weight " << rds->weight() << ")" << std::endl;
        }
    }
}
void bayesPosterior2D(const char *name, const char *title, double qqHmax=10., double ggHmax=4.) {
    TH2F *frame = new TH2F(name, Form("%s;#sigma/#sigma_{SM} pp #rightarrow qqH;#sigma/#sigma_{SM} gg #rightarrow H;",title),
                            50, 0, qqHmax, 50, 0., ggHmax); 
    frame->SetContour(100);
    bayesPosterior2D(frame,"r_qqH","r_ggH");
    TCanvas *c1 = new TCanvas("c1","c1");
    gStyle->SetOptStat(0);
    frame->Draw("COLZ");
    TGraph *sm = smValue(); sm->Draw("P SAME");
    c1->Print(Form("plots/%s.png", name));
    TFile *fOut = TFile::Open(Form("plots/%s.root", name), "RECREATE");
    fOut->WriteTObject(frame);
    fOut->Close();
}
