TGraph* bestFit(TH2* hist) {
    TGraph* ret = new TGraph(1);
    int nx = hist->GetNbinsX(), ny = hist->GetNbinsY();
    double max = -1;
    TAxis *ax = hist->GetXaxis(), *ay = hist->GetYaxis();
    for (int i = 1; i <= nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            double p = hist->GetBinContent(i,j);
            if (p > max) {
                max = p;
                ret->SetPoint(0, ax->GetBinCenter(i), ay->GetBinCenter(j));
            }
        }
    }
    ret->SetMarkerStyle(34); ret->SetMarkerSize(2.0);
    return ret;
}

TGraph* smValue(double x0 = 1.0, double y0 = 1.0) {
    TGraph* ret = new TGraph(1);
    ret->SetPoint(0, x0, y0);
    ret->SetMarkerStyle(29); ret->SetMarkerSize(4.0);
    ret->SetMarkerColor(4);
    return ret;
}

TGraph* bayesContour(TH2 *hist, double cl) {
    int nx = hist->GetNbinsX(), ny = hist->GetNbinsY(), n2 = nx * ny;
    Double_t *vals = new Double_t[n2];
    Double_t sum = 0;
    for (int i = 1, k = 0; i <= nx; ++i) { 
        for (int j = 1; j <= ny; ++j, ++k) {
            vals[k] = hist->GetBinContent(i,j);
            sum += vals[k];
        }
    }
    Int_t *index = new Int_t[n2];
    TMath::Sort(n2, vals, index);
    double cut = cl*sum, runsum = 0, threshold = -1;
    for (int i = 0; i < n2; ++i) {
        runsum += vals[index[i]];
        if (runsum > cut) { threshold = vals[index[i]]; break; }
    }
    TGraph *points = new TGraph(); int p = 0;
    TAxis *ax = hist->GetXaxis(), *ay = hist->GetYaxis();
    for (int i = 1, k = 0; i <= nx; ++i) {
        if (hist->GetBinContent(i,ny) > threshold) {
            points->Set(++p); points->SetPoint(p-1, ax->GetBinCenter(i), ay->GetBinUpEdge(ny));
        } else {
            for (int j = ny-1; j > 0; --j) {
                if (hist->GetBinContent(i,j) > threshold) {
                    points->Set(++p); points->SetPoint(p-1, ax->GetBinCenter(i), ay->GetBinCenter(j));
                    break;
                }
            }
        }
    }
    for (int i = nx; i > 0; --i) {
        if (hist->GetBinContent(i,1) > threshold) {
            points->Set(++p); points->SetPoint(p-1, ax->GetBinCenter(i), ay->GetBinLowEdge(1));
        } else {
            for (int j = 2; j <= ny; ++j) {
                if (hist->GetBinContent(i,j) > threshold) {
                    points->Set(++p); points->SetPoint(p-1, ax->GetBinCenter(i), ay->GetBinCenter(j));
                    break;
                }
            }
        }
    }
    return points;
}

void threeContours(TString name, TH2 *posterior) {
    TGraph *gr0 = bestFit(posterior), *sm = smValue();
    TGraph *gr68 = bayesContour(posterior, 0.68);
    TGraph *gr95 = bayesContour(posterior, 0.95);
    TGraph *gr99 = bayesContour(posterior, 0.99);
    gr68->SetLineWidth(1); gr68->SetLineStyle(1); gr68->SetLineColor(1); gr68->SetFillStyle(1001); gr68->SetFillColor(82);  
    gr95->SetLineWidth(1); gr95->SetLineStyle(7); gr95->SetLineColor(1); gr95->SetFillStyle(1001); gr95->SetFillColor(89);
    gr99->SetLineWidth(1); gr99->SetLineStyle(3); gr99->SetLineColor(1); gr99->SetFillStyle(1001); gr99->SetFillColor(93);
    TH2 *frame = (TH2*) posterior->Clone(); frame->Reset();
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

void bayesContours(const char *name, const char *title) {
    TFile *fIn = TFile::Open(Form("plots/%s.root", name));
    TFile *fOut = TFile::Open(Form("plots/%s_cl.root", name), "RECREATE");
    TH2 *frame = (TH2*) fIn->Get(name);
    TCanvas *c1 = new TCanvas("c1","c1");
    gStyle->SetOptStat(0);
    threeContours(name, frame);
    c1->Print(Form("plots/%s_cl.png", name));
    fOut->Close(); 
}
