TString gPrefix = "";
int neta = 2;

TH2F *qfData = 0, *qfDYMC = 0;

void loadData() {
    TFile *fDYMC = TFile::Open("../../data/fakerate/QF_DY_el.root");
    TFile *fData = TFile::Open("../../data/fakerate/QF_data_el.root");
    qfDYMC = (TH2F*) fDYMC->Get("QF_el_DY");
    qfData = (TH2F*) fData->Get("QF_el_data");
}

void cmsprelim(double x1=0.75, double y1=0.40, double x2=0.95, double y2=0.48, const int align=12, const char *text="CMS Preliminary", float textSize=0.033) { 
    TPaveText *cmsprel = new TPaveText(x1,y1,x2,y2,"NDC");
    cmsprel->SetTextSize(textSize);
    cmsprel->SetFillColor(0);
    cmsprel->SetFillStyle(0);
    cmsprel->SetLineStyle(0);
    cmsprel->SetLineColor(0);
    cmsprel->SetTextAlign(align);
    cmsprel->SetTextFont(42);
    cmsprel->AddText(text);
    cmsprel->Draw("same");
}



TGraph* drawSlice(TH2 *h2d, int ieta, int color, int ist=0, int ifill=0) {
    if (h2d == 0) return 0;
    TGraphAsymmErrors *ret = new TGraphAsymmErrors(h2d->GetNbinsX());
    TAxis *ax = h2d->GetXaxis();
    for (int i = 0, n = ret->GetN(); i < n; ++i) {
        ret->SetPoint(i, ax->GetBinCenter(i+1), 100.0*h2d->GetBinContent(i+1,ieta+1));
        if (ist == 2) {
            ret->SetPointError(i, 0.5*ax->GetBinWidth(i+1), 0.5*ax->GetBinWidth(i+1),
                                  100.0*0.4*h2d->GetBinContent(i+1,ieta+1), 100.0*0.4*h2d->GetBinContent(i+1,ieta+1));
        } else {
            ret->SetPointError(i, 0.5*ax->GetBinWidth(i+1), 0.5*ax->GetBinWidth(i+1),
                                  100.0*h2d->GetBinError(i+1,ieta+1), 100.0*h2d->GetBinError(i+1,ieta+1));
        }
    }
    ret->SetLineColor(color);
    ret->SetMarkerColor(color);
    if (ist) {
        ret->SetLineWidth(1);
        ret->SetFillColor(color);
        ret->SetFillStyle(ifill ? ifill : (ist == 1 ? 3004 : 1001));
        ret->Draw("E5 SAME");
        ret->Draw("E2 SAME");
    } else {
        ret->SetLineWidth(2);
        ret->Draw("P SAME");
    }
    return ret;
}


TCanvas *c1 = 0;
TH1 *frame = 0;
TLegend *newLeg(double x1, double y1, double x2, double y2, double textSize=0.035) {
    TLegend *ret = new TLegend(x1,y1,x2,y2);
    ret->SetTextFont(42);
    ret->SetFillColor(0);
    ret->SetShadowColor(0);
    ret->SetTextSize(textSize);
    return ret;
}

const char *ETALBL4[4] = { "b1", "b2", "e1", "e2" };
const char *ETALBL2[2] = { "b", "e" };
const char *ietalbl(int ieta) {
    if (neta == 4) return ETALBL4[ieta];
    else return ETALBL2[ieta];
}

void stackQFs(int ieta) {
    frame->Draw();
    TLegend *leg = newLeg(.27,.75,.52,.9);
    TGraph *MC   = drawSlice(qfDYMC, ieta, 214, 1, 3004);
    TGraph *Data = drawSlice(qfData, ieta, 1);
    leg->AddEntry(Data,  "Z#rightarrowee Data", "LPE");
    leg->AddEntry(MC,    "Z#rightarrowee Sim.", "F");
    leg->Draw();
    cmsprelim(.55, .84, .87, .9, 22, Form("|#eta| %s %s", ieta ? ">" : "<", "1.479"), 0.045);
    cmsprelim(.21, .945, .40, .995, 12, "CMS Preliminary"); 
    cmsprelim(.48, .945, .96, .995, 32, "#sqrt{s} = 8 TeV, L = 19.6 fb^{-1}");
    c1->Print(Form("ttH_plots/250513/QF_Simple/stacks/%s/ele_%s.png", gPrefix.Data(), ietalbl(ieta)));
    c1->Print(Form("ttH_plots/250513/QF_Simple/stacks/%s/ele_%s.pdf", gPrefix.Data(), ietalbl(ieta)));
}

void initCanvas(double w=600, double h=600, double lm=0.21, double rm=0.04) {
    gStyle->SetCanvasDefW(w); //Width of canvas
    gStyle->SetPaperSize(w/h*20.,20.);
    c1 = new TCanvas("c1","c1");
    c1->cd();
    c1->SetWindowSize(w + (w - c1->GetWw()), h + (h - c1->GetWh()));
    c1->SetLeftMargin(lm); 
    c1->SetRightMargin(rm);
    c1->SetGridy(0); c1->SetGridx(0);
}



void plotChargeFlip() {
    loadData();
    gPrefix = "";
    gSystem->Exec("mkdir -p ttH_plots/250513/QF_Simple/stacks/"+gPrefix);
    gROOT->ProcessLine(".x ~gpetrucc/cpp/tdrstyle.cc");
    gStyle->SetOptStat(0);
    initCanvas();
    c1->SetLogx(1);
    frame = new TH1F("frame",";p_{T} (GeV);Charge flip rate",100, 10, 100);
    frame->GetYaxis()->SetRangeUser(0.,1.);
    frame->GetYaxis()->SetDecimals(1);
    frame->GetXaxis()->SetDecimals(1);
    frame->GetXaxis()->SetMoreLogLabels(1);
    frame->GetXaxis()->SetNoExponent(1);
    //frame->GetXaxis()->SetLabelOffset(-0.01);
    frame->GetYaxis()->SetTitleOffset(1.40);
    frame->GetYaxis()->SetNdivisions(505);
    for (int i = 0; i < neta; ++i) {
        frame->GetYaxis()->SetRangeUser(0.0,i ? 0.5 : 0.2);
        frame->GetYaxis()->SetTitle("Charge flip rate [%]");
        stackQFs(i);
    }
}
