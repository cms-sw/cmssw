TString gPrefix = "";

const int  nlep = 4;
const char *leps[nlep] = { "el_b", "el_l", "mu_b", "mu_l" };

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



TGraph* drawSlice(TString hname, int ieta, float xdelta, int color, int ist=0, int ifill=0) {
    TH2 *h2d = gFile->Get(hname);
    TH2 *h2d_n = gFile->Get(hname+"_num");
    TH2 *h2d_d = gFile->Get(hname+"_den");
    if (h2d == 0) return 0;
    TGraphAsymmErrors *ret = new TGraphAsymmErrors(h2d->GetNbinsX());
    TAxis *ax = h2d->GetXaxis(); 
    int j = 0;
    for (int i = 0, n = ret->GetN(); i < n; ++i, ++j) {
        ret->SetPoint(j, ax->GetBinCenter(i+1)+xdelta, h2d->GetBinContent(i+1,ieta+1));
        double yup =  h2d->GetBinError(i+1,ieta+1), ydn = h2d->GetBinError(i+1,ieta+1);
        if (h2d_n != 0 && h2d_d != 0) {
            int pass = h2d_n->GetBinContent(i+1,ieta+1), tot =  h2d_d->GetBinContent(i+1,ieta+1); 
            ydn = fabs( h2d->GetBinContent(i+1,ieta+1) - TEfficiency::ClopperPearson(tot, pass, 0.68, 0) );
            yup = fabs( h2d->GetBinContent(i+1,ieta+1) - TEfficiency::ClopperPearson(tot, pass, 0.68, 1) );
        }
        ret->SetPointError(j, 0.5*ax->GetBinWidth(i+1)+xdelta, 0.5*ax->GetBinWidth(i+1)-xdelta, ydn, yup);
    }

    ret->SetLineColor(color);
    ret->SetMarkerColor(color);
    if (ist == 1 || ist == 2) {
        ret->SetLineWidth(1);
        ret->SetFillColor(color);
        ret->SetFillStyle(ifill ? ifill : (ist == 1 ? 3004 : 1001));
        ret->Draw("E5 SAME");
        ret->Draw("E2 SAME");
    } else {
        if (ist == 0) {
            ret->SetLineWidth(2);
        } else {
            ret->SetLineWidth(4);
            ret->SetMarkerSize(1.6);
        }
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

const char *ETALBL2[2] = { "b", "e" };
const char *ETALBL3[3] = { "cb", "fb", "ec" };
const char *ietalbl(int ieta, int ilep) {
    if (ilep < 2) return ETALBL3[ieta];
    else return ETALBL2[ieta];
}
const char *etaspam(int ieta, int ilep) { 
    if (ilep >= 2) {
        return Form("|#eta| %s 1.5", (ieta ? ">" : "<"));
    } else {
        switch (ieta) {
            case 0: return "0 < |#eta| < 0.8";
            case 1: return "0.8 < |#eta| < 1.479";
            case 2: return "|#eta| > 1.479";
        }
    }
    return "NON CAPISCO";
}

void plotFRs(int ilep, TString hbase, TString xtitle, double xmin, double xmax, double ymax) {

    frame = new TH1F("frame",Form(";%s;Fake rate (%s #rightarrow %s)", xtitle.Data(), ilep%2?"j":"b", ilep<2?"e":"#mu"),100,xmin,xmax);
    frame->GetYaxis()->SetRangeUser(0.,ymax > 0 ? ymax : 1);
    frame->GetYaxis()->SetDecimals(1);
    frame->GetXaxis()->SetDecimals(1);
    frame->GetXaxis()->SetNdivisions(505);
    //frame->GetXaxis()->SetLabelOffset(-0.01);
    frame->GetYaxis()->SetTitleOffset(1.40);
    frame->Draw();

    TString hname = Form(hbase.Data(),leps[ilep]);

    int neta = (ilep >= 2 ? 2 : 3);
    TLegend *leg = newLeg(.27,.85-0.05*neta,.45+0.15*(ilep<2),.9);
    int colors[3] = { 214, 209, 99 }; 
    double mymax = 0;
    for (int ieta = 0; ieta < neta; ++ieta) {
        TGraph *g = drawSlice(hname, ieta, (ieta-0.5*(neta-1))*(xmax-xmin)*0.02, colors[ieta]);
        if (g == 0) { std::cerr << "ERROR: missing " << hname << std::endl; continue; }
        leg->AddEntry(g, etaspam(ieta,ilep), "LP");
        for (int i = 0, n = g->GetN(); i < n; ++i) {
            mymax = TMath::Max(mymax, g->GetY()[i]+1.2*g->GetErrorYhigh(i));
        }
    }
    if (ymax <= 0) frame->GetYaxis()->SetRangeUser(0.,1.4*mymax);
    frame->Draw("AXIS SAME");
    leg->Draw();

    cmsprelim(.21, .945, .40, .995, 12, "CMS Preliminary"); 
    cmsprelim(.48, .945, .96, .995, 32, "#sqrt{s} = 8 TeV, L = 19.6 fb^{-1}");
    c1->Print(Form("ttH_plots/250513/FR_QCD_Simple_v2/deps/%s/%s.png", gPrefix.Data(), hname.Data()));
    c1->Print(Form("ttH_plots/250513/FR_QCD_Simple_v2/deps/%s/%s.pdf", gPrefix.Data(), hname.Data()));
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



void plotFRVars(int what=0, int idata=0, int itrigg=1) {
    gSystem->Exec("mkdir -p ttH_plots/250513/FR_QCD_Simple_v2/deps/"+gPrefix);
    gSystem->Exec("cp /afs/cern.ch/user/g/gpetrucc/php/index.php ttH_plots/250513/FR_QCD_Simple_v2/deps/"+gPrefix);
    gROOT->ProcessLine(".x ~gpetrucc/cpp/tdrstyle.cc");
    gStyle->SetOptStat(0);
    initCanvas();
    const int  nsels = 2;
    const char *sels[nsels] = { "FR_tight", "FRC_tight" };
    for (int is = 0; is < nsels; ++is) {
        for (int il = 0; il < nlep; ++il) {
            plotFRs(il, Form("%s_%%s_jet",sels[is]), "N(jet, p_{T} > 25)", 2.5, 6.5, 0.0);
            plotFRs(il, Form("%s_%%s_bjet",sels[is]), "N(bjet, p_{T} > 25, CSVM)", 0.5, 3.5, 0.0);
        }
    }
}
 
