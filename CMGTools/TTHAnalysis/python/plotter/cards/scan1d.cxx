double findCrossingOfScan1D(TGraph *graph, double threshold, bool leftSide, double xmin=-9e99, double xmax=9e99) {
    double *x = graph->GetX();
    double *y = graph->GetY();
    int imin = 0, n = graph->GetN();
    for (int i = 1; i < n; ++i) {
        if (x[i] < xmin || x[i] > xmax) continue;
        if (y[i] < y[imin]) imin = i;
    }
    int imatch = -1;
    if (leftSide) {
        for (int i = imin; i >= 0; --i) {
            if (x[i] < xmin || x[i] > xmax) continue;
            if (y[i] > threshold && y[i+1] < threshold) {
                imatch = i; break;
            }
        }
        if (imatch == -1) return x[0];
    } else {
        for (int i = imin; i < n; ++i) {
            if (x[i] < xmin || x[i] > xmax) continue;
            if (y[i-1] < threshold && y[i] > threshold) {
                imatch = i-1; break;
            }
        }
        if (imatch == -1) return x[n-1];
    }
    double d1 = fabs(y[imatch] - threshold), d2 = fabs(y[imatch+1] - threshold);
    return (x[imatch]*d2 + x[imatch+1]*d1)/(d1+d2);
}
double muHatFromTGraphScan(TGraph *graph, bool interpolate=false, double *y0=0, double xmin=-9e99, double xmax=9e99) {
    int iMin = 0, n = graph->GetN(); double *xi = graph->GetX(), *yi = graph->GetY();
    for (int i = 1; i < n; ++i) {
        if (xi[i] < xmin || xi[i] > xmax) continue;
        if (yi[i] < yi[iMin]) iMin = i;
    }
    if (!interpolate || iMin == 0 || iMin == n-1) {
        if (y0) *y0 = yi[iMin];
        return graph->GetX()[iMin];
    }
    TSpline3 spline("dummy",graph);
    double x0 = graph->GetX()[iMin-1], x2 = graph->GetX()[iMin+1];
    double xmin = x0, ymin = graph->GetY()[iMin-1];
    for (double x = x0, dx = (x2-x0)*0.02; x < x2; x += dx) {
        double y = spline.Eval(x);
        if (y < ymin) { ymin = y; xmin = x; }
    }
    if (ymin > yi[iMin]) ymin = yi[iMin];
    if (y0) *y0 = ymin;
    return x;
}

TLegend *newLegend(double x1, double y1, double x2, double y2) {
    TLegend *leg = new TLegend(x1,y1,x2,y2); 
    leg->SetFillColor(0);
    leg->SetShadowColor(0);
    leg->SetTextFont(42);
    leg->SetTextSize(0.05);
    return leg;
}

TPaveText *cmsprel;
void spam(const char *text=0, double x1=0.17, double y1=0.89, double x2=0.58, double y2=0.94, int textAlign=-1, bool fill=true, float fontSize=0.03) {
   if (textAlign == -1) textAlign=12;
   cmsprel = new TPaveText(x1,y1,x2,y2,"brtlNDC");
   cmsprel->SetTextSize(fontSize);
   cmsprel->SetFillColor(0);
   cmsprel->SetFillStyle((fill || text == 0) ? 1001 : 0);
   cmsprel->SetLineStyle(0);
   cmsprel->SetLineColor(0);
   cmsprel->SetLineWidth(0);
   cmsprel->SetTextAlign(textAlign);
   cmsprel->SetTextFont(42);
   cmsprel->AddText(text);
   cmsprel->SetBorderSize(0);
   cmsprel->Draw("same");
}

void drawOneScan1D(TGraph *obs, TGraph *fast, TString what="auto", double xmin0=-1, double xmax0=-1, double ndivx=510, double ymax=10, TString postfix="", bool renorm=true) { 
    TString myname, fastname, repostfix = "";
    myname   = "Observed";
    fastname = "Exp. for SM H";
    double xmin = obs->GetX()[0];
    double xmax = obs->GetX()[obs->GetN()-1];
    if (xmin0 < xmax0) { xmin = xmin0; xmax = xmax0; } 
    double ymin = 9e99;
    double muhat = muHatFromTGraphScan(obs, true, &ymin, xmin, xmax);
    if (renorm) {
        if (fabs(ymin) > 1e-4) { 
            obs = (TGraph*) obs->Clone();
            for (int i = 0, n = obs->GetN(); i < n; ++i) {
                obs->GetY()[i] -= ymin;
            }
        }
        if (fast) {
            ymin = 9e99;
            muHatFromTGraphScan(fast, true, &ymin, xmin, xmax);
            if (fabs(ymin) > 1e-4) {
                fast = (TGraph*) fast->Clone();
                for (int i = 0, n = fast->GetN(); i < n; ++i) {
                    fast->GetY()[i] -= ymin;
                }
            }
        }
    }
    int col = 1; //214;
    obs->SetLineWidth(4);
    obs->SetLineColor(col); obs->SetMarkerColor(col); 
    if (fast) {
        fast->SetLineWidth(4);   fast->SetLineWidth(2); fast->SetLineStyle(7);
        fast->SetLineColor(col); fast->SetMarkerColor(col); 
        leg = newLegend(.60,.78,.93,.93); leg->SetTextSize(0.04);
        leg->AddEntry(obs,  myname,   "L");
        leg->AddEntry(fast, fastname, "L");
    }
    TH1D *frame0 = new TH1D("frame","frame", 100, xmin, xmax); 
    frame0->Draw(); gStyle->SetOptStat(0);
    double ymin = 0; 
    if (fast) fast->Draw("CX");
    obs->Draw("CX");    
    frame0->GetXaxis()->SetTitle(what);
    frame0->GetYaxis()->SetTitle("- 2 #Delta ln L");
    frame0->GetYaxis()->SetRangeUser(ymin,ymax);
    double hi68 = findCrossingOfScan1D(obs, 1.00, false, xmin, xmax);
    double lo68 = findCrossingOfScan1D(obs, 1.00, true,  xmin, xmax);
    double hi95 = findCrossingOfScan1D(obs, 3.84, false, xmin, xmax);
    double lo95 = findCrossingOfScan1D(obs, 3.84, true,  xmin, xmax);
    printf("ranges for %s: [%.2f,%.2f] at 68%%, [%.2f,%.2f] at 95\%\n", obs->GetName(), lo68,hi68,lo95,hi95);
    TLine chi2(xmin, 3.84, xmax, 3.84);
    chi2.SetLineColor(2); chi2.SetLineStyle(1); chi2.SetLineWidth(2);
    double dx = 0.03*(xmax-xmin);
    if (ymax > 3.84 && (hi95 < xmax-dx || lo95 > xmin+dx)) chi2.DrawClone();
    chi2.SetLineWidth(2);
    if (lo95 > xmin+dx) chi2.DrawLine(lo95, 0, lo95, TMath::Min(ymax,3.84)); 
    if (lo68 > xmin+dx) chi2.DrawLine(lo68, 0, lo68, 1.00);
    if (hi95 < xmax-dx) chi2.DrawLine(hi95, 0, hi95, TMath::Min(ymax,3.84));    
    if (hi68 < xmax-dx) chi2.DrawLine(hi68, 0, hi68, 1.00);
    frame0->GetXaxis()->SetNdivisions(ndivx);
    spam("CMS Preliminary",                    .17, .955, .40, .995,  12, false, 0.0355);
    spam("#sqrt{s} = 8 TeV,  L = 19.6 fb^{-1}",.48, .955, .975, .995, 32, false, 0.0355);
    spam(what+Form(" = %.1f^{ +%.1f}_{ -%.1f}", muhat, hi68-muhat, fabs(lo68-muhat)), .23, .82, .58, .92, 22, true, 0.05);
    leg->Draw();
}

TCanvas *c1 = 0;
void squareCanvas(bool gridx=0,bool gridy=0) {
    gStyle->SetCanvasDefW(600); //Width of canvas
    gStyle->SetPaperSize(20.,20.);
    if (c1) c1->Close();
    c1 = new TCanvas("c1","c1");
    c1->cd();
    c1->SetWindowSize(600 + (600 - c1->GetWw()), 600 + (600 - c1->GetWh()));
    c1->SetRightMargin(0.05);
    c1->SetGridy(gridy); c1->SetGridx(gridx);
}

TString globalPrefix = "";
void justSave(TString name, double xmin=0., double xmax=0., double ymin=0., double ymax=0., const char *tspam=0, bool spamLow=false) {
    c1->Print(globalPrefix+name+".eps");
    TString convOpt = "-q  -dBATCH -dSAFER  -dNOPAUSE  -dAlignToPixels=0 -dEPSCrop  -dPrinted -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sDEVICE=png16m";
    TString convCmd = Form("gs %s -sOutputFile=%s.png -q \"%s.eps\" -c showpage -c quit", convOpt.Data(), (globalPrefix+name).Data(), (globalPrefix+name).Data());
    gSystem->Exec(convCmd);
    if (name.Contains("pval_ml_")) {
        gSystem->Exec("epstopdf "+globalPrefix+name+".eps --outfile="+globalPrefix+name+".pdf");
    } else {
        c1->Print(globalPrefix+name+".pdf");
    }
}



void scan1d(TString plot="scan1d", TString par="r") {
    globalPrefix = "plots/v4/";
    squareCanvas();

    TFile *fobs = (TFile*) gROOT->GetListOfFiles()->At(0);
    TFile *fexp = (TFile*) gROOT->GetListOfFiles()->At(1);
    TTree *tobs = (TTree*) fobs->Get("limit");
    TTree *texp = (TTree*) fexp->Get("limit");
    tobs->Draw("2*deltaNLL:"+par, "deltaNLL<100");
    TGraph *obs = (TGraph*) gROOT->FindObject("Graph")->Clone("obs"); obs->Sort();
    texp->Draw("2*deltaNLL:"+par, "deltaNLL<100");
    TGraph *exp = (TGraph*) gROOT->FindObject("Graph")->Clone("exp"); exp->Sort();
  
    obs->SetLineWidth(4); 
    exp->SetLineWidth(3); 
    exp->SetLineStyle(2); 

    TString what = "#mu(ttH)"; double xmin = -1, xmax = -1;
    if (plot.Contains("ttZ") && !plot.Contains("ttH")) { what = "#mu(ttZ)"; xmin = 0; xmax = 4; }
    if (plot.Contains("ttW") && !plot.Contains("ttH")) { what = "#mu(ttW)"; xmin = 0; xmax = 4; }

    drawOneScan1D(obs,exp,what,xmin,xmax);
    justSave(plot);
}
