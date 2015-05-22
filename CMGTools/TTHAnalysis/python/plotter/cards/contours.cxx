TGraph* bestFit(TTree *t, TString x, TString y, TCut cut) {
    int nfind = t->Draw(y+":"+x, cut + "deltaNLL == 0");
    if (nfind == 0) {
        TGraph *gr0 = new TGraph(1);
        gr0->SetPoint(0,-999,-999);
        gr0->SetMarkerStyle(34); gr0->SetMarkerSize(2.0);
        return gr0;
    } else {
        TGraph *gr0 = (TGraph*) gROOT->FindObject("Graph")->Clone();
        gr0->SetMarkerStyle(34); gr0->SetMarkerSize(2.0);
        if (gr0->GetN() > 1) gr0->Set(1);
        return gr0;
    }
}

TH2 *treeToHist2D(TTree *t, TString x, TString y, TString name, TCut cut, double xmin, double xmax, double ymin, double ymax, int xbins, int ybins) {
    t->Draw(Form("2*deltaNLL:%s:%s>>%s_prof(%d,%10g,%10g,%d,%10g,%10g)", y.Data(), x.Data(), name.Data(), xbins, xmin, xmax, ybins, ymin, ymax), cut + "deltaNLL != 0", "PROF");
    TH2 *prof = (TH2*) gROOT->FindObject(name+"_prof");
    TH2D *h2d = new TH2D(name, name, xbins, xmin, xmax, ybins, ymin, ymax);
    for (int ix = 1; ix <= xbins; ++ix) {
        for (int iy = 1; iy <= ybins; ++iy) {
             double z = prof->GetBinContent(ix,iy);
             if (z != z) z = (name.Contains("bayes") ? 0 : 999); // protect agains NANs
             h2d->SetBinContent(ix, iy, z);
        }
    }
    h2d->SetDirectory(0);
    return h2d;
}

TList* contourFromTH2(TH2 *h2in, double threshold, int minPoints=20) {
    std::cout << "Getting contour at threshold " << threshold << " from " << h2in->GetName() << std::endl;
    //http://root.cern.ch/root/html/tutorials/hist/ContourList.C.html
    Double_t contours[1];
    contours[0] = threshold;
    if (h2in->GetNbinsX() * h2in->GetNbinsY() > 10000) minPoints = 50;

    TH2D *h2 = frameTH2D((TH2D*)h2in,threshold);

    h2->SetContour(1, contours);

    // Draw contours as filled regions, and Save points
    h2->Draw("CONT Z LIST");
    gPad->Update(); // Needed to force the plotting and retrieve the contours in TGraphs


    // Get Contours
    TObjArray *conts = (TObjArray*)gROOT->GetListOfSpecials()->FindObject("contours");
    TList* contLevel = NULL;

    if (conts == NULL || conts->GetSize() == 0){
        printf("*** No Contours Were Extracted!\n");
        return 0;
    }

    TList *ret = new TList();
    for(int i = 0; i < conts->GetSize(); i++){
        contLevel = (TList*)conts->At(i);
        printf("Contour %d has %d Graphs\n", i, contLevel->GetSize());
        for (int j = 0, n = contLevel->GetSize(); j < n; ++j) {
            TGraph *gr1 = (TGraph*) contLevel->At(j);
            if (gr1->GetN() > minPoints) ret->Add(gr1->Clone());
            //break;
        }
    }
    return ret;
}

TH2D* frameTH2D(TH2D *in, double threshold){
        // NEW LOGIC:
        //   - pretend that the center of the last bin is on the border if the frame
        //   - add one tiny frame with huge values
        double frameValue = 1000;
        if (TString(in->GetName()).Contains("bayes")) frameValue = 0.0;

	Double_t xw = in->GetXaxis()->GetBinWidth(1);
	Double_t yw = in->GetYaxis()->GetBinWidth(1);

	Int_t nx = in->GetNbinsX();
	Int_t ny = in->GetNbinsY();

	Double_t x0 = in->GetXaxis()->GetXmin();
	Double_t x1 = in->GetXaxis()->GetXmax();

	Double_t y0 = in->GetYaxis()->GetXmin();
	Double_t y1 = in->GetYaxis()->GetXmax();
        Double_t xbins[999], ybins[999]; 
        double eps = 0.1;

        xbins[0] = x0 - eps*xw - xw; xbins[1] = x0 + eps*xw - xw;
        for (int ix = 2; ix <= nx; ++ix) xbins[ix] = x0 + (ix-1)*xw;
        xbins[nx+1] = x1 - eps*xw + 0.5*xw; xbins[nx+2] = x1 + eps*xw + xw;

        ybins[0] = y0 - eps*yw - yw; ybins[1] = y0 + eps*yw - yw;
        for (int iy = 2; iy <= ny; ++iy) ybins[iy] = y0 + (iy-1)*yw;
        ybins[ny+1] = y1 - eps*yw + yw; ybins[ny+2] = y1 + eps*yw + yw;
        
	TH2D *framed = new TH2D(
			Form("%s framed",in->GetName()),
			Form("%s framed",in->GetTitle()),
			nx + 2, xbins,
			ny + 2, ybins 
			);

	//Copy over the contents
	for(int ix = 1; ix <= nx ; ix++){
		for(int iy = 1; iy <= ny ; iy++){
			framed->SetBinContent(1+ix, 1+iy, in->GetBinContent(ix,iy));
		}
	}
	//Frame with huge values
	nx = framed->GetNbinsX();
	ny = framed->GetNbinsY();
	for(int ix = 1; ix <= nx ; ix++){
		framed->SetBinContent(ix,  1, frameValue);
		framed->SetBinContent(ix, ny, frameValue);
	}
	for(int iy = 2; iy <= ny-1 ; iy++){
		framed->SetBinContent( 1, iy, frameValue);
		framed->SetBinContent(nx, iy, frameValue);
	}

	return framed;
}
void styleMultiGraph(TList *tmg, int lineColor, int lineWidth, int lineStyle) {
    for (int i = 0; i < tmg->GetSize(); ++i) {
        TGraph *g = (TGraph*) tmg->At(i);
        g->SetLineColor(lineColor); g->SetLineWidth(lineWidth); g->SetLineStyle(lineStyle);
    }
}
void styleMultiGraphMarker(TList *tmg, int markerColor, int markerSize, int markerStyle) {
    for (int i = 0; i < tmg->GetSize(); ++i) {
        TGraph *g = (TGraph*) tmg->At(i);
        g->SetMarkerColor(markerColor); g->SetMarkerSize(markerSize); g->SetMarkerStyle(markerStyle);
    }
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




void contour(TString xvar, float xmin, float xmax, TString yvar, float ymin, float ymax, float yzoom) {
    TTree *tree = (TTree*) gFile->Get("limit") ;
    TH2 *hist2d = treeToHist2D(tree, xvar, yvar, "h2d", "", xmin, xmax, ymin, ymax, 50, 50);
    TGraph *fit = bestFit(tree, xvar, yvar, "");
    TList *c68 = contourFromTH2(hist2d, 2.30);
    TList *c95 = contourFromTH2(hist2d, 5.99);
    styleMultiGraph(c68, /*color=*/1, /*width=*/3, /*style=*/1);
    styleMultiGraph(c95, /*color=*/1, /*width=*/3, /*style=*/9);
    hist2d->Draw("AXIS"); gStyle->SetOptStat(0);
    hist2d->GetYaxis()->SetRangeUser(ymin,yzoom);
    c68->Draw("L SAME");
    c95->Draw("L SAME");
    fit->Draw("P SAME");
    double smx = 1.0, smy = 1.0; TMarker m;
    m.SetMarkerSize(3.0); m.SetMarkerColor(97); m.SetMarkerStyle(33); 
    m.DrawMarker(smx,smy);
    m.SetMarkerSize(1.8); m.SetMarkerColor(89); m.SetMarkerStyle(33); 
    m.DrawMarker(smx,smy);
    TString xtit = "#mu(ttH)";
    if (xvar != "r") { xtit = "#mu("+xvar+")"; xtit.ReplaceAll("r_",""); }
    TString ytit = "#mu("+yvar+")"; ytit.ReplaceAll("r_",""); 
    ytit.ReplaceAll("_el"," e"); ytit.ReplaceAll("_mu"," #mu");
    hist2d->GetXaxis()->SetTitle(xtit);
    hist2d->GetYaxis()->SetTitle(ytit);
    spam("CMS Preliminary",                    .17, .955, .40, .995,  12, false, 0.0355);
    spam("#sqrt{s} = 8 TeV,  L = 19.6 fb^{-1}",.48, .955, .975, .995, 32, false, 0.0355);
}
void contours(TString plot="scan2d",TString par2="r_ttW",TString par1="r") {
    globalPrefix = "plots/v4/";
    squareCanvas();
    double xmin = 0, xmax = 6;
    double ymin = 0, ymax = 6, yzoom = 6;
    if (par1 == "r")     { xmin = -2; xmax = 6; }
    if (par1 == "r_ttW") { xmin = 0; xmax = 6; }
    if (par2 == "r_ttW") { ymin = 0; ymax = 6; yzoom = 4.5;}
    if (par2 == "r_ttZ") { ymin = 0; ymax = 6; yzoom = 3.5;}
    if (par2 == "r_fake_mu") { ymin = 0; ymax = 10; yzoom = 2.5; }
    if (par2 == "r_fake_el") { ymin = 0; ymax = 10; yzoom = 2.5; }
    contour(par1,xmin,xmax,par2,ymin,ymax,yzoom);
    justSave(plot);
}
