void plotEffCurve(const char* canvas,
		  int npts, const double* signalEff, const char* classifier,
		  const double* bgrndEff, const double* bgrndErr,
		  int icolor) 
{
  TCanvas *c 
    = new TCanvas(canvas,"SPR BgrndEff vs SignalEff",200,10,600,400);
  TGraph *gr = new TGraphErrors(npts,signalEff,bgrndEff,0,bgrndErr);
  TLegend *leg = new TLegend(0.1,0.85,0.5,1.,"SPR Classifiers","NDC");
  gr->SetMarkerStyle(21);
  gr->SetLineColor(icolor+1);
  gr->SetLineWidth(3);
  gr->SetMarkerColor(icolor+1);
  gr->Draw("ALP");
  leg->AddEntry(gr,classifier,"P");
  leg->Draw();
}


void plotFOMCurve(const char* canvas, const char* title,
		  int npts, const double* signalEff, const char* classifier,
		  const double* fom, int icolor) 
{
  TCanvas *c 
    = new TCanvas(canvas,"SPR FOM vs SignalEff",200,10,600,400);
  TGraph *gr = new TGraphErrors(npts,signalEff,fom,0,0);
  TLegend *leg = new TLegend(0.1,0.85,0.5,1.,title,"NDC");
  gr->SetMarkerStyle(21);
  gr->SetLineColor(icolor+1);
  gr->SetLineWidth(3);
  gr->SetMarkerColor(icolor+1);
  gr->Draw("ALP");
  leg->AddEntry(gr,classifier,"P");
  leg->Draw();
}


void plotEffCurveMulti(const char* canvas,
		       int ngraph, int npts, const double* signalEff, 
		       const char classifiers[][200],
		       const double* bgrndEff, const double* bgrndErr)
{
  TCanvas *c
    = new TCanvas(canvas,"SPR BgrndEff vs SignalEff",200,10,600,400);
  TMultiGraph *mg = new TMultiGraph();
  TLegend *leg = new TLegend(0.1,0.85,0.5,1.,
			     "SPR BackgroundEff vs SignalEff","NDC");
  for( int i=0;i<ngraph;i++ ) {
    TGraph *gr = new TGraphErrors(npts,signalEff,
				  bgrndEff+(i*npts),0,bgrndErr+(i*npts));
    gr->SetMarkerStyle(21);
    gr->SetLineColor(i+1);
    gr->SetLineWidth(3);
    gr->SetMarkerColor(i+1);
    mg->Add(gr);
    leg->AddEntry(gr,classifiers[i],"P");
  }
  mg->Draw("ALP");
  leg->Draw();
}


void plotCorrelation(const char* canvas, const char* title,
		     const unsigned nVars, const char vars[][200], 
		     const double* corr)
{
  TCanvas *c
    = new TCanvas(canvas,"SPR Variable Correlations",200,10,600,400);
  gStyle->SetPalette(1);
  gStyle->SetOptTitle(1);
  gStyle->SetOptStat(0);
  TH2D* h2 = new TH2D("corr",title,nVars,0,nVars,nVars,0,nVars);
  for( int i=0;i<nVars;i++ ) {
    for( int j=0;j<nVars;j++ ) {
      h2->Fill(vars[i],vars[j],corr[i*nVars+j]);
    }
  }
  h2->LabelsDeflate("X");
  h2->LabelsDeflate("Y");
  h2->LabelsOption("v");
  h2->Draw("Z COL");
}


void plotImportance(const char* canvas, const char* title,
		    const unsigned nVars,
		    const char vars[][200], const double* importance)
{
  TCanvas *c 
    = new TCanvas(canvas,"SPR Variable Importance",200,10,600,400);
  gStyle->SetPalette(1);
  TH1D* h1 = new TH1D("importance",title,nVars,0,nVars);
  for( int i=0;i<nVars;i++ )
    h1->Fill(vars[i],importance[i]);
  h1->LabelsDeflate("X");
  h1->LabelsOption("v");
  h1->SetLineColor(4);
  h1->SetFillColor(4);
  TAxis* yaxis = h1->GetYaxis();
  yaxis->SetRangeUser(0.,1.);
  h1->SetBarWidth(0.8);
  h1->SetBarOffset(0.1);
  h1->Draw("B");
}


// Modes: linear or log
plotHistogram(const char* canvas, const char* mode, const char* title,
	      double xlo, double dx, int nbin,
	      double* sig, double* sigerr, double* bgr, double* bgrerr)
{
  TCanvas *c 
    = new TCanvas(canvas,"SPR Classifier Output",200,10,600,400);
  gStyle->SetPalette(1);
  TLegend *leg = new TLegend(0.1,0.85,0.5,1.,"Classifier Output","NDC");
  double xhi = xlo + nbin*dx;
  TH1D* hs = new TH1D("signal",    title,nbin,xlo,xhi);
  TH1D* hb = new TH1D("background",title,nbin,xlo,xhi);
  leg->AddEntry(hs,"Signal","L");
  leg->AddEntry(hb,"Background","L");
  for( int i=0;i<nbin;i++ ) {
    hs->SetBinContent(i+1,sig[i]);
    hs->SetBinError(i+1,sigerr[i]);
    hb->SetBinContent(i+1,bgr[i]);
    hb->SetBinError(i+1,bgrerr[i]);
  }
  TPad* pad = new TPad("pad","pad",0,0,1,1);
  if( strcmp(mode,"log") == 0 ) pad->SetLogy(1);
  pad->Draw();
  pad->cd();
  hs->SetLineColor(2);
  hs->SetLineWidth(3);
  hb->SetLineColor(4);
  hb->SetLineWidth(3);
  hb->Draw();
  hs->Draw("same");
  leg->Draw();
}


plotClasses(const char* canvas, const char* title,
	    int nClasses, const char classes[][200], 
	    const int* events, const double* weights)
{
  char evtitle[200], wttitle[200];
  strcpy(evtitle,title);
  strcpy(wttitle,title);
  strcat(evtitle,": Events");
  strcat(wttitle,": Weights");

  TCanvas *c 
    = new TCanvas(canvas,"SPR Input Classes",200,10,600,400);
  gStyle->SetPalette(1); 

  int maxEv = TMath::MaxElement(nClasses,events);
  double maxWt = TMath::MaxElement(nClasses,weights);

  TPad* pad1 = new TPad("events", evtitle,0,0,1.,0.5);
  TPad* pad2 = new TPad("weights",wttitle,0,0.5,1.,1.);
  pad1->Draw();
  pad2->Draw();

  // events
  pad1->cd();
  TH1I* hev = new TH1I("events",evtitle,nClasses,0,nClasses);
  for( int i=0;i<nClasses;i++ )
    hev->Fill(classes[i],events[i]);
  hev->LabelsDeflate("X");
  hev->SetLabelSize(0.1,"X");
  hev->SetLabelSize(0.1,"Y");
  hev->SetLineColor(4);
  hev->SetFillColor(4);
  hev->SetBarWidth(0.8);
  hev->SetBarOffset(0.1);
  TAxis* yaxis1 = hev->GetYaxis();
  yaxis1->SetRangeUser(0.,1.1*maxEv);
  hev->Draw("B");

  // weights
  pad2->cd();
  TH1D* hwt = new TH1D("weights",wttitle,nClasses,0,nClasses);
  for( int i=0;i<nClasses;i++ )
    hwt->Fill(classes[i],weights[i]);
  hwt->LabelsDeflate("X");
  hwt->SetLabelSize(0.1,"X");
  hwt->SetLabelSize(0.1,"Y");
  hwt->SetLineColor(3);
  hwt->SetFillColor(3);
  hwt->SetBarWidth(0.8);
  hwt->SetBarOffset(0.1);
  TAxis* yaxis2 = hwt->GetYaxis();
  yaxis2->SetRangeUser(0.,1.1*maxWt);
  hwt->Draw("B");
}
