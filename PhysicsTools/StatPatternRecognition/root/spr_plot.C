// $Id: spr_plot.C,v 1.6 2007/11/30 20:13:30 narsky Exp $

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
		       const double* bgrndEff, const double* bgrndErr,
		       double ymax, bool setMax=false)
{
  TCanvas *c
    = new TCanvas(canvas,"SPR BgrndEff vs SignalEff",200,10,600,400);
  TMultiGraph *mg = new TMultiGraph();
  if( setMax ) mg->SetMaximum(ymax);
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
  h2->SetLabelSize(0.06,"X");
  h2->SetLabelSize(0.06,"Y");
  h2->LabelsDeflate("X");
  h2->LabelsDeflate("Y");
  h2->LabelsOption("v");
  h2->Draw("Z COL");
}


void plotImportance(const char* canvas, const char* title,
		    const unsigned nVars,
		    const char vars[][200], 
		    const double* importance, const double* error,
		    bool automaticRange=false)
{
  TCanvas *c 
    = new TCanvas(canvas,"SPR Variable Importance",200,10,600,400);
  gStyle->SetPalette(1);
  TH1D* h1 = new TH1D("importance",title,nVars,0,nVars);
  double ymin(0), ymax(0);
  for( int i=0;i<nVars;i++ ) {
    h1->Fill(vars[i],importance[i]);
    if( automaticRange ) {
      ymin = ( ymin>importance[i] ? importance[i] : ymin );
      ymax = ( ymax<importance[i] ? importance[i] : ymax );
    }
  }
  if( automaticRange ) {
    ymin = ( ymin<0 ? ymin*1.2 : ymin*0.8 );
    ymax *= 1.2;
  }
  else {
    ymin = 0.;
    ymax = 1.;
  }
  if( error == 0 ) {
    for( int i=0;i<nVars;i++ )
      h1->SetBinError(i+1,0.);
  }
  else {
    for( int i=0;i<nVars;i++ )
      h1->SetBinError(i+1,error[i]);
  }
  h1->LabelsDeflate("X");
  h1->LabelsOption("v");
  h1->SetLabelSize(0.06,"X");
  h1->SetLineColor(4);
  h1->SetMarkerStyle(21);
  h1->SetMarkerColor(4);
  h1->SetLineWidth(2);
  TAxis* yaxis = h1->GetYaxis();
  yaxis->SetRangeUser(ymin,ymax);
  if( error == 0 ) {
    h1->SetLineColor(4);
    h1->SetFillColor(4);
    h1->SetBarWidth(0.8);
    h1->SetBarOffset(0.1);
    h1->Draw("B");
  }
  else
    h1->Draw("E0P");
  l = new TLine(0,0,nVars,0); l->Draw();
}


// Modes: linear or log
plotHistogram(const char* canvas, const char* mode, const char* title,
	      double xlo, double xhi, int nbin,
	      double* sig, double* sigerr, double* bgr, double* bgrerr)
{
  TCanvas *c 
    = new TCanvas(canvas,"SPR Classifier Output",200,10,600,400);
  gStyle->SetPalette(1);
  TLegend *leg = new TLegend(0.1,0.85,0.5,1.,"Classifier Output","NDC");
  double dx = (xhi-xlo) / nbin;
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
  hev->SetLabelSize(0.06,"X");
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
  hwt->SetLabelSize(0.06,"X");
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


void plotMultiClassTable(const char* canvas, const char* title,
			 const int nClass, const char classes[][200], 
			 const double* mcTable)
{
  TCanvas *c = 
    new TCanvas(canvas,"SPR Assigned Class Label (Y) vs True Class Label (X)",
		200,10,600,400);
  gStyle->SetPalette(1);
  gStyle->SetOptTitle(1);
  gStyle->SetOptStat(0);
  TH2D* h2 = new TH2D("MultiClass",title,nClass,0,nClass,nClass,0,nClass);
  for( int i=0;i<nClass;i++ ) {
    for( int j=0;j<nClass;j++ ) {
      h2->Fill(classes[i],classes[j],mcTable[i*nClass+j]);
    }
  }
  h2->LabelsDeflate("X");
  h2->LabelsDeflate("Y");
  h2->SetLabelSize(0.06,"X");
  h2->SetLabelSize(0.06,"Y");
  h2->LabelsOption("v");
  h2->Draw("Z COL");
}
