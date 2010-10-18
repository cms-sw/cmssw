#include "/Users/bechtel/RootMacros/plotStuff.C"

//----------------------------------------------------------------------------------

double error( double a, double b, double aerror, double berror )
{
	// calculate error on ratio a/b 
	// by error propagation

	double theError( 666. );

	double ratio( a / b );
	theError  = ( aerror / a ) * ( aerror / a );
	theError += ( berror / b ) * ( berror / b );
	theError  = ratio * TMath::Sqrt( the Error );

	return theError;
}

//----------------------------------------------------------------------------------

void numberdensity()
{
	//plotNumberDensityByDphiParticles( 500 );
	//plotNumberDensityByDphiParticles( 900 );
	//plotNumberDensityByDphiParticles( 1500 );
	plotNumberDensityByDphiTracks( 500 );
	plotNumberDensityByDphiTracks( 900 );
	plotNumberDensityByDphiTracks( 1500 );

	//plotNumberDensityParticles( 500 );
	//plotNumberDensityParticles( 900 );
	//plotNumberDensityParticles( 1500 );
	plotNumberDensityTracksByHLT( 500 );
	plotNumberDensityTracksByHLT( 900 );
	plotNumberDensityTracksByHLT( 1500 );

	plotNumberDensityTrackParticleRatio( 500 );
	plotNumberDensityTrackParticleRatio( 900 );
	plotNumberDensityTrackParticleRatio( 1500 );

	//plotNumberDensityParticlesRatio( 500, 900 );
	//plotNumberDensityParticlesRatio( 500, 1500 );
	//plotNumberDensityParticlesRatio( 900, 1500 );
	plotNumberDensityTracksRatio( 500, 900 );
	plotNumberDensityTracksRatio( 500, 1500 );
	plotNumberDensityTracksRatio( 900, 1500 );
}

//----------------------------------------------------------------------------------

void plotNumberDensityByDphiParticles(int PTTHRESHOLD)
{
	// plot generator prediction for charged particle ptsum-density vs azimuth with respect to leading jet
	// smooth distribution with TH1::SmoothArray (http://root.cern.ch/root/html/TH1.html#TH1:SmoothArray)
	// based on algorithm 353QH twice presented by J. Friedman
	// in Proc.of the 1974 CERN School of Computing, Norway, 11-24 August, 1974.

	double binwidth( 1. );
	char buffer[200];
	sprintf( buffer, "NumberDensityByDphiParticles%i", PTTHRESHOLD );
	plotCanvas( buffer );
	gPad->SetLogy();

	TH1D* trans = new TH1D("trans","trans", 6, -180., 180.);
	trans->GetXaxis()->SetTitle("#Delta#phi(particles, chg. gen jet) (#circ)");
	trans->GetYaxis()->SetTitle("< dN/d#Delta#phid#eta > (GeV/c/deg)");
	trans->Fill( -90., 150);
	trans->Fill(  90., 150);
	trans->SetMinimum( 0.3 );
	trans->SetMaximum( 10. );
	plotHisto(trans);
	trans->SetFillColor(400);
	trans->SetLineColor(0);
	
	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", PTTHRESHOLD );
	TFile *_file1 = TFile::Open( buffer );

	_file1->cd("All"); TH2D* gen2d = dN_vs_dphiRECO->Clone(); TProfile* genx = gen2d->ProfileX()->Clone();
	genx->SetMarkerStyle( 4 );
	genx->SetMarkerSize( 1.3 );
	
	TH1D* gen = new TH1D( "gen", "", genx->GetNbinsX(), -180., 180. );
	
	// fill gen with smoothed values
	double *bincontents = new double[ gen->GetNbinsX() ];
	for (int i(0); i < gen->GetNbinsX(); ++i) {	bincontents[i] = genx->GetBinContent(i+1); }	
	TH1::SmoothArray(  gen->GetNbinsX(), bincontents, 1);	
	for (int i(0); i < gen->GetNbinsX(); ++i) {	gen->SetBinContent(i+1, bincontents[i]);   }

	plotHisto( gen , "same"     );
	plotHisto( genx, "pe, same" );

	TLatex t;
	t.SetNDC();
	t.DrawLatex(0.281667, 0.943333,"#font[72]{Summer08 MinBias 900GeV}");
//	t.SetTextAngle(90);

	sprintf( buffer, "p_{T}(particles) > %i MeV/c", PTTHRESHOLD );
	t.DrawLatex(0.25, 0.193333, buffer );
	
	sprintf( buffer, "NumberDensityByDphiParticles%i->Print(\"MinBias900GeVNumberDensityByDphiParticlesByHLT%i.eps\");", PTTHRESHOLD, PTTHRESHOLD );
	std::cout << buffer << std::endl;
	
}

//----------------------------------------------------------------------------------

void plotNumberDensityByDphiTracks(int PTTHRESHOLD)
{
	// plot charged particle ptsum-density vs azimuth with respect to leading jet

	double binwidth( 1. );
	char buffer[200];
	sprintf( buffer, "NumberDensityByDphiTracks%i", PTTHRESHOLD );
	plotCanvas( buffer );
	gPad->SetLogy();

	TH1D* trans = new TH1D("trans","trans", 6, -180., 180.);
	trans->GetXaxis()->SetTitle("#Delta#phi(track, tracks jet) (#circ)");
	trans->GetYaxis()->SetTitle("< dN/d#Delta#phid#eta > (GeV/c/deg)");
	trans->Fill( -90., 150);
	trans->Fill(  90., 150);
	trans->SetMinimum( 0.3 );
	trans->SetMaximum( 10. );
	plotHisto(trans);
	trans->SetFillColor(400);
	trans->SetLineColor(0);
	
	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", PTTHRESHOLD );
	TFile *_file1 = TFile::Open( buffer );

	_file1->cd("HLTMinBiasPixel"); TH2D* minbiaspixel = dN_vs_dphiRECO->Clone();
	_file1->cd("HLTMinBiasHcal" ); TH2D* minbiashcal  = dN_vs_dphiRECO->Clone();
	_file1->cd("HLTMinBiasEcal" ); TH2D* minbiasecal  = dN_vs_dphiRECO->Clone();
	_file1->cd("HLTMinBias"     ); TH2D* minbias      = dN_vs_dphiRECO->Clone();
	_file1->cd("HLTZeroBias"    ); TH2D* zerobias     = dN_vs_dphiRECO->Clone();
	_file1->cd("HLT1jet30"      ); TH2D* hlt1jet30    = dN_vs_dphiRECO->Clone();
	_file1->cd("HLT1jet50"      ); TH2D* hlt1jet50    = dN_vs_dphiRECO->Clone();

	TProfile* minbiaspixelx = minbiaspixel->ProfileX()->Clone();
	TProfile* minbiashcalx  = minbiashcal ->ProfileX()->Clone();
	TProfile* minbiasecalx  = minbiasecal ->ProfileX()->Clone();
	TProfile* minbiasx      = minbias     ->ProfileX()->Clone();
	TProfile* zerobiasx     = zerobias    ->ProfileX()->Clone();
	TProfile* hlt1jet30x    = hlt1jet30   ->ProfileX()->Clone();
//	TProfile* hlt1jet50x    = hlt1jet50   ->ProfileX()->Clone();

	minbiaspixelx->SetMarkerStyle( 20 );
	minbiaspixelx->SetMarkerSize( 1. );
	
	minbiashcalx->SetMarkerStyle( 21 );
	minbiashcalx->SetMarkerColor( 4 );
	minbiashcalx->SetLineColor( 4 );
	
	minbiasecalx->SetMarkerStyle( 22 );
	minbiasecalx->SetMarkerSize( 1. );
	minbiasecalx->SetMarkerColor( 632 );
	minbiasecalx->SetLineColor( 632 );
	
	minbiasx->SetMarkerStyle( 24 );
	minbiasx->SetMarkerSize( 1. );
	minbiasx->SetMarkerColor( 402 );
	minbiasx->SetLineColor( 402 );
		
	zerobiasx->SetMarkerStyle( 25 );
	zerobiasx->SetMarkerSize( 1. );
	zerobiasx->SetMarkerColor( 420 );
	zerobiasx->SetLineColor( 420 );
	
	hlt1jet30x->SetMarkerStyle( 26 );
	hlt1jet30x->SetMarkerSize( 1. );
	hlt1jet30x->SetMarkerColor( 619 );
	hlt1jet30x->SetLineColor( 619 );
		
//	hlt1jet50x->SetMarkerStyle( 22 );
//	hlt1jet50x->SetMarkerSize( 1.3 );
//	hlt1jet50x->SetMarkerColor( 632 );
//	hlt1jet50x->SetLineColor( 632 );

	plotHisto( minbiaspixelx, "pe, same" );
	plotHisto( minbiashcalx , "pe, same" );
	plotHisto( minbiasecalx , "pe, same" );
	plotHisto( minbiasx     , "pe, same" );
	plotHisto( zerobiasx    , "pe, same" );
//	plotHisto( hlt1jet30x   , "pe, same" );
//	plotHisto( hlt1jet50x   , "pe, same" );
		
	
	TLegend* leg = new TLegend(0.235 , 0.703333 , 0.641667 , 0.898333);
	leg->SetFillStyle( 0 ); 
	leg->SetBorderSize( 0. ); 
 	leg->SetTextFont(42); 
	leg->SetTextSize(0.045); 
	leg->AddEntry(minbiaspixelx, "MinBiasPixel", "lp");
	leg->AddEntry(minbiashcalx, "#color[4]{MinBiasHcal}", "lp");
	leg->AddEntry(minbiasecalx, "#color[632]{MinBiasEcal}", "lp");
	leg->Draw();
		
	TLegend* leg2 = new TLegend(0.605 , 0.766667 , 0.98 , 0.898333);
//	TLegend* leg2 = new TLegend(0.605 , 0.703333 , 0.98 , 0.898333);
	leg2->SetFillStyle( 0 );
	leg2->SetBorderSize( 0. );
	leg2->SetTextFont(42);
	leg2->SetTextSize(0.045);
	leg2->AddEntry(minbiasx, "#color[402]{MinBias}", "lp");
	leg2->AddEntry(zerobiasx, "#color[420]{ZeroBias}", "lp");
//	leg2->AddEntry(f1_dpt_900, "Pythia D6T", "l");
//	leg2->AddEntry(hlt1jet30x, "#color[420]{Jet30}", "lp");
	leg2->Draw();

	TLatex t;
	t.SetNDC();
	t.DrawLatex(0.281667, 0.943333,"#font[72]{Summer08 MinBias 900GeV}");
//	t.SetTextAngle(90);

	sprintf( buffer, "p_{T}(tracks) > %i MeV/c", PTTHRESHOLD );
	t.DrawLatex(0.25, 0.193333, buffer );
	
	sprintf( buffer, "NumberDensityByDphiTracks%i->Print(\"MinBias900GeVNumberDensityByDphiTracksByHLT%i.eps\");", PTTHRESHOLD, PTTHRESHOLD );
	std::cout << buffer << std::endl;
	
}


//----------------------------------------------------------------------------------

void plotNumberDensityParticles(int PTTHRESHOLD)
{
	// plot generator prediction for charged particle ptsum-density in transverse region vs pT(leading jet)
	// smooth distribution with TH1::SmoothArray (http://root.cern.ch/root/html/TH1.html#TH1:SmoothArray)
	// based on algorithm 353QH twice presented by J. Friedman
	// in Proc.of the 1974 CERN School of Computing, Norway, 11-24 August, 1974.

	double binwidth( 2. );
	char buffer[200];
	sprintf( buffer, "NumberDensityParticles%i", PTTHRESHOLD );
	plotCanvas( buffer );
	
	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", PTTHRESHOLD );
	TFile *_file1 = TFile::Open( buffer );

	_file1->cd("All"); TH2D* gen2d = dN_vs_ptJTransMC->Clone();
	TProfile* genx = gen2d->ProfileX()->Clone();
	genx->SetMarkerStyle( 4 );
	genx->SetMarkerSize( 1.3 );

	TH1D* gen = new TH1D( "gen", ";p_{T}(chg. gen jet) (GeV/c);<dN/d#phid#eta> / 2 GeV/c (GeV/c/rad)", 10, 0., 20. );
	gen->SetMinimum( 0. );
	gen->SetMaximum( 0.8 );
	
	// fill gen with smoothed values
	double *bincontents = new double[ gen->GetNbinsX() ];
	for (int i(0); i < gen->GetNbinsX(); ++i) {	bincontents[i] = genx->GetBinContent(i+1); }	
	TH1::SmoothArray(  gen->GetNbinsX(), bincontents, 50);	
	for (int i(0); i < gen->GetNbinsX(); ++i) {	gen->SetBinContent(i+1, bincontents[i]);   }

	plotHisto( gen              );
	plotHisto( genx, "pe, same" );
		
	TLatex t;
	t.SetNDC();
	t.DrawLatex(0.31, 0.943333,"900GeV, Transverse region");

	sprintf( buffer, "p_{T}(chg. particles) > %i MeV/c", PTTHRESHOLD );
	t.DrawLatex(0.25, 0.193333, buffer );
	
	sprintf( buffer, "NumberDensityParticles%i->Print(\"MinBias900GeVNumberDensityParticles%i.eps\");", PTTHRESHOLD, PTTHRESHOLD );
	std::cout << buffer << std::endl;
	return;
	
}

//----------------------------------------------------------------------------------

TH1D* getTH1DNumberDensityParticles(int PTTHRESHOLD)
{
	// create histogram of generator prediction for charged particle ptsum-density in transverse region vs pT(leading jet)
	// smooth distribution with TH1::SmoothArray (http://root.cern.ch/root/html/TH1.html#TH1:SmoothArray)
	// based on algorithm 353QH twice presented by J. Friedman
	// in Proc.of the 1974 CERN School of Computing, Norway, 11-24 August, 1974.

	char buffer[200];
	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", PTTHRESHOLD );
	TFile *_file1 = TFile::Open( buffer );

	_file1->cd("All"); TH2D* gen2d = dN_vs_ptJTransMC->Clone(); TProfile* genx = gen2d->ProfileX()->Clone();
	TH1D* gen = new TH1D( "gen", "", 10, 0., 20. );
	
	// fill gen with smoothed values
	double *bincontents = new double[ gen->GetNbinsX() ];
	for (int i(0); i < gen->GetNbinsX(); ++i) {	bincontents[i] = genx->GetBinContent(i+1); }	
	TH1::SmoothArray(  gen->GetNbinsX(), bincontents, 50);	
	for (int i(0); i < gen->GetNbinsX(); ++i) {	gen->SetBinContent(i+1, bincontents[i]); gen->SetBinError(i+1,genx->GetBinError(i+1));  }

	gen->SetMarkerSize( 0. );
	gen->SetLineColor(594);
	gen->SetFillColor(594);

	return gen;
	
}

//----------------------------------------------------------------------------------

void plotNumberDensityParticlesRatio( int NOM, int DENOM )
{	
	// plot ratio between smoothed generator predictions for pT thresholds NOM and DENOM
	// smooth done with TH1::SmoothArray (http://root.cern.ch/root/html/TH1.html#TH1:SmoothArray)
	// based on algorithm 353QH twice presented by J. Friedman
	// in Proc.of the 1974 CERN School of Computing, Norway, 11-24 August, 1974.

	char buffer[200];
	sprintf( buffer, "ratioParticles%iBy%i", NOM, DENOM );
	plotCanvas( buffer );

	sprintf( buffer, ";p_{T}(chg. gen jet) (GeV/c);dN(%i MeV/c) / dN(%i GeV/c)", NOM, DENOM );
	TH1D* h_ratio = new TH1D( "h_ratio", buffer, 10, 0., 20. );

	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", NOM ); TFile *_file1 = TFile::Open( buffer );
	_file1->cd("All"); TH2D* nom2d = dN_vs_ptJTransMC->Clone(); TProfile* nomx = nom2d->ProfileX()->Clone();

	// fill nom with smoothed values
	TH1D* nom = new TH1D( "nom", "", 10, 0., 20. );
	double *bincontents = new double[ nom->GetNbinsX() ];
	for (int i(0); i < nom->GetNbinsX(); ++i) {	bincontents[i] = nomx->GetBinContent(i+1); }	
	TH1::SmoothArray(  nom->GetNbinsX(), bincontents, 50);	
	for (int i(0); i < nom->GetNbinsX(); ++i) {	nom->SetBinContent(i+1, bincontents[i]); nom->SetBinError(i+1,nomx->GetBinError(i+1)); }

	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", DENOM ); TFile *_file1 = TFile::Open( buffer );
	_file1->cd("All"); TH2D* denom2d = dN_vs_ptJTransMC->Clone(); TProfile* denomx = denom2d->ProfileX()->Clone();

	// fill denom with smoothed values
	TH1D* denom = new TH1D( "denom", "", 10, 0., 20. );
	for (int i(0); i < denom->GetNbinsX(); ++i) {	bincontents[i] = denomx->GetBinContent(i+1); }	
	TH1::SmoothArray(  denom->GetNbinsX(), bincontents, 50);	
	for (int i(0); i < denom->GetNbinsX(); ++i) {	denom->SetBinContent(i+1, bincontents[i]); denom->SetBinError(i+1,denomx->GetBinError(i+1)); }

	for ( size_t ibin(1); ibin<=nom->GetNbinsX(); ++ibin )
	{
		if (   nom->GetBinContent( ibin ) == 0. ) continue;
		if ( denom->GetBinContent( ibin ) == 0. ) continue;
		h_ratio->SetBinContent( ibin, nom->GetBinContent( ibin ) / denom->GetBinContent( ibin ) );
		h_ratio->SetBinError  ( ibin, error( nom->GetBinContent( ibin ), denom->GetBinContent( ibin ),
											 nom->GetBinError( ibin ), denom->GetBinError( ibin )));
	}


	h_ratio->SetMinimum( 0.5 );
	h_ratio->SetMaximum( 2.5 );
	h_ratio->SetMarkerSize( 0.0 );
	h_ratio->SetFillColor(594);
	h_ratio->GetXaxis()->SetRange(0,15);
	plotHisto( h_ratio, "p,e3" );
	
	TLatex t;
	t.SetNDC();
	//t.DrawLatex(0.308333, 0.946667,"Prompt reco, startup cond.");
	//t.DrawLatex(0.308333, 0.946667,"Re-reco 1 pb^{-1} cond. (S43)");
	//t.DrawLatex(0.308333, 0.946667,"Re-reco 10 pb^{-1} cond. (S156)");
	t.DrawLatex(0.261667, 0.211667,"900 GeV, Transverse region");

	sprintf( buffer, "#frac{dN(p_{T}(particles) > %i MeV/c)}{dN(p_{T}(particles) > %i MeV/c)}", NOM, DENOM );
	t.DrawLatex(0.258333, 0.79, buffer);
	
	sprintf( buffer, "ratioParticles%iBy%i->Print(\"MinBias900GeVNumberDensityParticlesRatio%iBy%i.eps\");", NOM, DENOM, NOM, DENOM );
	std::cout << buffer << std::endl;



}

//----------------------------------------------------------------------------------

TH1D* getTH1DNumberDensityParticlesRatio( int NOM, int DENOM )
{	
	// create ratio histogram between smoothed generator predictions for pT thresholds NOM and DENOM
	// smooth done with TH1::SmoothArray (http://root.cern.ch/root/html/TH1.html#TH1:SmoothArray)
	// based on algorithm 353QH twice presented by J. Friedman
	// in Proc.of the 1974 CERN School of Computing, Norway, 11-24 August, 1974.

	char buffer[200];

	sprintf( buffer, ";p_{T}(chg. gen jet) (GeV/c);dN(%i MeV/c) / dN(%i GeV/c)", NOM, DENOM );
	TH1D* h_ratio = new TH1D( "h_ratio", buffer, 10, 0., 20. );

	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", NOM ); TFile *_file1 = TFile::Open( buffer );
	_file1->cd("All"); TH2D* nom2d = dN_vs_ptJTransMC->Clone(); TProfile* nomx = nom2d->ProfileX()->Clone();

	// fill nom with smoothed values
	TH1D* nom = new TH1D( "nom", "", 10, 0., 20. );
	double *bincontents = new double[ nom->GetNbinsX() ];
	for (int i(0); i < nom->GetNbinsX(); ++i) {	bincontents[i] = nomx->GetBinContent(i+1); }	
	TH1::SmoothArray(  nom->GetNbinsX(), bincontents, 50);	
	for (int i(0); i < nom->GetNbinsX(); ++i) {	nom->SetBinContent(i+1, bincontents[i]); nom->SetBinError(i+1,nomx->GetBinError(i+1)); }

	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", DENOM ); TFile *_file1 = TFile::Open( buffer );
	_file1->cd("All"); TH2D* denom2d = dN_vs_ptJTransMC->Clone(); TProfile* denomx = denom2d->ProfileX()->Clone();

	// fill denom with smoothed values
	TH1D* denom = new TH1D( "denom", "", 10, 0., 20. );
	for (int i(0); i < denom->GetNbinsX(); ++i) {	bincontents[i] = denomx->GetBinContent(i+1); }	
	TH1::SmoothArray(  denom->GetNbinsX(), bincontents, 50);	
	for (int i(0); i < denom->GetNbinsX(); ++i) {	denom->SetBinContent(i+1, bincontents[i]); denom->SetBinError(i+1,denomx->GetBinError(i+1)); }

	for ( size_t ibin(1); ibin<=nom->GetNbinsX(); ++ibin )
	{
		if (   nom->GetBinContent( ibin ) == 0. ) continue;
		if ( denom->GetBinContent( ibin ) == 0. ) continue;
		h_ratio->SetBinContent( ibin, nom->GetBinContent( ibin ) / denom->GetBinContent( ibin ) );
		h_ratio->SetBinError  ( ibin, error( nom->GetBinContent( ibin ), denom->GetBinContent( ibin ),
											 nom->GetBinError( ibin ), denom->GetBinError( ibin )));
	}


	h_ratio->SetMinimum( 0.5 );
	h_ratio->SetMaximum( 2.5 );
	h_ratio->SetMarkerSize( 0.0 );
	h_ratio->SetLineColor(594);
	h_ratio->SetFillColor(594);
	h_ratio->GetXaxis()->SetRange(0,15);
	return h_ratio;
}

//----------------------------------------------------------------------------------

void plotNumberDensityTracksByHLT(int PTTHRESHOLD)
{
	double binwidth( 2. );
	char buffer[200];
	sprintf( buffer, "NumberDensityTracks%i", PTTHRESHOLD );
	plotCanvas( buffer );
	
	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", PTTHRESHOLD );
	TFile *_file1 = TFile::Open( buffer );

	_file1->cd("HLTMinBiasPixel"); TH2D* minbiaspixel = dN_vs_ptJTransRECO->Clone();
	_file1->cd("HLTMinBiasHcal" ); TH2D* minbiashcal  = dN_vs_ptJTransRECO->Clone();
	_file1->cd("HLTMinBiasEcal" ); TH2D* minbiasecal  = dN_vs_ptJTransRECO->Clone();
	_file1->cd("HLTMinBias"     ); TH2D* minbias      = dN_vs_ptJTransRECO->Clone();
	_file1->cd("HLTZeroBias"    ); TH2D* zerobias     = dN_vs_ptJTransRECO->Clone();
	_file1->cd("HLT1jet30"      ); TH2D* hlt1jet30    = dN_vs_ptJTransRECO->Clone();
	_file1->cd("HLT1jet50"      ); TH2D* hlt1jet50    = dN_vs_ptJTransRECO->Clone();

	TH1D* dummy = new TH1D( "dummy", ";p_{T}(tracks jet) (GeV/c);<dN/d#phid#eta> / 2 GeV/c (GeV/c/rad)", 100, 0., 20. );
	dummy->SetMinimum( 0. );
	dummy->SetMaximum( 0.65 );

	TProfile* minbiaspixelx = minbiaspixel->ProfileX()->Clone();
	TProfile* minbiashcalx  = minbiashcal ->ProfileX()->Clone();
	TProfile* minbiasecalx  = minbiasecal ->ProfileX()->Clone();
	TProfile* minbiasx      = minbias     ->ProfileX()->Clone();
	TProfile* zerobiasx     = zerobias    ->ProfileX()->Clone();
	TProfile* hlt1jet30x    = hlt1jet30   ->ProfileX()->Clone();
//	TProfile* hlt1jet50x    = hlt1jet50   ->ProfileX()->Clone();

	minbiaspixelx->SetMarkerStyle( 20 );
	minbiaspixelx->SetMarkerSize( 1. );
	
	minbiashcalx->SetMarkerStyle( 21 );
	minbiashcalx->SetMarkerColor( 4 );
	minbiashcalx->SetLineColor( 4 );
	
	minbiasecalx->SetMarkerStyle( 22 );
	minbiasecalx->SetMarkerSize( 1. );
	minbiasecalx->SetMarkerColor( 632 );
	minbiasecalx->SetLineColor( 632 );
	
	minbiasx->SetMarkerStyle( 24 );
	minbiasx->SetMarkerSize( 1. );
	minbiasx->SetMarkerColor( 402 );
	minbiasx->SetLineColor( 402 );
		
	zerobiasx->SetMarkerStyle( 25 );
	zerobiasx->SetMarkerSize( 1. );
	zerobiasx->SetMarkerColor( 420 );
	zerobiasx->SetLineColor( 420 );
	
	hlt1jet30x->SetMarkerStyle( 26 );
	hlt1jet30x->SetMarkerSize( 1. );
	hlt1jet30x->SetMarkerColor( 619 );
	hlt1jet30x->SetLineColor( 619 );
		
//	hlt1jet50x->SetMarkerStyle( 22 );
//	hlt1jet50x->SetMarkerSize( 1.3 );
//	hlt1jet50x->SetMarkerColor( 632 );
//	hlt1jet50x->SetLineColor( 632 );

	TH1D* gen    = getTH1DNumberDensityParticles( PTTHRESHOLD );

	plotHisto( dummy                     );
	plotHisto( gen          , "e3, same" );
	plotHisto( minbiaspixelx, "pe, same" );
	plotHisto( minbiashcalx , "pe, same" );
	plotHisto( minbiasecalx , "pe, same" );
	plotHisto( minbiasx     , "pe, same" );
	plotHisto( zerobiasx    , "pe, same" );
//	plotHisto( hlt1jet30x   , "pe, same" );
//	plotHisto( hlt1jet50x   , "pe, same" );
		
	
	TLegend* leg = new TLegend(0.235 , 0.703333 , 0.641667 , 0.898333);
	leg->SetFillStyle( 0 ); 
	leg->SetBorderSize( 0. ); 
 	leg->SetTextFont(42); 
	leg->SetTextSize(0.045); 
	leg->AddEntry(minbiaspixelx, "MinBiasPixel", "lp");
	leg->AddEntry(minbiashcalx, "#color[4]{MinBiasHcal}", "lp");
	leg->AddEntry(minbiasecalx, "#color[632]{MinBiasEcal}", "lp");
	leg->Draw();
		
//	TLegend* leg2 = new TLegend(0.605 , 0.766667 , 0.98 , 0.898333);
	TLegend* leg2 = new TLegend(0.605 , 0.703333 , 0.98 , 0.898333);
	leg2->SetFillStyle( 0 );
	leg2->SetBorderSize( 0. );
	leg2->SetTextFont(42);
	leg2->SetTextSize(0.045);
	leg2->AddEntry(minbiasx, "#color[402]{MinBias}", "lp");
	leg2->AddEntry(zerobiasx, "#color[420]{ZeroBias}", "lp");
	leg2->AddEntry(gen, "#color[594]{Pythia 6 D6T}", "f");
//	leg2->AddEntry(hlt1jet30x, "#color[420]{Jet30}", "lp");
	leg2->Draw();

	TLatex t;
	t.SetNDC();
	t.DrawLatex(0.31, 0.943333,"900GeV, Transverse region");
//	t.SetTextAngle(90);

	sprintf( buffer, "p_{T}(tracks) > %i MeV/c", PTTHRESHOLD );
	t.DrawLatex(0.25, 0.193333, buffer );
	
	sprintf( buffer, "NumberDensityTracks%i->Print(\"MinBias900GeVNumberDensityTracksByHLT%i.eps\");", PTTHRESHOLD, PTTHRESHOLD );
	std::cout << buffer << std::endl;
	return;
	
}

//----------------------------------------------------------------------------------

void plotNumberDensityTracksRatio( int NOM, int DENOM )
{	
	char buffer[200];
	sprintf( buffer, "ratioTracks%iBy%i", NOM, DENOM );
	plotCanvas( buffer );

	sprintf( buffer, ";p_{T}(tracks jet) (GeV/c);dN(%i MeV/c) / dN(%i GeV/c)", NOM, DENOM );
	TH1D* h_ratio = new TH1D( "h_ratio", buffer, 100, 0., 200. );

	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", NOM ); TFile *_file1 = TFile::Open( buffer );
	_file1->cd("HLTZeroBias"); TH2D* nom2d = dN_vs_ptJTransRECO->Clone(); TProfile* nom = nom2d->ProfileX()->Clone();

	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", DENOM ); TFile *_file1 = TFile::Open( buffer );
	_file1->cd("HLTZeroBias"); TH2D* denom2d = dN_vs_ptJTransRECO->Clone(); TProfile* denom = denom2d->ProfileX()->Clone();

	for ( size_t ibin(1); ibin<=nom->GetNbinsX(); ++ibin )
	{
		if (   nom->GetBinContent( ibin ) == 0. ) continue;
		if ( denom->GetBinContent( ibin ) == 0. ) continue;
		h_ratio->SetBinContent( ibin, nom->GetBinContent( ibin ) / denom->GetBinContent( ibin ) );
		h_ratio->SetBinError  ( ibin, error( nom->GetBinContent( ibin ), denom->GetBinContent( ibin ),
											 nom->GetBinError( ibin ), denom->GetBinError( ibin )));
	}

	h_ratio->SetMarkerStyle( 4 );
	h_ratio->SetMarkerSize( 1.3 );
	h_ratio->GetXaxis()->SetRange(0,15);

	TH1D* gen = getTH1DNumberDensityParticlesRatio( NOM, DENOM );
	gen->SetMinimum( 0.5 );
	gen->SetMaximum( 4. );
	plotHisto( gen    , "p, e3"    );
	plotHisto( h_ratio, "pe, same" );

	TLegend* leg = new TLegend(0.236667 , 0.575 , 0.62 , 0.666667);
	leg->SetFillStyle( 0 ); 
	leg->SetBorderSize( 0. ); 
 	leg->SetTextFont(42); 
	leg->SetTextSize(0.045); 
	leg->AddEntry(gen, "#color[594]{Pythia 6 D6T}", "f");
	leg->Draw();
	
	TLatex t;
	t.SetNDC();
	//t.DrawLatex(0.308333, 0.946667,"Prompt reco, startup cond.");
	//t.DrawLatex(0.308333, 0.946667,"Re-reco 1 pb^{-1} cond. (S43)");
	//t.DrawLatex(0.308333, 0.946667,"Re-reco 10 pb^{-1} cond. (S156)");
	t.DrawLatex(0.261667, 0.211667,"900 GeV, Transverse region");

	sprintf( buffer, "#frac{dN(p_{T}(tracks) > %i MeV/c)}{dN(p_{T}(tracks) > %i MeV/c)}", NOM, DENOM );
	t.DrawLatex(0.298333, 0.785, buffer);
	
	sprintf( buffer, "ratioTracks%iBy%i->Print(\"MinBias900GeVNumberDensityTracksRatio%iBy%i.eps\");", NOM, DENOM, NOM, DENOM );
	std::cout << buffer << std::endl;



}

//----------------------------------------------------------------------------------

void plotNumberDensityTrackParticleRatio( int PTTHRESHOLD )
{	
	// plot ratio between smoothed generator predictions and track measurements for pT threshold PTTHRESHOLD
	// smooth done with TH1::SmoothArray (http://root.cern.ch/root/html/TH1.html#TH1:SmoothArray)
	// based on algorithm 353QH twice presented by J. Friedman
	// in Proc.of the 1974 CERN School of Computing, Norway, 11-24 August, 1974.

	char buffer[200];
	sprintf( buffer, "ratioTrackParticle%i", PTTHRESHOLD );
	plotCanvas( buffer );

	TH1D* h_ratio_temp = new TH1D( "h_ratio_temp", "", 10, 0., 20. );

	sprintf( buffer, "MinBias900GeV.UE%iMeV.root", PTTHRESHOLD ); TFile *_file1 = TFile::Open( buffer );
	_file1->cd("All"); TH2D* nom2d = dN_vs_ptJTransMC->Clone(); TProfile* nomx = nom2d->ProfileX()->Clone();

	// fill nom with smoothed values
	TH1D* nom = new TH1D( "nom", "", 10, 0., 20. );
	double *bincontents = new double[ nom->GetNbinsX() ];
	for (int i(0); i < nom->GetNbinsX(); ++i) {	bincontents[i] = nomx->GetBinContent(i+1); }	
	TH1::SmoothArray(  nom->GetNbinsX(), bincontents, 50);	
	for (int i(0); i < nom->GetNbinsX(); ++i) {	nom->SetBinContent(i+1, bincontents[i]); nom->SetBinError(i+1,nomx->GetBinError(i+1)); }
	
	_file1->cd("HLTZeroBias"); TH2D* denom2d = dN_vs_ptJTransRECO->Clone(); TProfile* denom = denom2d->ProfileX()->Clone();

	for ( size_t ibin(1); ibin<=nom->GetNbinsX(); ++ibin )
	{
		if (   nom->GetBinContent( ibin ) == 0. ) continue;
		if ( denom->GetBinContent( ibin ) == 0. ) continue;
		h_ratio_temp->SetBinContent( ibin, nom->GetBinContent( ibin ) / denom->GetBinContent( ibin ) );
		h_ratio_temp->SetBinError  ( ibin, error( nom->GetBinContent( ibin ), denom->GetBinContent( ibin ),
												  nom->GetBinError  ( ibin ), denom->GetBinError  ( ibin ) ));
	}
	
	// fill h_ratio with smoothed values
	TH1D* h_ratio = new TH1D( "h_ratio", ";p_{T}(charged jet) (GeV/c);Multiplicity correction", 10, 0., 20. );
	for (int i(0); i < h_ratio->GetNbinsX(); ++i) {	bincontents[i] = h_ratio_temp->GetBinContent(i+1); }	
	TH1::SmoothArray(  h_ratio->GetNbinsX(), bincontents, 50);	
	for (int i(0); i < h_ratio->GetNbinsX(); ++i) {	h_ratio->SetBinContent(i+1, bincontents[i]); h_ratio->SetBinError(i+1,h_ratio_temp->GetBinError(i+1)); }

	h_ratio_temp->SetMarkerStyle( 4 );
	h_ratio_temp->SetMarkerSize( 1.3 );
	h_ratio_temp->SetMarkerColor( 921 );
	h_ratio_temp->SetLineColor( 921 );

	h_ratio->SetMinimum( 0.65 );
	h_ratio->SetMaximum( 1.45 );
	h_ratio->GetXaxis()->SetRange(0,15);
	plotHisto( h_ratio     , "hist"    );
	plotHisto( h_ratio_temp, "pe same" );
	
	TLatex t;
	t.SetNDC();
	//t.DrawLatex(0.308333, 0.946667,"Prompt reco, startup cond.");
	//t.DrawLatex(0.308333, 0.946667,"Re-reco 1 pb^{-1} cond. (S43)");
	t.DrawLatex(0.25, 0.838333,"#font[72]{HLT_ZeroBias}");
	t.DrawLatex(0.31, 0.943333,"900 GeV, Transverse region");

	sprintf( buffer, "p_{T}(particles, tracks) > %i MeV/c", PTTHRESHOLD );
	t.DrawLatex(0.25, 0.193333, buffer);
	
	sprintf( buffer, "ratioTrackParticle%i->Print(\"MinBias900GeVNumberDensityTrackParticleRatio%i.eps\");", PTTHRESHOLD, PTTHRESHOLD );
	std::cout << buffer << std::endl;



}

