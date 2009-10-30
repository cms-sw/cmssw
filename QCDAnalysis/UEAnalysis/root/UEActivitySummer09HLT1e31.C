#include "UEActivity.h"
#include <vector>
#include <math.h>

using std::string;

///
///_______________________________________________________________________
///
UEActivity::UEActivity()
{
  _leadingLet = new TLorentzVector();

  h_pTChg       = new TH1D*[3];
  h_dN_vs_dphi  = new TH1D*[3];
  h_dpT_vs_dphi = new TH1D*[3];

  char buffer[200];
  for ( unsigned int iregion(0); iregion<3; ++iregion )
    {
      sprintf ( buffer, "h_pTChg[%i]", iregion );
      h_pTChg[iregion] = new TH1D( buffer, buffer, 1000,  0. , 100. );
      
      sprintf ( buffer, "h_dN_vs_dphi[%i]", iregion );
      h_dN_vs_dphi[iregion] = new TH1D( buffer, buffer, 360, -180., 180.);

      sprintf ( buffer, "h_dpT_vs_dphi[%i]", iregion );
      h_dpT_vs_dphi[iregion] = new TH1D( buffer, buffer, 360, -180., 180.);
    }
}


///
///_______________________________________________________________________
///
UEActivityFinder::UEActivityFinder( double theEtaRegion, double thePtThreshold )
{
  etaRegion   = theEtaRegion;
  ptThreshold = thePtThreshold;

  _h_pTChg       = new TH1D*[3];
  _h_dN_vs_dphi  = new TH1D*[3];
  _h_dpT_vs_dphi = new TH1D*[3];

  char buffer[200];
  for ( unsigned int iregion(0); iregion<3; ++iregion )
    {
      sprintf ( buffer, "_h_pTChg[%i]", iregion );
      _h_pTChg[iregion] = new TH1D( buffer, buffer, 1000,  0. , 100. );

      sprintf ( buffer, "_h_dN_vs_dphi[%i]", iregion );
      _h_dN_vs_dphi[iregion] = new TH1D( buffer, buffer, 360, -180., 180.);

      sprintf ( buffer, "_h_dpT_vs_dphi[%i]", iregion );
      _h_dpT_vs_dphi[iregion] = new TH1D( buffer, buffer, 360, -180., 180.);
    }
}


///
///_______________________________________________________________________
///
Bool_t 
UEActivityFinder::find( TClonesArray& Jet, TClonesArray& Particles, UEActivity& theUEActivity )
{
  for ( unsigned int iregion(0); iregion<3; ++iregion )
    {
      _h_pTChg      [iregion]->Reset();
      _h_dN_vs_dphi [iregion]->Reset();
      _h_dpT_vs_dphi[iregion]->Reset();
    }

  ///
  /// find leading jet in chosen (pT, eta)-range
  ///
  TLorentzVector* leadingJet;
  bool foundLeadingJet( false );
  for(int j=0; j<Jet.GetSize();++j)
    {
      TLorentzVector *v = (TLorentzVector*)Jet.At(j);
      if( TMath::Abs(v->Eta()) <= etaRegion)
	{
	  leadingJet = v;
	  foundLeadingJet = true;
	  break;
	}
    }
  if ( ! foundLeadingJet ) return kFALSE;
  theUEActivity.SetLeadingJet( *leadingJet );
  
  ///
  /// fill activity vs dphi
  ///
  for(int i=0;i<Particles.GetSize();++i)
    {
      TLorentzVector *v = (TLorentzVector*)Particles.At(i);

      double pT ( v->Pt()  );
      if ( pT                   <= ptThreshold ) continue;
      if ( TMath::Abs(v->Eta()) >  etaRegion   ) continue;

      double dphi( v->DeltaPhi( *leadingJet ) * 180./TMath::Pi() );

      unsigned int iregion(666);
      if      ( TMath::Abs(dphi) >=   0. && TMath::Abs(dphi) <   60. ) iregion = 0; /// towards
      else if ( TMath::Abs(dphi) >=  60. && TMath::Abs(dphi) <  120. ) iregion = 1; /// transverse
      else if ( TMath::Abs(dphi) >= 120. && TMath::Abs(dphi) <  180. ) iregion = 2; /// away
      else
	{
	  cout << "[UEActivityFinder] Error: dphi = " << dphi << endl;
	  return kFALSE;
	}

      _h_pTChg      [iregion]->Fill(       pT );
      _h_dN_vs_dphi [iregion]->Fill( dphi     );
      _h_dpT_vs_dphi[iregion]->Fill( dphi, pT );
    }

  ///
  /// divide histograms by eta-range to get densities
  ///
  double totalEtaRange( 2. * etaRegion );

  for ( unsigned int iregion(0); iregion<3; ++iregion )
    {
      _h_pTChg      [iregion]->Scale( 1./totalEtaRange );
      _h_dN_vs_dphi [iregion]->Scale( 1./totalEtaRange );
      _h_dpT_vs_dphi[iregion]->Scale( 1./totalEtaRange );
      
      theUEActivity.SetTH1D_pTChg      ( iregion, _h_pTChg[iregion]       );
      theUEActivity.SetTH1D_dN_vs_dphi ( iregion, _h_dN_vs_dphi[iregion]  );
      theUEActivity.SetTH1D_dpT_vs_dphi( iregion, _h_dpT_vs_dphi[iregion] );
    }

  return kTRUE;
}

///
///_______________________________________________________________________
///
UEActivityHistograms::UEActivityHistograms( const char* fileName, string *triggerNames )
{
  ///
  /// Constructor for histogram filler.
  ///

  char buffer[200];

  cout << "[UEActivityHistograms] Create file " << fileName << endl;
  file = TFile::Open( fileName, "recreate" );

  TDirectory*  dir    = file->mkdir( "UEActivity" );
  TDirectory** subdir = new TDirectory*[13];


  ///
  /// Reserve space for histograms.
  ///
  h_pTJet             = new TH1D*[13];
  h_etaJet            = new TH1D*[13];
  h_dN_vs_dphi        = new TH2D*[13];
  h_dpT_vs_dphi       = new TH2D*[13];
  h_averagePt_vs_nChg = new TH2D*[13];
  h_pTChg             = new TH2D*[39]; /// one for each region
  h_dN_vs_dpTjet      = new TH2D*[39];
  h_dpT_vs_dpTjet     = new TH2D*[39];

  
  ///
  /// 12 HLT bits :
  /// 4 Min-Bias (Pixel, Hcal, Ecal, Pixel_Trk5), Zero-Bias, 6 Jet (L115, 30, 80, 110, 140, 180)
  ///
  unsigned int iHLTbit(0);
  for ( ; iHLTbit<12; ++iHLTbit )
    {
      HLTBitNames[iHLTbit] = triggerNames[iHLTbit];

      subdir[iHLTbit] = dir->mkdir( triggerNames[iHLTbit].c_str() );
      sprintf ( buffer, "UEActivity/%s", triggerNames[iHLTbit].c_str() );
      file->cd( buffer );

      ///
      /// Initialize histograms.
      ///
      
      h_pTJet[iHLTbit]
	= new TH1D("h_pTJet",
		   "h_pTJet;p_{T}(jet) (GeV/c);",
		   300, 0., 600. );
      
      h_etaJet[iHLTbit]
	= new TH1D("h_etaJet",
		   "h_etaJet;#eta(jet);",
		   100, -2., 2. );
      
      h_dN_vs_dphi[iHLTbit]
	= new TH2D("h_dN_vs_dphi",
		   "h_dN_vs_dphi;#Delta#phi(charged, jet) (deg);< dN/d#Delta#phid#eta > (1/deg)",
		   360, -180., 180., 1000, 0., 100. );
      h_dpT_vs_dphi[iHLTbit]
	= new TH2D("h_dpT_vs_dphi",
		   "h_dpT_vs_dphi;#Delta#phi(charged, jet) (deg);< d#Sigmap_{T}/d#Delta#phid#eta > (GeV/c / deg)",
		   360, -180., 180., 1000, 0., 100. );
      h_averagePt_vs_nChg[iHLTbit]
	= new TH2D("h_averagePt_vs_nChg",
		   "h_averagePt_vs_nChg;;",
		   100, 0.5, 100.5, 200, 0., 20. );
      
      string regions[] = { "Towards", "Transverse", "Away" };
      for ( int i(0); i<3; ++i )
	{
	  subdir[iHLTbit]->mkdir( regions[i].c_str() );
	  sprintf ( buffer, "UEActivity/%s/%s", triggerNames[iHLTbit].c_str(), regions[i].c_str() );
	  file->cd( buffer );

	  sprintf ( buffer, "h_pTChg;p_{T}(charged) (GeV/c);%s 1/#sigma d#sigma/dp_{T} (1 / GeV/c)", regions[i].c_str() );
	  h_pTChg[i*13+iHLTbit]
	    = new TH2D("h_pTChg",
		       buffer,
		       1000, 0., 100., 1000, 0., 100. );
	  
	  sprintf ( buffer, "h_dN_vs_dpTjet;p_{T}(track jet) (GeV/c);%s < dN/d#phid#eta > / 2 GeV/c (1/rad)", regions[i].c_str() );
	  h_dN_vs_dpTjet[i*13+iHLTbit]
	    = new TH2D("h_dN_vs_dpTjet",
		       buffer,
		       300, 0., 600., 1000, 0., 100. );
	  
	  sprintf ( buffer, "h_dpT_vs_dpTjet;p_{T}(track jet) (GeV/c);%s < d#Sigmap_{T}/d#phid#eta > / 2 GeV/c (GeV/c / rad)", regions[i].c_str() );
	  h_dpT_vs_dpTjet[i*13+iHLTbit]
	    = new TH2D("h_dpT_vs_dpTjet",
		       buffer,
		   300, 0., 600., 1000, 0., 100. );
	}
    }


  ///
  /// Hadron level
  ///
  iHLTbit         = 12;
  subdir[iHLTbit] = dir->mkdir( "Gen" );
  file->cd( "UEActivity/Gen" );

  ///
  /// Initialize histograms.
  ///
  h_pTJet[iHLTbit]
    = new TH1D("h_pTJet",
	       "h_pTJet;p_{T}(jet) (GeV/c);",
	       300, 0., 600. );
  
  h_etaJet[iHLTbit]
    = new TH1D("h_etaJet",
	       "h_etaJet;#eta(jet);",
	       100, -2., 2. );
  
  h_dN_vs_dphi[iHLTbit]
    = new TH2D("h_dN_vs_dphi",
	       "h_dN_vs_dphi;#Delta#phi(charged, jet) (deg);< dN/d#Delta#phid#eta > (1/deg)",
	       360, -180., 180., 1000, 0., 100. );
  h_dpT_vs_dphi[iHLTbit]
    = new TH2D("h_dpT_vs_dphi",
	       "h_dpT_vs_dphi;#Delta#phi(charged, jet) (deg);< d#Sigmap_{T}/d#Delta#phid#eta > (GeV/c / deg)",
	       360, -180., 180., 1000, 0., 100. );
  h_averagePt_vs_nChg[iHLTbit]
    = new TH2D("h_averagePt_vs_nChg",
	       "h_averagePt_vs_nChg;;",
	       100, 0.5, 100.5, 200, 0., 20. );
  
  string regions[] = { "Towards", "Transverse", "Away" };
  for ( int i(0); i<3; ++i )
    {
      subdir[iHLTbit]->mkdir( regions[i].c_str() );
      sprintf ( buffer, "UEActivity/Gen/%s", regions[i].c_str() );
      file->cd( buffer );

      sprintf ( buffer, "h_pTChg;p_{T}(charged) (GeV/c);%s 1/#sigma d#sigma/dp_{T} (1 / GeV/c)", regions[i].c_str() );
      h_pTChg[i*13+iHLTbit]
	= new TH2D("h_pTChg",
		   buffer,
		   1000, 0., 100., 1000, 0., 100. );
      
      sprintf ( buffer, "h_dN_vs_dpTjet;p_{T}(track jet) (GeV/c);%s < dN/d#phid#eta > / 2 GeV/c (1/rad)", regions[i].c_str() );
      h_dN_vs_dpTjet[i*13+iHLTbit]
	= new TH2D("h_dN_vs_dpTjet",
		   buffer,
		   300, 0., 600., 1000, 0., 100. );
      
      sprintf ( buffer, "h_dpT_vs_dpTjet;p_{T}(track jet) (GeV/c);%s < d#Sigmap_{T}/d#phid#eta > / 2 GeV/c (GeV/c / rad)", regions[i].c_str() );
          h_dpT_vs_dpTjet[i*13+iHLTbit]
            = new TH2D("h_dpT_vs_dpTjet",
                       buffer,
		       300, 0., 600., 1000, 0., 100. );
    }

  
  ///
  /// save histogram parameters
  ///
  _nbinsDphi    = h_dN_vs_dphi[0]->GetNbinsX();
  _xminDphi     = h_dN_vs_dphi[0]->GetXaxis()->GetXmin();
  _xmaxDphi     = h_dN_vs_dphi[0]->GetXaxis()->GetXmax();
  _binwidthDphi = (_xmaxDphi-_xminDphi)/double(_nbinsDphi);

  _nbinsPtChg    = h_pTChg[0]->GetNbinsX();
  _xminPtChg     = h_pTChg[0]->GetXaxis()->GetXmin();
  _xmaxPtChg     = h_pTChg[0]->GetXaxis()->GetXmax();
  _binwidthPtChg = (_xmaxPtChg-_xminPtChg)/double(_nbinsPtChg);

}

///
///_______________________________________________________________________
///
void
UEActivityHistograms::fill( UEActivity& activity )
{
  ///
  /// histo filler for gen-only analysis
  /// i.e. no trigger bits are available
  ///

  unsigned int iHLTbit( 12 );

  ///
  /// pT-distribution of leading jet
  ///
  double pTjet ( activity.GetLeadingJet().Pt()  );
  double etajet( activity.GetLeadingJet().Eta() );
  h_pTJet [iHLTbit]->Fill( pTjet  );
  h_etaJet[iHLTbit]->Fill( etajet );


  ///
  /// distributions vs dphi(charged, jet)
  ///
  for ( int i(1); i<=_nbinsDphi; ++i )
    {
      double x( _xminDphi + (i-1)*_binwidthDphi + _binwidthDphi/2. );
      
      unsigned int iregion(666);
      if      ( TMath::Abs(x) >=   0. && TMath::Abs(x) <   60. ) iregion = 0; /// towards
      else if ( TMath::Abs(x) >=  60. && TMath::Abs(x) <  120. ) iregion = 1; /// transverse
      else if ( TMath::Abs(x) >= 120. && TMath::Abs(x) <  180. ) iregion = 2; /// away
      else
        {
          cout << "[UEActivityHistograms] Error: x = " << x << endl;
          return;
        }

      TH1D* TH1D_dN_vs_dphi  = activity.GetTH1D_dN_vs_dphi(iregion);
      TH1D* TH1D_dpT_vs_dphi = activity.GetTH1D_dpT_vs_dphi(iregion);
      
      /// divide y-entry by binwidth
      /// to get density per degree
      double y(666.);
      y = TH1D_dN_vs_dphi->GetBinContent(i)/_binwidthDphi;
      h_dN_vs_dphi[iHLTbit]->Fill( x, y );
      
      y = TH1D_dpT_vs_dphi->GetBinContent(i)/_binwidthDphi;
      h_dpT_vs_dphi[iHLTbit]->Fill( x, y );
    }
      

  ///
  /// loop on regions
  ///
  double nChg ( 0. );
  double pTsum( 0. );
  for ( unsigned int iregion(0); iregion<3; ++iregion )
    {
      ///
      /// sum up overall activities per unit phi/rad and eta
      ///
      nChg  += activity.GetNumParticles(iregion) /(2*TMath::Pi()/3.);
      pTsum += activity.GetParticlePtSum(iregion)/(2*TMath::Pi()/3.);


      ///
      /// densities vs pT(jet) in different regions
      ///
      /// ( divide activity per eta by phi-range (pi/3)
      ///   to get density per eta and phi in radian ) 
      ///
      h_dN_vs_dpTjet [iregion*13+iHLTbit]->Fill( pTjet, activity.GetNumParticles(iregion) /(2*TMath::Pi()/3.) );
      h_dpT_vs_dpTjet[iregion*13+iHLTbit]->Fill( pTjet, activity.GetParticlePtSum(iregion)/(2*TMath::Pi()/3.) );


      ///
      /// pT-distributions of particles/tracks in different regions
      ///
      TH1D* TH1D_pTChg = activity.GetTH1D_pTChg(iregion);
      for ( int i(1); i<=_nbinsPtChg; ++i )
	{
	  double x( _xminPtChg + (i-1)*_binwidthPtChg + _binwidthPtChg/2. );
	  double y( TH1D_pTChg->GetBinContent(i) / _binwidthPtChg         );

	  h_pTChg[iregion*13+iHLTbit]->Fill( x, y );
	}

    }

  h_averagePt_vs_nChg[iHLTbit]->Fill( nChg, pTsum/nChg );

}

///
///_______________________________________________________________________
///
void
UEActivityHistograms::fill( UEActivity& activity, TClonesArray& acceptedTriggers )
{
  ///
  /// Histo filler for reco-only analysis
  /// HL trigger bits *are* available
  ///

  ///
  /// 11 HLT bits :
  /// 4 Min-Bias (Pixel, Hcal, Ecal, general), Zero-Bias, 6 Jet (30, 50, 80, 110, 180, 250)
  ///
  unsigned int iHLTbit(0);
  for ( ; iHLTbit<12; ++iHLTbit )
    {
      ///
      /// ask if trigger was accepted
      ///
      bool hltAccept( false );
      unsigned int nAcceptedTriggers( acceptedTriggers.GetSize() );
      for ( unsigned int itrig(0); itrig<nAcceptedTriggers; ++itrig )
	{
	  std::string filterName( acceptedTriggers.At(itrig)->GetName() );      
	  if ( filterName == HLTBitNames[iHLTbit] ) hltAccept = true;
	}
      if ( ! hltAccept ) continue;

      ///
      /// pT-distribution of leading jet
      ///
      double pTjet ( activity.GetLeadingJet().Pt()  );
      double etajet( activity.GetLeadingJet().Eta() );
      h_pTJet [iHLTbit]->Fill( pTjet  );
      h_etaJet[iHLTbit]->Fill( etajet );


      ///
      /// distributions vs dphi(charged, jet)
      ///
      for ( int i(1); i<=_nbinsDphi; ++i )
	{
	  double x( _xminDphi + (i-1)*_binwidthDphi + _binwidthDphi/2. );
	  
	  unsigned int iregion(666);
	  if      ( TMath::Abs(x) >=   0. && TMath::Abs(x) <   60. ) iregion = 0; /// towards
	  else if ( TMath::Abs(x) >=  60. && TMath::Abs(x) <  120. ) iregion = 1; /// transverse
	  else if ( TMath::Abs(x) >= 120. && TMath::Abs(x) <  180. ) iregion = 2; /// away
	  else
	    {
	      cout << "[UEActivityHistograms] Error: x = " << x << endl;
	      return;
	    }
	  
	  TH1D* TH1D_dN_vs_dphi  = activity.GetTH1D_dN_vs_dphi(iregion);
	  TH1D* TH1D_dpT_vs_dphi = activity.GetTH1D_dpT_vs_dphi(iregion);
	  
	  /// divide y-entry by binwidth
	  /// to get density per degree
	  double y(666.);
	  y = TH1D_dN_vs_dphi->GetBinContent(i)/_binwidthDphi;
	  h_dN_vs_dphi[iHLTbit]->Fill( x, y );
	  
	  y = TH1D_dpT_vs_dphi->GetBinContent(i)/_binwidthDphi;
	  h_dpT_vs_dphi[iHLTbit]->Fill( x, y );
	}
      
      
      ///
      /// loop on regions
      ///
      double nChg ( 0. );
      double pTsum( 0. );
      for ( unsigned int iregion(0); iregion<3; ++iregion )
	{
	  ///
	  /// sum up overall activities per unit phi/rad and eta
	  ///
	  nChg  += activity.GetNumParticles(iregion) /(2*TMath::Pi()/3.);
	  pTsum += activity.GetParticlePtSum(iregion)/(2*TMath::Pi()/3.);
	  
	  
	  ///
	  /// densities vs pT(jet) in different regions
	  ///
	  /// ( divide activity per eta by phi-range (pi/3)
	  ///   to get density per eta and phi in radian ) 
	  ///
	  h_dN_vs_dpTjet [iregion*13+iHLTbit]->Fill( pTjet, activity.GetNumParticles(iregion) /(2*TMath::Pi()/3.) );
	  h_dpT_vs_dpTjet[iregion*13+iHLTbit]->Fill( pTjet, activity.GetParticlePtSum(iregion)/(2*TMath::Pi()/3.) );
	  
	  
	  ///
	  /// pT-distributions of particles/tracks in different regions
	  ///
	  TH1D* TH1D_pTChg = activity.GetTH1D_pTChg(iregion);
	  for ( int i(1); i<=_nbinsPtChg; ++i )
	    {
	      double x( _xminPtChg + (i-1)*_binwidthPtChg + _binwidthPtChg/2. );
	      double y( TH1D_pTChg->GetBinContent(i) / _binwidthPtChg         );
	      
	      h_pTChg[iregion*13+iHLTbit]->Fill( x, y );
	    }
	  
	}

      h_averagePt_vs_nChg[iHLTbit]->Fill( nChg, pTsum/nChg );
    }
}

///
///_______________________________________________________________________
///


