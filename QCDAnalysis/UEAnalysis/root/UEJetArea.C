#include "UEJetArea.h"
#include <vector>
#include <math.h>


///
///___________________________________________________________________
///
UEJetWithArea::UEJetWithArea()
{
  //  cout << "UEJetWithArea::UEJetWithArea()" << endl;

  _momentum           = new TLorentzVector();
  _area               = 0.;
  _nconstituents      = 0;
}

UEJetWithArea::UEJetWithArea( TLorentzVector& theMomentum, double theArea, unsigned int theNconstituents)
{
  //  cout << "UEJetWithArea::UEJetWithArea( TLorentzVector& theMomentum, double theArea, unsigned int theNconstituents)" << endl;

  _momentum           = new TLorentzVector( theMomentum );
  _area               = theArea;
  _nconstituents      = theNconstituents;
}

///
///________________________________________________________________________________
///
UEJetAreaFinder::UEJetAreaFinder( float _etaRegion, float _ptThreshold, string algorithm )
{
  //  cout << "UEJetAreaFinder::UEJetAreaFinder( float _etaRegion, float _ptThreshold, string algorithm )" << endl;


  ///
  /// input selection
  ///
  etaRegionInput = 2.5;

  ///
  /// jet selection
  ///
  etaRegion   = _etaRegion;
  ptThreshold = _ptThreshold/1000.;

  if ( algorithm == "SISCone" )
    {
      /// SISCone 0.5
      coneRadius           = 0.5;
      coneOverlapThreshold = 0.75;
      maxPasses            = 0;
      protojetPtMin        = 0.;
      caching              = false;
      scale                = fastjet::SISConePlugin::SM_pttilde;
      mPlugin              = new fastjet::SISConePlugin( coneRadius,
							 coneOverlapThreshold,
							 maxPasses,
							 protojetPtMin,
							 caching,
							 scale );
      mJetDefinition       = new fastjet::JetDefinition (mPlugin);
    }
  else if ( algorithm == "kT" )
    {
      /// kT 0.4
      rParam         = 0.4;
      fjStrategy     = fastjet::Best;
      mJetDefinition = new fastjet::JetDefinition (fastjet::kt_algorithm, rParam, fjStrategy);

    }
  else if ( algorithm == "AntiKt" )
    {
      /// Anti-kT 0.4
      rParam         = 0.4;
      fjStrategy     = fastjet::Best;
      mJetDefinition = new fastjet::JetDefinition (fastjet::antikt_algorithm, rParam, fjStrategy);
    }
  else
    {
      cout << "Sorry! " << algorithm << " not set up." << endl;
    }

  /// jet areas
  //   ghostEtaMax       = 6.;
  ghostEtaMax       = 3.0;
  activeAreaRepeats = 5;
  ghostArea         = 0.01;
  //mActiveArea       = new fastjet::ActiveAreaSpec (ghostEtaMax, activeAreaRepeats, ghostArea);
  theGhostedAreaSpec = new fastjet::GhostedAreaSpec( ghostEtaMax, activeAreaRepeats, ghostArea );
  theAreaDefinition  = new fastjet::AreaDefinition( fastjet::active_area, *theGhostedAreaSpec );
}

///
///________________________________________________________________________________
///
Bool_t
UEJetAreaFinder::find( TClonesArray& Input, vector<UEJetWithArea>& _jets )
{
  /// return if no four-vectors are provided
  if ( Input.GetSize() == 0 ) return kFALSE;

  /// prepare input
  std::vector<fastjet::PseudoJet> fjInputs;
  fjInputs.reserve ( Input.GetSize() );

  int iJet( 0 );
  for( int i(0); i < Input.GetSize(); ++i )
    {
      TLorentzVector *v = (TLorentzVector*)Input.At(i);

      if ( TMath::Abs(v->Eta()) > etaRegionInput ) continue;
      if ( v->Pt()              < ptThreshold    ) continue;

      fjInputs.push_back (fastjet::PseudoJet (v->Px(), v->Py(), v->Pz(), v->E()) );
      fjInputs.back().set_user_index(iJet);
      ++iJet;
    }

  /// return if no four-vectors in visible phase space
  if ( fjInputs.size() == 0 ) return kFALSE;
  
  /// print out info on current jet algorithm
  //   cout << endl;
  //   cout << mJetDefinition->description() << endl;
  //   cout << theAreaDefinition->description() << endl;

  /// return if active area is not chosen to be calculated
  if ( ! theAreaDefinition ) return kFALSE;

  //  cout << "fastjet::ClusterSequenceActiveArea* clusterSequence" << endl;
  
  fastjet::ClusterSequenceArea* clusterSequence
    = new fastjet::ClusterSequenceArea (fjInputs, *mJetDefinition, *theAreaDefinition );

  //  cout << "retrieve jets for selected mode" << endl;

  /// retrieve jets for selected mode
  double mJetPtMin( 1. );
  std::vector<fastjet::PseudoJet> jets( clusterSequence->inclusive_jets (mJetPtMin) );
  unsigned int nJets( jets.size() );

  if ( nJets == 0 ) 
    {
      delete clusterSequence;
      return kFALSE;
    }
  //Double_t ptByArea[ nJets ];

  //   int columnwidth( 10 );
  //cout << "found " << jets.size() << " jets" << endl;
  //   cout.width( 5 );
  //   cout << "jet";
  //   cout.width( columnwidth );
  //   cout << "eta";
  //   cout.width( columnwidth );
  //   cout << "phi";
  //   cout.width( columnwidth );
  //   cout << "pT";
  //   cout.width( columnwidth );
  //   cout << "jetArea";
  //   cout.width( 15 );
  //   cout << "pT / jetArea";
  //   cout << endl;
  
  _jets.reserve( nJets );

  vector< fastjet::PseudoJet > sorted_jets ( sorted_by_pt( jets ));
  for ( int i(0); i<nJets; ++i )
    {
      //ptByArea[i] = jets[i].perp()/clusterSequence->area(jets[i]);

      //       cout.width( 5 );
      //       cout << i;
      //       cout.width( columnwidth );
      //       cout << jets[i].eta();
      //       cout.width( columnwidth );
      //       cout << jets[i].phi();
      //       cout.width( columnwidth );
      //       cout << jets[i].perp();
      //       cout.width( columnwidth );
      //       cout << clusterSequence->area(jets[i]);
      //       cout.width( 15 );
      //       cout << ptByArea[i];
      //       cout << endl;

      /// save
      ///
      /// TLorentzVector
      /// area
      /// nconstituents
      
      fastjet::PseudoJet jet( sorted_jets[i] );
      vector< fastjet::PseudoJet > constituents( clusterSequence->constituents(jet) );
      
      TLorentzVector* mom    = new TLorentzVector( jet.px(), jet.py(), jet.pz(), jet.e() );
      double          area   = clusterSequence->area(jet);
      //  double          median = TMath::Median( nJets, ptByArea );
      unsigned int    nconst = constituents.size();
      
      UEJetWithArea* theJet = new UEJetWithArea( *mom, area, nconst);
      //_jets[i] = *theJet;
      _jets.push_back( *theJet );

      delete mom;
      delete theJet;
    }
  delete clusterSequence;

  return kTRUE;
}


///
///________________________________________________________________________________
///
UEJetAreaHistograms::UEJetAreaHistograms( const char* fileName, string *triggerNames )
{
  //  cout << "UEJetAreaHistograms::UEJetAreaHistograms( const char* fileName, string *triggerNames )" << endl;

  ///
  /// Constructor for histogram filler.
  ///

  char buffer[200];

  cout << "[UEJetAreaHistograms] Create file " << fileName << endl;
  file = TFile::Open( fileName, "recreate" );

  TDirectory*  dir = file->mkdir( "UEJetArea" );
  subdir           = new TDirectory*[12];

  ///
  /// Reserve space for histograms.
  ///
  h_pTAllJets            = new TH1D*[12]; // all jets
  h_etaAllJets           = new TH1D*[12];
  h_areaAllJets          = new TH1D*[12];
  h_ptByAreaAllJets      = new TH1D*[12];
  h_nConstituentsAllJets = new TH1D*[12];
  h2d_pTAllJets_vs_pTjet            = new TH2D*[12];
  h2d_areaAllJets_vs_pTjet          = new TH2D*[12];
  h2d_ptByAreaAllJets_vs_pTjet      = new TH2D*[12];
  h2d_nConstituentsAllJets_vs_pTjet = new TH2D*[12];

  h_pTJet                = new TH1D*[12]; // leading jet
  h_etaJet               = new TH1D*[12];
  h_areaJet              = new TH1D*[12];
  h_ptByAreaJet          = new TH1D*[12];
  h_nConstituentsJet     = new TH1D*[12];
  h2d_areaJet_vs_pTjet          = new TH2D*[12];
  h2d_ptByAreaJet_vs_pTjet      = new TH2D*[12];
  h2d_nConstituentsJet_vs_pTjet = new TH2D*[12];

  h_medianPt        = new TH1D*[12]; // event-by-event medians
  h_medianArea      = new TH1D*[12];
  h_medianPtByArea = new TH1D*[12];
  h2d_medianPt_vs_pTjet       = new TH2D*[12];
  h2d_medianArea_vs_pTjet     = new TH2D*[12];
  h2d_medianPtByArea_vs_pTjet = new TH2D*[12];

  ///
  /// 11 HLT bits :
  /// 4 Min-Bias (Pixel, Hcal, Ecal, general), Zero-Bias, 6 Jet (30, 50, 80, 110, 180, 250)
  ///
  unsigned int iHLTbit(0);
  for ( ; iHLTbit<11; ++iHLTbit )
    {
      HLTBitNames[iHLTbit] = triggerNames[iHLTbit];

      subdir[iHLTbit] = dir->mkdir( triggerNames[iHLTbit].c_str() );
      sprintf ( buffer, "UEJetArea/%s", triggerNames[iHLTbit].c_str() );
      file->cd( buffer );

      ///
      /// Initialize histograms.
      ///

      h_pTJet    [iHLTbit] = new TH1D("h_pTJet"    , "h_pTJet;p_{T}(leading jet) (GeV/c);", 300, 0., 600. );
      h_pTAllJets[iHLTbit] = new TH1D("h_pTAllJets", "h_pTAllJets;p_{T}(jet) (GeV/c);"    , 300, 0., 600. );

      h_etaJet    [iHLTbit] = new TH1D("h_etaJet"    , "h_etaJet;#eta(leading jet);", 100, -2., 2. );
      h_etaAllJets[iHLTbit] = new TH1D("h_etaAllJets", "h_etaAllJets;#eta(jet);"    , 100, -2., 2. );

      h_areaJet    [iHLTbit] = new TH1D("h_areaJet"    , "h_areaJet;Leading jet area A;", 100, 0., 2. );
      h_areaAllJets[iHLTbit] = new TH1D("h_areaAllJets", "h_areaAllJets;Jet area A;"    , 100, 0., 2. );

      h_ptByAreaJet    [iHLTbit] = new TH1D("h_ptByAreaJet"    , "h_ptByAreaJet;Leading jet p_{T}/A;", 1000, 0., 200. );
      h_ptByAreaAllJets[iHLTbit] = new TH1D("h_ptByAreaAllJets", "h_ptByAreaAllJets;Jet p_{T}/A;"    , 1000, 0., 200. );

      h_nConstituentsJet    [iHLTbit] = new TH1D("h_nConstituentsJet"    , 
						 "h_nConstituentsJet;N(leading jet constituents);", 50, 0.5, 50.5 );
      h_nConstituentsAllJets[iHLTbit] = new TH1D("h_nConstituentsAllJets", 
						 "h_nConstituentsAllJets;N(jet constituents);"    , 50, 0.5, 50.5 );

      h2d_pTAllJets_vs_pTjet[iHLTbit] = new TH2D( "h2d_pTAllJets_vs_pTjet",
						  "h2d_pTAllJets_vs_pTjet;p_{T}(leading jet) (GeV/c);p_{T}(jet) (GeV/c)",
						  300, 0., 600., 300, 0., 600. );

      h2d_areaJet_vs_pTjet    [iHLTbit]	
	= new TH2D("h2d_areaJet_vs_pTjet",
		   "h2d_areaJet_vs_pTjet;p_{T}(leading jet) (GeV/c);Leading Jet area A /rad", 300, 0., 600., 100, 0., 2. );
      h2d_areaAllJets_vs_pTjet[iHLTbit]	
	= new TH2D("h2d_areaAllJets_vs_pTjet",
		   "h2d_areaAllJets_vs_pTjet;p_{T}(leading jet) (GeV/c);Jet area A /rad", 300, 0., 600., 100, 0., 2. );

      h2d_ptByAreaJet_vs_pTjet    [iHLTbit] = 
	new TH2D("h2d_ptByAreaJet_vs_pTjet",
		 "h2d_ptByAreaJet_vs_pTjet;p_{T}(leading jet) (GeV/c);Leading Jet p_{T}/A (GeV/c rad^{-1})", 300, 0., 600., 1000, 0., 200. );
      h2d_ptByAreaAllJets_vs_pTjet[iHLTbit]	
	= new TH2D("h2d_ptByAreaAllJets_vs_pTjet",
		   "h2d_ptByAreaAllJets_vs_pTjet;p_{T}(leading jet) (GeV/c);Jet area p_{T}/A (GeV/c rad^{-1})", 300, 0., 600., 1000, 0., 200. );

      h2d_nConstituentsJet_vs_pTjet[iHLTbit]
	= new TH2D("h2d_nConstituentsJet_vs_pTjet",
		   "h2d_nConstituentsJet_vs_pTjet;p_{T}(leading jet) (GeV/c);N(leading jet constituents)",
		   300, 0., 600., 50, 0.5, 50.5 );
      h2d_nConstituentsAllJets_vs_pTjet[iHLTbit]
	= new TH2D("h2d_nConstituentsAllJets_vs_pTjet",
		   "h2d_nConstituentsAllJets_vs_pTjet;p_{T}(leading jet) (GeV/c);N(jet constituents)",
		   300, 0., 600., 50, 0.5, 50.5 );
      
      h_medianPt      [iHLTbit] = new TH1D("h_medianPt"  , "h_medianPt;#mu_{1/2}({p_{Ti}}) (GeV/c);", 300, 0., 600. );
      h_medianArea    [iHLTbit] = new TH1D("h_medianArea", "h_medianArea;#mu_{1/2}({A_{Ti}}) (rad);", 100, 0., 2.   );
      h_medianPtByArea[iHLTbit] = new TH1D("h_medianPtByArea",
					   "h_medianPtByArea;#mu_{1/2}({p_{Ti}/A_{i}}) (GeV/c rad^{-1});",
					   1000, 0., 200. );

      h2d_medianPt_vs_pTjet[iHLTbit] = 
	new TH2D("h2d_medianPt_vs_pTjet", 
		 "h2d_medianPt_vs_pTjet;p_{T}(leading jet) (GeV/c);#mu_{1/2}({p_{Ti}}) (GeV/c)", 300, 0., 600., 300, 0., 600. );
      h2d_medianArea_vs_pTjet[iHLTbit] = 
	new TH2D("h2d_medianArea_vs_pTjet", 
		 "h2d_medianArea_vs_pTjet;p_{T}(leading jet) (GeV/c);#mu_{1/2}({A_{Ti}}) (rad)", 300, 0., 600., 100, 0., 2. );
      h2d_medianPtByArea_vs_pTjet[iHLTbit]
	= new TH2D("h2d_medianPtByArea_vs_pTjet",
		   "h2d_medianPtByArea_vs_pTjet;p_{T}(leading jet) (GeV/c);#mu_{1/2}({p_{Ti}/A_{i}}) (GeV/c rad^{-1})",
		   300, 0., 600., 100, 0., 20. );

    }

  ///
  /// Hadron level
  ///
  iHLTbit         = 11;
  subdir[iHLTbit] = dir->mkdir( "Gen" );
  file->cd( "UEJetArea/Gen" );


  h_pTJet    [iHLTbit] = new TH1D("h_pTJet"    , "h_pTJet;p_{T}(leading jet) (GeV/c);", 300, 0., 600. );
  h_pTAllJets[iHLTbit] = new TH1D("h_pTAllJets", "h_pTAllJets;p_{T}(jet) (GeV/c);"    , 300, 0., 600. );
  
  h_etaJet    [iHLTbit] = new TH1D("h_etaJet"    , "h_etaJet;#eta(leading jet);", 100, -2., 2. );
  h_etaAllJets[iHLTbit] = new TH1D("h_etaAllJets", "h_etaAllJets;#eta(jet);"    , 100, -2., 2. );
  
  h_areaJet    [iHLTbit] = new TH1D("h_areaJet"    , "h_areaJet;Leading jet area A;", 100, 0., 2. );
  h_areaAllJets[iHLTbit] = new TH1D("h_areaAllJets", "h_areaAllJets;Jet area A;"    , 100, 0., 2. );

  h_ptByAreaJet    [iHLTbit] = new TH1D("h_ptByAreaJet"    , "h_ptByAreaJet;Leading jet p_{T}/A;", 1000, 0., 200. );
  h_ptByAreaAllJets[iHLTbit] = new TH1D("h_ptByAreaAllJets", "h_ptByAreaAllJets;Jet p_{T}/A;"    , 1000, 0., 200. );

  h_nConstituentsJet    [iHLTbit] = new TH1D("h_nConstituentsJet"    ,
					     "h_nConstituentsJet;N(leading jet constituents);", 50, 0.5, 50.5 );
  h_nConstituentsAllJets[iHLTbit] = new TH1D("h_nConstituentsAllJets",
					     "h_nConstituentsAllJets;N(jet constituents);"    , 50, 0.5, 50.5 );

  h2d_pTAllJets_vs_pTjet[iHLTbit] = new TH2D( "h2d_pTAllJets_vs_pTjet",
					      "h2d_pTAllJets_vs_pTjet;p_{T}(leading jet) (GeV/c);p_{T}(jet) (GeV/c)",
					      300, 0., 600., 300, 0., 600. );

  h2d_areaJet_vs_pTjet    [iHLTbit]
    = new TH2D("h2d_areaJet_vs_pTjet",
	       "h2d_areaJet_vs_pTjet;p_{T}(leading jet) (GeV/c);Leading Jet area A /rad", 300, 0., 600., 100, 0., 2. );
  h2d_areaAllJets_vs_pTjet[iHLTbit]
    = new TH2D("h2d_areaAllJets_vs_pTjet",
	       "h2d_areaAllJets_vs_pTjet;p_{T}(leading jet) (GeV/c);Jet area A /rad", 300, 0., 600., 100, 0., 2. );
  
  h2d_ptByAreaJet_vs_pTjet    [iHLTbit] =
    new TH2D("h2d_ptByAreaJet_vs_pTjet",
	     "h2d_ptByAreaJet_vs_pTjet;p_{T}(leading jet) (GeV/c);Leading Jet p_{T}/A (GeV/c rad^{-1})", 300, 0., 600., 1000, 0., 200. );
  h2d_ptByAreaAllJets_vs_pTjet[iHLTbit]
    = new TH2D("h2d_ptByAreaAllJets_vs_pTjet",
	       "h2d_ptByAreaAllJets_vs_pTjet;p_{T}(leading jet) (GeV/c);Jet area p_{T}/A (GeV/c rad^{-1})", 300, 0., 600., 1000, 0., 200. );
  
  h2d_nConstituentsJet_vs_pTjet[iHLTbit]
    = new TH2D("h2d_nConstituentsJet_vs_pTjet",
	       "h2d_nConstituentsJet_vs_pTjet;p_{T}(leading jet) (GeV/c);N(leading jet constituents)",
	       300, 0., 600., 50, 0.5, 50.5 );
  h2d_nConstituentsAllJets_vs_pTjet[iHLTbit]
    = new TH2D("h2d_nConstituentsAllJets_vs_pTjet",
	       "h2d_nConstituentsAllJets_vs_pTjet;p_{T}(leading jet) (GeV/c);N(jet constituents)",
	       300, 0., 600., 50, 0.5, 50.5 );
  
  h_medianPt      [iHLTbit] = new TH1D("h_medianPt"  , "h_medianPt;#mu_{1/2}({p_{Ti}}) (GeV/c);", 300, 0., 600. );
  h_medianArea    [iHLTbit] = new TH1D("h_medianArea", "h_medianArea;#mu_{1/2}({A_{Ti}}) (rad);", 100, 0., 2.   );
  h_medianPtByArea[iHLTbit] = new TH1D("h_medianPtByArea",
				       "h_medianPtByArea;#mu_{1/2}({p_{Ti}/A_{i}}) (GeV/c rad^{-1});",
				       1000, 0., 200. );
  
  h2d_medianPt_vs_pTjet[iHLTbit] =
    new TH2D("h2d_medianPt_vs_pTjet",
	     "h2d_medianPt_vs_pTjet;p_{T}(leading jet) (GeV/c);#mu_{1/2}({p_{Ti}}) (GeV/c)", 300, 0., 600., 300, 0., 600. );
  h2d_medianArea_vs_pTjet[iHLTbit] =
    new TH2D("h2d_medianArea_vs_pTjet",
	     "h2d_medianArea_vs_pTjet;p_{T}(leading jet) (GeV/c);#mu_{1/2}({A_{Ti}}) (rad)", 300, 0., 600., 100, 0., 2. );
  h2d_medianPtByArea_vs_pTjet[iHLTbit]
    = new TH2D("h2d_medianPtByArea_vs_pTjet",
	       "h2d_medianPtByArea_vs_pTjet;p_{T}(leading jet) (GeV/c);#mu_{1/2}({p_{Ti}/A_{i}}) (GeV/c rad^{-1})",
	       300, 0., 600., 100, 0., 20. );
  
}

///
///________________________________________________________________________________
///
void
UEJetAreaHistograms::fill( vector<UEJetWithArea>& theJets )
{
  //  cout << "UEJetAreaHistograms::fill( vector<UEJetWithArea>& theJets )" << endl;

  ///
  /// histo filler for gen-only analysis
  /// i.e. no trigger bits are available
  ///

  unsigned int iHLTbit( 11 );
  Double_t ptarray  [ theJets.size() ];
  Double_t areaarray[ theJets.size() ];
  Double_t ptByArea [ theJets.size() ];

  for ( unsigned int ijet(0); ijet < theJets.size(); ++ijet )
    {
      double pTLeadingJet       ( theJets[0].GetMomentum()->Pt() );

      double pTJet  ( theJets[ijet].GetMomentum()->Pt()  );
      double etaJet ( theJets[ijet].GetMomentum()->Eta() );
      double areaJet( theJets[ijet].GetArea()            );

      ptarray  [ijet] = pTJet;
      areaarray[ijet] = areaJet;
      ptByArea [ijet] = pTJet / areaJet;

      unsigned int nConstituents( theJets[ijet].GetNConstituents() );
      
      h_pTAllJets           [iHLTbit]->Fill( pTJet          );
      h_etaAllJets          [iHLTbit]->Fill( etaJet         );
      h_areaAllJets         [iHLTbit]->Fill( areaJet        );
      h_ptByAreaAllJets     [iHLTbit]->Fill( ptByArea[ijet] );
      h_nConstituentsAllJets[iHLTbit]->Fill( nConstituents  );
      
      h2d_pTAllJets_vs_pTjet           [iHLTbit]->Fill( pTLeadingJet, pTJet          );
      h2d_areaAllJets_vs_pTjet         [iHLTbit]->Fill( pTLeadingJet, areaJet        );
      h2d_ptByAreaAllJets_vs_pTjet     [iHLTbit]->Fill( pTLeadingJet, ptByArea[ijet] );
      h2d_nConstituentsAllJets_vs_pTjet[iHLTbit]->Fill( pTLeadingJet, nConstituents  );

      ///
      /// histograms for leading jet
      ///
      if ( ijet == 0 )
	{
	  h_pTJet           [iHLTbit]->Fill( pTJet           );
	  h_etaJet          [iHLTbit]->Fill( etaJet          );
	  h_areaJet         [iHLTbit]->Fill( areaJet         );
	  h_ptByAreaJet     [iHLTbit]->Fill( ptByArea[ijet]  );
	  h_nConstituentsJet[iHLTbit]->Fill( nConstituents   );
	  
	  h2d_areaJet_vs_pTjet         [iHLTbit]->Fill( pTJet, areaJet         );
	  h2d_ptByAreaJet_vs_pTjet     [iHLTbit]->Fill( pTJet, ptByArea[ijet]  );
	  h2d_nConstituentsJet_vs_pTjet[iHLTbit]->Fill( pTJet, nConstituents   );
	}
    }

  double medianPt       ( TMath::Median( theJets.size(), areaarray ) );
  double medianArea     ( TMath::Median( theJets.size(), ptarray   ) );
  double medianPtPerArea( TMath::Median( theJets.size(), ptByArea  ) );

  h_medianPt      [iHLTbit]->Fill( medianArea      );
  h_medianArea    [iHLTbit]->Fill( medianPt        );
  h_medianPtByArea[iHLTbit]->Fill( medianPtPerArea );

  h2d_medianPt_vs_pTjet      [iHLTbit]->Fill( ptarray[0], medianPt        );
  h2d_medianArea_vs_pTjet    [iHLTbit]->Fill( ptarray[0], medianArea      );
  h2d_medianPtByArea_vs_pTjet[iHLTbit]->Fill( ptarray[0], medianPtPerArea );
}

void 
UEJetAreaHistograms::fill( vector<UEJetWithArea>& theJets, TClonesArray& acceptedTriggers )
{
  //  cout << "UEJetAreaHistograms::fill( vector<UEJetWithArea>& theJets, TClonesArray& acceptedTriggers )" << endl;

  ///
  /// Histo filler for reco-only analysis
  /// HL trigger bits *are* available
  ///
  
  ///
  /// 11 HLT bits :
  /// 4 Min-Bias (Pixel, Hcal, Ecal, general), Zero-Bias, 6 Jet (30, 50, 80, 110, 180, 250)
  ///
  unsigned int iHLTbit(0);
  for ( ; iHLTbit<11; ++iHLTbit )
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
      
      Double_t ptarray  [ theJets.size() ];
      Double_t areaarray[ theJets.size() ];
      Double_t ptByArea [ theJets.size() ];
      
      for ( unsigned int ijet(0); ijet < theJets.size(); ++ijet )
	{
	  double pTLeadingJet       ( theJets[0].GetMomentum()->Pt() );
	  
	  double pTJet  ( theJets[ijet].GetMomentum()->Pt()  );
	  double etaJet ( theJets[ijet].GetMomentum()->Eta() );
	  double areaJet( theJets[ijet].GetArea()            );
	  
	  ptarray  [ijet] = pTJet;
	  areaarray[ijet] = areaJet;
	  ptByArea [ijet] = pTJet / areaJet;
	  
	  unsigned int nConstituents( theJets[ijet].GetNConstituents() );
	  
	  h_pTAllJets           [iHLTbit]->Fill( pTJet          );
	  h_etaAllJets          [iHLTbit]->Fill( etaJet         );
	  h_areaAllJets         [iHLTbit]->Fill( areaJet        );
	  h_ptByAreaAllJets     [iHLTbit]->Fill( ptByArea[ijet] );
	  h_nConstituentsAllJets[iHLTbit]->Fill( nConstituents  );

	  h2d_pTAllJets_vs_pTjet           [iHLTbit]->Fill( pTLeadingJet, pTJet          );
	  h2d_areaAllJets_vs_pTjet         [iHLTbit]->Fill( pTLeadingJet, areaJet        );
	  h2d_ptByAreaAllJets_vs_pTjet     [iHLTbit]->Fill( pTLeadingJet, ptByArea[ijet] );
	  h2d_nConstituentsAllJets_vs_pTjet[iHLTbit]->Fill( pTLeadingJet, nConstituents  );

	  ///
	  /// histograms for leading jet
	  ///
	  if ( ijet == 0 )
	    {
	      h_pTJet           [iHLTbit]->Fill( pTJet           );
	      h_etaJet          [iHLTbit]->Fill( etaJet          );
	      h_areaJet         [iHLTbit]->Fill( areaJet         );
	      h_ptByAreaJet     [iHLTbit]->Fill( ptByArea[ijet]  );
	      h_nConstituentsJet[iHLTbit]->Fill( nConstituents   );

	      h2d_areaJet_vs_pTjet         [iHLTbit]->Fill( pTJet, areaJet         );
	      h2d_ptByAreaJet_vs_pTjet     [iHLTbit]->Fill( pTJet, ptByArea[ijet]  );
	      h2d_nConstituentsJet_vs_pTjet[iHLTbit]->Fill( pTJet, nConstituents   );
	    }
	}

      double medianPt       ( TMath::Median( theJets.size(), areaarray ) );
      double medianArea     ( TMath::Median( theJets.size(), ptarray   ) );
      double medianPtPerArea( TMath::Median( theJets.size(), ptByArea  ) );
      
      h_medianPt      [iHLTbit]->Fill( medianArea      );
      h_medianArea    [iHLTbit]->Fill( medianPt        );
      h_medianPtByArea[iHLTbit]->Fill( medianPtPerArea );
      
      h2d_medianPt_vs_pTjet      [iHLTbit]->Fill( ptarray[0], medianPt        );
      h2d_medianArea_vs_pTjet    [iHLTbit]->Fill( ptarray[0], medianArea      );
      h2d_medianPtByArea_vs_pTjet[iHLTbit]->Fill( ptarray[0], medianPtPerArea );
    }
}
