// Authors: F. Ambroglini, L. Fano', F. Bechtel
#include <QCDAnalysis/UEAnalysis/interface/UEJetValidation.h>
 
using namespace edm;
using namespace std;
using namespace reco;

typedef vector<string> vstring;

UEJetValidation::UEJetValidation( const ParameterSet& pset )
{
  ChgGenJetsInputTag = pset.getParameter<InputTag>( "ChgGenJetCollectionName" );
  TrackJetsInputTag  = pset.getParameter<InputTag>( "TracksJetCollectionName" );
  CaloJetsInputTag   = pset.getParameter<InputTag>( "CaloJetCollectionName"   );
  genEventScaleTag   = pset.getParameter<InputTag>( "genEventScale"           );

  // trigger results
  triggerResultsTag = pset.getParameter<InputTag>("triggerResults");
  triggerEventTag   = pset.getParameter<InputTag>("triggerEvent");
  //   hltFilterTag      = pset.getParameter<InputTag>("hltFilter");
  //   triggerName       = pset.getParameter<InputTag>("triggerName");

  _eventScaleMin = pset.getParameter<double>("eventScaleMin");
  _eventScaleMax = pset.getParameter<double>("eventScaleMax");
  _PTTHRESHOLD   = pset.getParameter<double>("pTThreshold");
  _ETALIMIT      = pset.getParameter<double>("etaLimit");
  _dR            = pset.getParameter<double>("dRLimitForMatching");

  selectedHLTBits = pset.getParameter<vstring>("selectedHLTBits");
}

void UEJetValidation::beginJob( const EventSetup& )
{
  TFileDirectory gendir = fs->mkdir("Gen");

  h2d_jetsizeNchg_chggenjet =
    gendir.make<TH2D>("h2d_jetsizeNchg_chggenjet",
		   "h2d_jetsizeNchg_chggenjet;p_{T}(chg gen jet) (GeV/c);N(chg) Jet Size (rad)",
		   300, 0., 600., 100, 0., 1.);
  
  h2d_jetsizePtsum_chggenjet =
    gendir.make<TH2D>("h2d_jetsizePtsum_chggenjet",
		   "h2d_jetsizePtsum_chggenjet;p_{T}(chg gen jet) (GeV/c);#Sigmap_{T} Jet Size (rad)",
		   300, 0., 600., 100, 0., 1.);

  h_dR_tracksjet_calojet                        = new TH1D*[ selectedHLTBits.size() ];
  h_dR_tracksjet_chggenjet                      = new TH1D*[ selectedHLTBits.size() ];
  h_pTratio_tracksjet_chggenjet                 = new TH1D*[ selectedHLTBits.size() ];
  h_eta_chggenjet                               = new TH1D*[ selectedHLTBits.size() ];
  h_phi_chggenjet                               = new TH1D*[ selectedHLTBits.size() ];
  h_pT_chggenjet                                = new TH1D*[ selectedHLTBits.size() ];
  h_eta_chggenjetMatched                        = new TH1D*[ selectedHLTBits.size() ];
  h_phi_chggenjetMatched                        = new TH1D*[ selectedHLTBits.size() ];
  h_pT_chggenjetMatched                         = new TH1D*[ selectedHLTBits.size() ];
  h_pT_tracksjet                                = new TH1D*[ selectedHLTBits.size() ];
  h_pT_calojet                                  = new TH1D*[ selectedHLTBits.size() ];
  h_pT_calojet_hadronic                         = new TH1D*[ selectedHLTBits.size() ];
  h_pT_calojet_electromagnetic                  = new TH1D*[ selectedHLTBits.size() ];
  h_nConstituents_tracksjet                     = new TH1D*[ selectedHLTBits.size() ];
  h_nConstituents_calojet                       = new TH1D*[ selectedHLTBits.size() ];
  h_nConstituents_chggenjet                     = new TH1D*[ selectedHLTBits.size() ];
  h_maxDistance_tracksjet                       = new TH1D*[ selectedHLTBits.size() ];
  h_maxDistance_calojet                         = new TH1D*[ selectedHLTBits.size() ];
  h_maxDistance_chggenjet                       = new TH1D*[ selectedHLTBits.size() ];
  h_jetsizeNchg_tracksjet                       = new TH1D*[ selectedHLTBits.size() ];
  h_jetsizePtsum_tracksjet                      = new TH1D*[ selectedHLTBits.size() ];
  h_jetFragmentation_tracksjet                  = new TH1D*[ selectedHLTBits.size() ];

  h2d_DrTrackJetCaloJet_PtTrackJet              = new TH2D*[ selectedHLTBits.size() ];
  h2d_DrTrackJetChgGenJet_PtTrackJet            = new TH2D*[ selectedHLTBits.size() ];
  h2d_pTratio_tracksjet_calojet                 = new TH2D*[ selectedHLTBits.size() ];
  h2d_pTRatio_tracksjet_calojet_hadronic        = new TH2D*[ selectedHLTBits.size() ];
  h2d_pTRatio_tracksjet_calojet_electromagnetic = new TH2D*[ selectedHLTBits.size() ];
  h2d_pTratio_tracksjet_chggenjet               = new TH2D*[ selectedHLTBits.size() ];
  h2d_nConstituents_tracksjet_calojet           = new TH2D*[ selectedHLTBits.size() ];
  h2d_nConstituents_tracksjet_chggenjet         = new TH2D*[ selectedHLTBits.size() ];
  h2d_maxDistance_tracksjet_calojet             = new TH2D*[ selectedHLTBits.size() ];
  h2d_maxDistance_tracksjet_chggenjet           = new TH2D*[ selectedHLTBits.size() ];
  h2d_nConstituents_tracksjet                   = new TH2D*[ selectedHLTBits.size() ];
  h2d_nConstituents_calojet                     = new TH2D*[ selectedHLTBits.size() ];
  h2d_nConstituents_chggenjet                   = new TH2D*[ selectedHLTBits.size() ];
  h2d_maxDistance_tracksjet                     = new TH2D*[ selectedHLTBits.size() ];
  h2d_maxDistance_calojet                       = new TH2D*[ selectedHLTBits.size() ];
  h2d_maxDistance_chggenjet                     = new TH2D*[ selectedHLTBits.size() ];
  h2d_jetsizeNchg_tracksjet                     = new TH2D*[ selectedHLTBits.size() ];
  h2d_jetsizePtsum_tracksjet                    = new TH2D*[ selectedHLTBits.size() ];
  h2d_nchg_vs_dR                                = new TH2D*[ selectedHLTBits.size() ];
  h2d_ptsum_vs_dR                               = new TH2D*[ selectedHLTBits.size() ];

  vector<string>::iterator it(selectedHLTBits.begin()),itEnd(selectedHLTBits.end());
  for( unsigned int iHLTbit(0); it != itEnd; ++it, ++iHLTbit) 
    {
      std::string selBit = *it;
      TFileDirectory dir = fs->mkdir( selBit );
      
      h_dR_tracksjet_calojet[iHLTbit] = 
	dir.make<TH1D>( "h_dR_tracksjet_calojet", 
			"h_dR_tracksjet_calojet;#DeltaR(tracks jet, calo jet) (rad / #pi);", 
			100,0.,1.);
  
      h_dR_tracksjet_chggenjet[iHLTbit] = 
	dir.make<TH1D>("h_dR_tracksjet_chggenjet",
		       "h_dR_tracksjet_chggenjet;#DeltaR(tracks jet, chg. gen jet) (rad / #pi);",
		       100,0.,1.);
  
      h_pTratio_tracksjet_chggenjet[iHLTbit] =
	dir.make<TH1D>("h_pTratio_tracksjet_chggenjet",
		       "h_pTratio_tracksjet_chggenjet;p_{T}(tracks jet)/p_{T}(calo jet);",
		       160, 0.96, 1.04);

      h_eta_chggenjet[iHLTbit] = 
	dir.make<TH1D>("h_eta_chggenjet",
		       "h_eta_chggenjet;All jets: #eta(chg. gen jet);",
		       100, -2.5, 2.5 );

      h_phi_chggenjet[iHLTbit] = 
	dir.make<TH1D>("h_phi_chggenjet",
		       "h_phi_chggenjet;All jets: #phi(chg. gen jet) (rad / #pi);",
		       100, -1., 1. );

      h_pT_chggenjet[iHLTbit] = 
	dir.make<TH1D>("h_pT_chggenjet",
		       "h_pT_chggenjet;All jets: p_{T}(chg. gen jet) (GeV/c);",
		       300, 0., 600. );

      h_eta_chggenjetMatched[iHLTbit] = 
	dir.make<TH1D>("h_eta_chggenjetMatched",
		       "h_eta_chggenjetMatched;Matched jets: #eta(chg. gen jet);",
		       100, -2.5, 2.5 );

      h_phi_chggenjetMatched[iHLTbit] = 
	dir.make<TH1D>("h_phi_chggenjetMatched",
		       "h_phi_chggenjetMatched;Matched jets: #phi(chg. gen jet) (rad / #pi);",
		       100, -1., 1. );

      h_pT_chggenjetMatched[iHLTbit] = 
	dir.make<TH1D>("h_pT_chggenjetMatched",
		       "h_pT_chggenjetMatched;Matched jets: p_{T}(chg. gen jet) (GeV/c);",
		       300, 0., 600. );
      
      h_pT_tracksjet[iHLTbit] = 
	dir.make<TH1D>("h_pT_tracksjet",
		       "h_pT_tracksjet;p_{T}(tracks jet) (GeV/c);",
		       300, 0., 600. );

      h_pT_calojet[iHLTbit] = 
	dir.make<TH1D>("h_pT_calojet",
		       "h_pT_calojet;p_{T}(calo jet) (GeV/c);",
		       300, 0., 600. );

      h_pT_calojet_hadronic[iHLTbit] = 
	dir.make<TH1D>("h_pT_calojet_hadronic",
		       "h_pT_calojet_hadronic;p_{T,had}(calo jet) (GeV/c);",
		       300, 0., 600. );

      h_pT_calojet_electromagnetic[iHLTbit] = 
	dir.make<TH1D>("h_pT_calojet_electromagnetic",
		       "h_pT_calojet_electromagnetic;p_{T,em}(calo jet) (GeV/c);",
		       300, 0., 600. );

      h_nConstituents_tracksjet[iHLTbit] = 
	dir.make<TH1D>("h_nConstituents_tracksjet",
		       "h_nConstituents_tracksjet;N_{const}(tracks jet);",
		       50, 0.5, 50.5 );

      h_nConstituents_calojet[iHLTbit] = 
	dir.make<TH1D>("h_nConstituents_calojet",
		       "h_nConstituents_calojet;N_{const}(calo jet);",
		       50, 0.5, 50.5 );

      h_nConstituents_chggenjet[iHLTbit] = 
	dir.make<TH1D>("h_nConstituents_chggenjet",
		       "h_nConstituents_chggenjet;N_{const}(chg. gen jet);",
		       50, 0.5, 50.5 );

      h_maxDistance_tracksjet[iHLTbit] = 
	dir.make<TH1D>("h_maxDistance_tracksjet",
		       "h_maxDistance_tracksjet;D_{max}(tracks jet) (rad);",
		       100, 0., 1. );

      h_maxDistance_calojet[iHLTbit] = 
	dir.make<TH1D>("h_maxDistance_calojet",
		       "h_maxDistance_calojet;D_{max}(calo jet) (rad);",
		       100, 0., 1. );

      h_maxDistance_chggenjet[iHLTbit] = 
	dir.make<TH1D>("h_maxDistance_chggenjet",
		       "h_maxDistance_chggenjet;D_{max}(chg. gen jet) (rad);",
		       100, 0., 1. );

      h_jetsizeNchg_tracksjet[iHLTbit] =
        dir.make<TH1D>("h_jetsizeNchg_tracksjet",
		       "h_jetsizeNchg_tracksjet;N(chg) Jet Size (rad);",
		       100, 0., 1. );

      h_jetsizePtsum_tracksjet[iHLTbit] =
        dir.make<TH1D>("h_jetsizePtsum_tracksjet",
		       "h_jetsizePtsum_tracksjet;#Sigmap_{T} Jet Size (rad);",
		       100, 0., 1. );

      h_jetFragmentation_tracksjet[iHLTbit] =
	dir.make<TH1D>("h_jetFragmentation_tracksjet",
		       "h_jetFragmentation_tracksjet;z = p(track) / p(track jet);",
		       50, 0., 1. );

      h2d_DrTrackJetCaloJet_PtTrackJet[iHLTbit] = 
	dir.make<TH2D>("h2d_DrTrackJetCaloJet_PtTrackJet",
		       "h2d_DrTrackJetCaloJet_PtTrackJet;p_{T}(track jet) (GeV/c);#DeltaR(track jet, calo jet) (rad / #pi)",
		       300, 0., 600., 100, 0., 1. );

      h2d_DrTrackJetChgGenJet_PtTrackJet[iHLTbit] = 
	dir.make<TH2D>("h2d_DrTrackJetChgGenJet_PtTrackJet",
		       "h2d_DrTrackJetChgGenJet_PtTrackJet;p_{T}(track jet) (GeV/c);#DeltaR(track jet, chg gen jet) (rad / #pi)",
		       300, 0., 600., 100, 0., 1. );

      h2d_pTratio_tracksjet_calojet[iHLTbit] = 
	dir.make<TH2D>("h2d_pTratio_tracksjet_calojet",
		       "h2d_pTratio_tracksjet_calojet;p_{T}(track jet) (GeV/c);p_{T}(track jet)/p_{T}(calo jet)",
		       300, 0., 600., 100, 0., 4.);

      h2d_pTRatio_tracksjet_calojet_hadronic[iHLTbit] = 
	dir.make<TH2D>("h2d_pTRatio_tracksjet_calojet_hadronic",
		       "h2d_pTRatio_tracksjet_calojet_hadronic;p_{T}(track jet) (GeV/c);p_{T}(track jet)/p_{T,had}(calo jet)",
		       300, 0., 600., 100, 0., 4.);

      h2d_pTRatio_tracksjet_calojet_electromagnetic[iHLTbit] = 
	dir.make<TH2D>("h2d_pTRatio_tracksjet_calojet_electromagnetic",
		       "h2d_pTRatio_tracksjet_calojet_electromagnetic;p_{T}(tracks jet) (GeV/c);p_{T}(tracks jet)/p_{T,em}(calo jet)",
		       300, 0., 600., 100, 0., 4.);

      h2d_pTratio_tracksjet_chggenjet[iHLTbit] = 
	dir.make<TH2D>("h2d_pTratio_tracksjet_chggenjet",
		       "h2d_pTratio_tracksjet_chggenjet;p_{T}(tracks jet) (GeV/c);p_{T}(chg. gen jet)/p_{T}(calo jet)",
		       300, 0., 600., 160, 0.96, 1.04);

      h2d_nConstituents_tracksjet_calojet[iHLTbit] = 
	dir.make<TH2D>("h2d_nConstituents_tracksjet_calojet",
		       "h2d_nConstituents_tracksjet_calojet;p_{T}(tracks jet) (GeV/c);N_{const}(tracks jet)/N_{const}(calo jet)",
		       300, 0., 600., 100, 0., 10.);

      h2d_nConstituents_tracksjet_chggenjet[iHLTbit] = 
	dir.make<TH2D>("h2d_nConstituents_tracksjet_chggenjet",
		       "h2d_nConstituents_tracksjet_chggenjet;p_{T}(tracks jet) (GeV/c);N_{const}(tracks jet)/N_{const}(chg. gen jet)",
		       300, 0., 600., 100, 0., 10.);
  
      h2d_maxDistance_tracksjet_calojet[iHLTbit] = 
	dir.make<TH2D>("h2d_maxDistance_tracksjet_calojet",
		       "h2d_maxDistance_tracksjet_calojet;p_{T}(tracks jet) (GeV/c);D_{max}(tracks jet)/D_{max}(calo jet)",
		       300, 0., 600., 100, 0., 5.);
  
      h2d_maxDistance_tracksjet_chggenjet[iHLTbit] = 
	dir.make<TH2D>("h2d_maxDistance_tracksjet_chggenjet",
		       "h2d_maxDistance_tracksjet_chggenjet;p_{T}(tracks jet) (GeV/c);D_{max}(tracks jet)/D_{max}(chg. gen jet)",
		       300, 0., 600., 100, 0., 5.);

      h2d_nConstituents_tracksjet[iHLTbit] =
	dir.make<TH2D>("h2d_nConstituents_tracksjet",
		       "h2d_nConstituents_tracksjet;p_{T}(tracks jet) (GeV/c);N_{const}(tracks jet)",
		       300, 0., 600., 50, 0.5, 50.5 );

      h2d_nConstituents_calojet[iHLTbit] =
	dir.make<TH2D>("h2d_nConstituents_calojet",
		       "h2d_nConstituents_calojet;p_{T}(calo jet) (GeV/c);N_{const}(calo jet)",
		       300, 0., 600., 50, 0.5, 50.5 );

      h2d_nConstituents_chggenjet[iHLTbit] =
	dir.make<TH2D>("h2d_nConstituents_chggenjet",
		       "h2d_nConstituents_chggenjet;p_{T}(chg. gen jet) (GeV/c);N_{const}(chg. gen jet)",
		       300, 0., 600., 50, 0.5, 50.5 );

      h2d_maxDistance_tracksjet[iHLTbit] =
	dir.make<TH2D>("h2d_maxDistance_tracksjet",
		       "h2d_maxDistance_tracksjet;p_{T}(tracks jet) (GeV/c);D_{max}(tracks jet) (rad)",
		       300, 0., 600., 100, 0., 1. );

      h2d_maxDistance_calojet[iHLTbit] =
	dir.make<TH2D>("h2d_maxDistance_calojet",
		       "h2d_maxDistance_calojet;p_{T}(calo jet) (GeV/c);D_{max}(calo jet) (rad)",
		       300, 0., 600., 100, 0., 1.);

      h2d_maxDistance_chggenjet[iHLTbit] =
	dir.make<TH2D>("h2d_maxDistance_chggenjet",
		       "h2d_maxDistance_chggenjet;p_{T}(chg. gen jet) (GeV/c);D_{max}(chg. gen jet) (rad)",
		       300, 0., 600., 100, 0., 1.);

      h2d_jetsizeNchg_tracksjet[iHLTbit] =
        dir.make<TH2D>("h2d_jetsizeNchg_tracksjet",
		       "h2d_jetsizeNchg_tracksjet;p_{T}(tracks jet) (GeV/c);N(chg) Jet Size (rad)",
                       300, 0., 600., 100, 0., 1.);

      h2d_jetsizePtsum_tracksjet[iHLTbit] =
        dir.make<TH2D>("h2d_jetsizePtsum_tracksjet",
		       "h2d_jetsizePtsum_tracksjet;p_{T}(tracks jet) (GeV/c);#Sigmap_{T} Jet Size (rad)",
		       300, 0., 600., 100, 0., 1.);

      h2d_nchg_vs_dR [iHLTbit] =
	dir.make<TH2D>("h2d_nchg_vs_dR",
		       "h2d_nchg_vs_dR;#DeltaR(constitutent, jet) (rad);N(chg)",
		       30, 0., 0.6, 51, -0.5, 50.5 );
      h2d_ptsum_vs_dR[iHLTbit] = 
	dir.make<TH2D>("h2d_ptsum_vs_dR",
		       "h2d_ptsum_vs_dR;#DeltaR(constitutent, jet) (rad);#Sigmap_{T} (GeV/c)",
		       30, 0., 0.6, 150, 0., 300. ); 
    

    }
}

  
void UEJetValidation::analyze( const Event& e, const EventSetup& es)
{
  ///
  /// ask for event scale (e.g. pThat) and return if it is outside the requested range
  ///
  double genEventScale( -1. );
  if ( e.getByLabel( genEventScaleTag, genEventScaleHandle ) ) genEventScale = *genEventScaleHandle;
  if ( genEventScale <  _eventScaleMin ) return;
  if ( genEventScale >= _eventScaleMax ) return;


  ///
  /// look for leading tracks jet, 
  /// return if none found in visible range
  /// (as defined by _PTTHRESHOLD and _ETALIMIT)
  ///
  bool foundLeadingTrackJet( false );
  BasicJet* theLeadingTrackJet;
  if ( e.getByLabel( TrackJetsInputTag, TrackJetsHandle ) )
    {
      if(TrackJetsHandle->size())
        {
          theTrackJets = *TrackJetsHandle;
	  std::stable_sort( theTrackJets.begin(), theTrackJets.end(), PtSorter() );

	  BasicJetCollection::const_iterator it   ( theTrackJets.begin() );
	  BasicJetCollection::const_iterator itEnd( theTrackJets.end()   );
          for ( ; it != itEnd; ++it )
            {
              if ( !foundLeadingTrackJet &&
                   it->pt() >= _PTTHRESHOLD &&
                   TMath::Abs(it->eta()) <= _ETALIMIT )
                {
                  theLeadingTrackJet = (*it).clone();
                  foundLeadingTrackJet = true;
		  break;
                }
            }
        }
    }
  if ( ! foundLeadingTrackJet ) return;

  ///
  /// calculate charged jet size
  ///
  std::vector< const Candidate* > trackJetConstitutents( theLeadingTrackJet->getJetConstituentsQuick() );
  int    nConsitutents( trackJetConstitutents.size() ), nTemp    ( 0  );
  double ptsumConstituents( 0. )                      , ptsumTemp( 0. );
  double pLeadingTrackJet( theLeadingTrackJet->p() );
  TH1D* h_jetFragmentation_tracksjet_temp  = new TH1D("h_jetFragmentation_tracksjet_temp" , "", 50, 0., 1. );

  std::map   < double, double   > constituentsDRvsPT;
  for ( unsigned int iconst(0); iconst<trackJetConstitutents.size(); ++iconst )
    {
      double dR( deltaR( *theLeadingTrackJet, *(trackJetConstitutents[iconst]) ) );
      double pT( trackJetConstitutents[iconst]->pt() );
      double pConstituent( trackJetConstitutents[iconst]->p() );

      constituentsDRvsPT[ dR ] = pT;
      ptsumConstituents       += pT;

      h_jetFragmentation_tracksjet_temp->Fill( pConstituent/pLeadingTrackJet );
   }

  double jetSizeNchg( 0. ), jetSizePtsum( 0. );
  TH1D* h_nchg_vs_dR_temp  = new TH1D("h_nchg_vs_dR_temp" , "", 30, 0., 0.6 );
  TH1D* h_ptsum_vs_dR_temp = new TH1D("h_ptsum_vs_dR_temp", "", 30, 0., 0.6 );

  std::map<double, double>::const_iterator mapit   ( constituentsDRvsPT.begin() );
  std::map<double, double>::const_iterator mapitEnd( constituentsDRvsPT.end()   );
  for ( int iconst(0); mapit != mapitEnd; ++mapit, ++iconst )
    {
      double dR( (*mapit).first  );
      double pT( (*mapit).second );

      ++nTemp;
      ptsumTemp += pT;

      if ( nTemp     > 0.8 * nConsitutents     && jetSizeNchg  == 0. ) jetSizeNchg  = dR;
      if ( ptsumTemp > 0.8 * ptsumConstituents && jetSizePtsum == 0. ) jetSizePtsum = dR;

      h_nchg_vs_dR_temp ->Fill( dR );
      h_ptsum_vs_dR_temp->Fill( dR, pT );
    }

  ///
  /// look for leading charged generator particles jet
  ///
  bool foundLeadingChgGenJet( false );
  GenJet* theLeadingChgGenJet;
  if ( e.getByLabel( ChgGenJetsInputTag, ChgGenJetsHandle ) )
    {
      if ( ChgGenJetsHandle->size() )
        {
          theChgGenJets = *ChgGenJetsHandle;
	  std::stable_sort( theChgGenJets.begin(), theChgGenJets.end(), PtSorter() );

	  GenJetCollection::const_iterator it   ( theChgGenJets.begin() );
	  GenJetCollection::const_iterator itEnd( theChgGenJets.end()   );
          for ( ; it != itEnd; ++it )
            {
              if ( !foundLeadingChgGenJet  &&
                   it->pt() >= _PTTHRESHOLD &&
                   TMath::Abs(it->eta()) <= _ETALIMIT )
                {
                  theLeadingChgGenJet   = (*it).clone();
                  foundLeadingChgGenJet = true;
		  break;
                }
            }
        }
    }
  if ( ! foundLeadingChgGenJet ) return;

  ///
  /// calculate chg gen jet size
  ///
  std::vector< const Candidate* > chgGenJetConstitutents( theLeadingChgGenJet->getJetConstituentsQuick() );
  nConsitutents     = chgGenJetConstitutents.size();
  nTemp             = 0;
  ptsumConstituents = 0.;
  ptsumTemp         = 0.;
  pLeadingTrackJet  = theLeadingChgGenJet->p();

  constituentsDRvsPT.clear();
  for ( unsigned int iconst(0); iconst<chgGenJetConstitutents.size(); ++iconst )
    {
      double dR( deltaR( *theLeadingChgGenJet, *(chgGenJetConstitutents[iconst]) ) );
      double pT( chgGenJetConstitutents[iconst]->pt() );

      constituentsDRvsPT[ dR ] = pT;
      ptsumConstituents       += pT;
   }

  double jetSizeNchgChgGen( 0. ), jetSizePtsumChgGen( 0. );

  mapit    = constituentsDRvsPT.begin();
  mapitEnd = constituentsDRvsPT.end();
  for ( int iconst(0); mapit != mapitEnd; ++mapit, ++iconst )
    {
      double dR( (*mapit).first  );
      double pT( (*mapit).second );

      ++nTemp;
      ptsumTemp += pT;

      if ( nTemp     > 0.8 * nConsitutents     && jetSizeNchgChgGen  == 0. ) jetSizeNchgChgGen  = dR;
      if ( ptsumTemp > 0.8 * ptsumConstituents && jetSizePtsumChgGen == 0. ) jetSizePtsumChgGen = dR;
    }

  h2d_jetsizeNchg_chggenjet ->Fill( theLeadingChgGenJet->pt(), jetSizeNchgChgGen  );
  h2d_jetsizePtsum_chggenjet->Fill( theLeadingChgGenJet->pt(), jetSizePtsumChgGen );
  
  ///
  /// look for leading calo jet
  ///
  bool foundLeadingCaloJet( false );
  CaloJet* theLeadingCaloJet;
  if ( e.getByLabel( CaloJetsInputTag, CaloJetsHandle ) )
    {
      if( CaloJetsHandle->size())
        {
          theCaloJets = *CaloJetsHandle;
          std::stable_sort( theCaloJets.begin(), theCaloJets.end(), PtSorter() );

          CaloJetCollection::const_iterator it   ( theCaloJets.begin() );
          CaloJetCollection::const_iterator itEnd( theCaloJets.end()   );
          for ( ; it != itEnd; ++it )
            {
              if ( !foundLeadingCaloJet &&
                   it->pt() >= _PTTHRESHOLD &&
                   TMath::Abs(it->eta()) <= _ETALIMIT )
                {
                  theLeadingCaloJet = (*it).clone();
                  foundLeadingCaloJet = true;
		  break;
                }
            }
        }
    }
  if ( ! foundLeadingCaloJet ) return;

  ///
  /// access trigger bits by TriggerResults
  ///
  if (e.getByLabel( triggerResultsTag, triggerResults ) )
    {
      triggerNames.init( *(triggerResults.product()) );
      
      if ( triggerResults.product()->wasrun() )
    	{
    	  LogDebug("UEJetValidation") << "at least one path out of " << triggerResults.product()->size() 
 				      << " ran? " << triggerResults.product()->wasrun();
	  
    	  if ( triggerResults.product()->accept() ) 
    	    {
    	      LogDebug("UEJetValidation") << "at least one path accepted? " << triggerResults.product()->accept() ;
	      
    	      const unsigned int n_TriggerResults( triggerResults.product()->size() );
    	      for ( unsigned int itrig( 0 ); itrig < n_TriggerResults; ++itrig )
    		{
    		  LogDebug("UEJetValidation") << "path " << triggerNames.triggerName( itrig ) 
 					      << ", module index " << triggerResults.product()->index( itrig )
 					      << ", state (Ready = 0, Pass = 1, Fail = 2, Exception = 3) " << triggerResults.product()->state( itrig )
 					      << ", accept " << triggerResults.product()->accept( itrig );
		  
      		  if ( triggerResults.product()->accept( itrig ) )
    		    {
		      vector<string>::iterator it(selectedHLTBits.begin()),itEnd(selectedHLTBits.end());
		      for( int iHLTbit(0); it != itEnd; ++it, ++iHLTbit)
			{
			  std::string selBit( *it );
			  bool triggered( selBit == (triggerNames.triggerName( itrig )).c_str() ); 

			  /// fill histograms for trigger selBit
			  if ( triggered )
			    {
			      //fillHistograms( e, es, iHLTbit );
			      if ( foundLeadingChgGenJet ) fillHistogramsChgGen( iHLTbit,
										 *theLeadingTrackJet,
										 *theLeadingChgGenJet );
			      if ( foundLeadingCaloJet ) fillHistogramsCalo( iHLTbit,
									     *theLeadingTrackJet,
									     *theLeadingCaloJet );

			      h_pT_tracksjet           [iHLTbit]->Fill( theLeadingTrackJet->pt()            );
			      h_nConstituents_tracksjet[iHLTbit]->Fill( theLeadingTrackJet->nConstituents() );
			      h_maxDistance_tracksjet  [iHLTbit]->Fill( theLeadingTrackJet->maxDistance()   );
			      h_jetsizeNchg_tracksjet  [iHLTbit]->Fill( jetSizeNchg                         );
			      h_jetsizePtsum_tracksjet [iHLTbit]->Fill( jetSizePtsum                        );

			      h2d_nConstituents_tracksjet[iHLTbit]->Fill( theLeadingTrackJet->pt(),
									  theLeadingTrackJet->nConstituents() );
			      h2d_maxDistance_tracksjet  [iHLTbit]->Fill( theLeadingTrackJet->pt(),
									  theLeadingTrackJet->maxDistance()   );
			      h2d_jetsizeNchg_tracksjet  [iHLTbit]->Fill( theLeadingTrackJet->pt(),
									  jetSizeNchg                         );
			      h2d_jetsizePtsum_tracksjet [iHLTbit]->Fill( theLeadingTrackJet->pt(),
									  jetSizePtsum                        );

			      /// fill jet shape
			      for ( int ibin(1); ibin<=h_nchg_vs_dR_temp->GetNbinsX(); ++ibin )
				{
				  double conversion( double(h_nchg_vs_dR_temp->GetNbinsX())/h_nchg_vs_dR_temp->GetXaxis()->GetXmax() );
				  
				  h2d_nchg_vs_dR [iHLTbit]->Fill( double(ibin-1)/conversion, h_nchg_vs_dR_temp->GetBinContent(ibin) );
				  h2d_ptsum_vs_dR[iHLTbit]->Fill( double(ibin-1)/conversion, h_ptsum_vs_dR_temp->GetBinContent(ibin) );
				}

			      /// fill jet fragmentation
			      for ( int ibin(1); ibin<=h_jetFragmentation_tracksjet_temp->GetNbinsX(); ++ibin )
				{
				  double bincontent( h_jetFragmentation_tracksjet[iHLTbit]->GetBinContent(ibin)
						     + h_jetFragmentation_tracksjet_temp->GetBinContent(ibin) );

				  h_jetFragmentation_tracksjet[iHLTbit]->SetBinContent( ibin, bincontent );
				}
			    }
			}
    		    }
    		}
    	    }
    	}
    }
}
 
///___________________________________________________________________________________
///
void UEJetValidation::fillHistogramsChgGen( int       iHLTbit,
					    BasicJet &theLeadingTrackJet,
					    GenJet   &theLeadingChgGenJet )
{
  double dRByPi( deltaR( theLeadingTrackJet, theLeadingChgGenJet )/TMath::Pi() );
  double pTratio( theLeadingTrackJet.pt()/theLeadingChgGenJet.pt() );

  h_dR_tracksjet_chggenjet     [iHLTbit]->Fill( dRByPi                              );
  h_pTratio_tracksjet_chggenjet[iHLTbit]->Fill( pTratio                             );
  h_nConstituents_chggenjet    [iHLTbit]->Fill( theLeadingChgGenJet.nConstituents() );
  h_maxDistance_chggenjet      [iHLTbit]->Fill( theLeadingChgGenJet.maxDistance()   );

  ///
  /// test reconstruction performance
  ///
  h_eta_chggenjet[iHLTbit]->Fill( theLeadingChgGenJet.eta() );
  h_phi_chggenjet[iHLTbit]->Fill( theLeadingChgGenJet.phi()/TMath::Pi() );
  h_pT_chggenjet [iHLTbit]->Fill( theLeadingChgGenJet.pt() );
 
  h2d_DrTrackJetChgGenJet_PtTrackJet[iHLTbit]->Fill( theLeadingTrackJet.pt(), dRByPi );

  if ( dRByPi <= _dR/TMath::Pi() )
    {
      h_eta_chggenjetMatched[iHLTbit]->Fill( theLeadingChgGenJet.eta() );
      h_phi_chggenjetMatched[iHLTbit]->Fill( theLeadingChgGenJet.phi()/TMath::Pi() );
      h_pT_chggenjetMatched[iHLTbit]->Fill( theLeadingChgGenJet.pt() );

      h2d_pTratio_tracksjet_chggenjet      [iHLTbit]->Fill( theLeadingTrackJet.pt(), pTratio              );
      h2d_nConstituents_tracksjet_chggenjet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
							    double(theLeadingTrackJet.nConstituents())
							    / double(theLeadingChgGenJet.nConstituents()) );
      h2d_maxDistance_tracksjet_chggenjet  [iHLTbit]->Fill( theLeadingTrackJet.pt(),
							    theLeadingTrackJet.maxDistance()
							    / theLeadingChgGenJet.maxDistance()           );
      h2d_nConstituents_chggenjet          [iHLTbit]->Fill( theLeadingChgGenJet.pt(),
							    theLeadingChgGenJet.nConstituents()           );
      h2d_maxDistance_chggenjet            [iHLTbit]->Fill( theLeadingChgGenJet.pt(),
							    theLeadingChgGenJet.maxDistance()             );
    }
}

///___________________________________________________________________________________
///
void UEJetValidation::fillHistogramsCalo( int       iHLTbit,
					  BasicJet &theLeadingTrackJet,
					  CaloJet  &theLeadingCaloJet )
{
  double dRByPi( deltaR( theLeadingTrackJet, theLeadingCaloJet )/TMath::Pi() );
  double pTratio( theLeadingTrackJet.pt()/theLeadingCaloJet.pt() );

  h_dR_tracksjet_calojet      [iHLTbit]->Fill( dRByPi                                                            );
  h_pT_calojet                [iHLTbit]->Fill( theLeadingCaloJet.pt()                                            );
  h_pT_calojet_hadronic       [iHLTbit]->Fill( theLeadingCaloJet.energyFractionHadronic()*theLeadingCaloJet.pt() );
  h_pT_calojet_electromagnetic[iHLTbit]->Fill( theLeadingCaloJet.emEnergyFraction()*theLeadingCaloJet.pt()       );
  h_nConstituents_calojet     [iHLTbit]->Fill( theLeadingCaloJet.nConstituents()                                 );
  h_maxDistance_calojet       [iHLTbit]->Fill( theLeadingCaloJet.maxDistance()                                   );

  h2d_DrTrackJetCaloJet_PtTrackJet[iHLTbit]->Fill( theLeadingTrackJet.pt(), dRByPi );

  if ( dRByPi <= _dR/TMath::Pi() )
    {
      h2d_pTratio_tracksjet_calojet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
						    pTratio );
      h2d_nConstituents_tracksjet_calojet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
							  double(theLeadingTrackJet.nConstituents())/double(theLeadingCaloJet.nConstituents()) );
      h2d_maxDistance_tracksjet_calojet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
							theLeadingTrackJet.maxDistance()/theLeadingCaloJet.maxDistance() );
      
      h2d_pTRatio_tracksjet_calojet_hadronic[iHLTbit]->Fill( theLeadingTrackJet.pt(),
							     pTratio / theLeadingCaloJet.energyFractionHadronic() );
      h2d_pTRatio_tracksjet_calojet_electromagnetic[iHLTbit]->Fill( theLeadingTrackJet.pt(),
								    pTratio / theLeadingCaloJet.emEnergyFraction() );
      h2d_nConstituents_calojet[iHLTbit]->Fill( theLeadingCaloJet.pt(),
						theLeadingCaloJet.nConstituents() );
      h2d_maxDistance_calojet[iHLTbit]->Fill( theLeadingCaloJet.pt(),
					      theLeadingCaloJet.maxDistance() );
    }
}


///___________________________________________________________________________________
///
void UEJetValidation::endJob()
{
}

