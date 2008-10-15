// Authors: F. Ambroglini, L. Fano', F. Bechtel
#include <QCDAnalysis/UEAnalysis/interface/UEJetValidation.h>
 
using namespace edm;
using namespace std;
using namespace reco;

typedef vector<string> vstring;

UEJetValidation::UEJetValidation( const ParameterSet& pset )
{
  ChgGenJetsInputTag = pset.getUntrackedParameter<InputTag>("ChgGenJetCollectionName",std::string(""));
  TrackJetsInputTag  = pset.getUntrackedParameter<InputTag>("TracksJetCollectionName",std::string(""));
  CaloJetsInputTag   = pset.getUntrackedParameter<InputTag>("CaloJetCollectionName",std::string(""));

  // trigger results
  triggerResultsTag = pset.getParameter<InputTag>("triggerResults");
  triggerEventTag   = pset.getParameter<InputTag>("triggerEvent");
  //   hltFilterTag      = pset.getParameter<InputTag>("hltFilter");
  //   triggerName       = pset.getParameter<InputTag>("triggerName");

  _eventScaleMin = pset.getParameter<double>("eventScaleMin");
  _eventScaleMax = pset.getParameter<double>("eventScaleMax");
  _PTTHRESHOLD   = pset.getParameter<double>("pTThreshold");
  _ETALIMIT      = pset.getParameter<double>("etaLimit");
  _dRByPi        = pset.getParameter<double>("dRByPiLimitForMatching");
  _pTratioRange  = pset.getParameter<double>("pTratioRangeForMatching");

  selectedHLTBits = pset.getParameter<vstring>("selectedHLTBits");
}

void UEJetValidation::beginJob( const EventSetup& )
{
  h_dR_tracksjet_calojet                        = new TH1D*[ selectedHLTBits.size() ];
  h_dR_tracksjet_chggenjet                      = new TH1D*[ selectedHLTBits.size() ];
  h_pTratio_tracksjet_chggenjet                 = new TH1D*[ selectedHLTBits.size() ];
  h_eta_chggenjet                               = new TH1D*[ selectedHLTBits.size() ];
  h_phi_chggenjet                               = new TH1D*[ selectedHLTBits.size() ];
  h_pT_chggenjet                                = new TH1D*[ selectedHLTBits.size() ];
  h_eta_chggenjetMatched                        = new TH1D*[ selectedHLTBits.size() ];
  h_phi_chggenjetMatched                        = new TH1D*[ selectedHLTBits.size() ];
  h_pT_chggenjetMatched                         = new TH1D*[ selectedHLTBits.size() ];

  h2d_pTratio_tracksjet_calojet                 = new TH2D*[ selectedHLTBits.size() ];
  h2d_pTRatio_tracksjet_calojet_hadronic        = new TH2D*[ selectedHLTBits.size() ];
  h2d_pTRatio_tracksjet_calojet_electromagnetic = new TH2D*[ selectedHLTBits.size() ];
  h2d_pTratio_tracksjet_chggenjet               = new TH2D*[ selectedHLTBits.size() ];
  h2d_nConstituents_tracksjet_calojet           = new TH2D*[ selectedHLTBits.size() ];
  h2d_nConstituents_tracksjet_chggenjet         = new TH2D*[ selectedHLTBits.size() ];
  h2d_maxDistance_tracksjet_calojet             = new TH2D*[ selectedHLTBits.size() ];
  h2d_maxDistance_tracksjet_chggenjet           = new TH2D*[ selectedHLTBits.size() ];

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
		       100, 0., 4.);

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
		       150, 0., 300. );

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
		       150, 0., 300. );

      h2d_pTratio_tracksjet_calojet[iHLTbit] = 
	dir.make<TH2D>("h2d_pTratio_tracksjet_calojet",
		       "h2d_pTratio_tracksjet_calojet;p_{T}(tracks jet) (GeV/c);p_{T}(tracks jet)/p_{T}(calo jet)",
		       150, 0., 300., 100, 0., 4.);

      h2d_pTRatio_tracksjet_calojet_hadronic[iHLTbit] = 
	dir.make<TH2D>("h2d_pTRatio_tracksjet_calojet_hadronic",
		       "h2d_pTRatio_tracksjet_calojet_hadronic;p_{T}(tracks jet) (GeV/c);p_{T}(tracks jet)/p_{T,had}(calo jet)",
		       150, 0., 300., 100, 0., 4.);

      h2d_pTRatio_tracksjet_calojet_electromagnetic[iHLTbit] = 
	dir.make<TH2D>("h2d_pTRatio_tracksjet_calojet_electromagnetic",
		       "h2d_pTRatio_tracksjet_calojet_electromagnetic;p_{T}(tracks jet) (GeV/c);p_{T}(tracks jet)/p_{T,em}(calo jet)",
		       150, 0., 300., 100, 0., 4.);

      h2d_pTratio_tracksjet_chggenjet[iHLTbit] = 
	dir.make<TH2D>("h2d_pTratio_tracksjet_chggenjet",
		       "h2d_pTratio_tracksjet_chggenjet;p_{T}(tracks jet) (GeV/c);p_{T}(chg. gen jet)/p_{T}(calo jet)",
		       150, 0., 300., 100, 0., 4.);

      h2d_nConstituents_tracksjet_calojet[iHLTbit] = 
	dir.make<TH2D>("h2d_nConstituents_tracksjet_calojet",
		       "h2d_nConstituents_tracksjet_calojet;p_{T}(tracks jet) (GeV/c);N_{const}(tracks jet)/N_{const}(calo jet)",
		       150, 0., 300., 100, 0., 10.);

      h2d_nConstituents_tracksjet_chggenjet[iHLTbit] = 
	dir.make<TH2D>("h2d_nConstituents_tracksjet_chggenjet",
		       "h2d_nConstituents_tracksjet_chggenjet;p_{T}(tracks jet) (GeV/c);N_{const}(tracks jet)/N_{const}(chg. gen jet)",
		       150, 0., 300., 100, 0., 10.);
  
      h2d_maxDistance_tracksjet_calojet[iHLTbit] = 
	dir.make<TH2D>("h2d_maxDistance_tracksjet_calojet",
		       "h2d_maxDistance_tracksjet_calojet;p_{T}(tracks jet) (GeV/c);D_{max}(tracks jet)/D_{max}(calo jet)",
		       150, 0., 300., 100, 0., 5.);
  
      h2d_maxDistance_tracksjet_chggenjet[iHLTbit] = 
	dir.make<TH2D>("h2d_maxDistance_tracksjet_chggenjet",
		       "h2d_maxDistance_tracksjet_chggenjet;p_{T}(tracks jet) (GeV/c);D_{max}(tracks jet)/D_{max}(chg. gen jet)",
		       150, 0., 300., 100, 0., 5.);
    }
}

  
void UEJetValidation::analyze( const Event& e, const EventSetup& es)
{
  ///
  /// ask for event scale (e.g. pThat) and return if it is outside the requested range
  ///
  double genEventScale( -1. );
  if ( e.getByLabel( "genEventScale", genEventScaleHandle ) ) genEventScale = *genEventScaleHandle;
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

  ///
  /// only continue if jets are found which can be compared with tracks jet
  ///
  if ( ! foundLeadingChgGenJet && ! foundLeadingCaloJet ) return;

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

  h_dR_tracksjet_chggenjet[iHLTbit]->Fill( dRByPi );
  h_pTratio_tracksjet_chggenjet[iHLTbit]->Fill( pTratio );
  h2d_pTratio_tracksjet_chggenjet[iHLTbit]->Fill( theLeadingTrackJet.pt(), pTratio );
  h2d_nConstituents_tracksjet_chggenjet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
							double(theLeadingTrackJet.nConstituents())/double(theLeadingChgGenJet.nConstituents()) );
  h2d_maxDistance_tracksjet_chggenjet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
						      theLeadingTrackJet.maxDistance()/theLeadingChgGenJet.maxDistance() );

  ///
  /// test reconstruction performance
  ///
  h_eta_chggenjet[iHLTbit]->Fill( theLeadingChgGenJet.eta() );
  h_phi_chggenjet[iHLTbit]->Fill( theLeadingChgGenJet.phi()/TMath::Pi() );
  h_pT_chggenjet[iHLTbit] ->Fill( theLeadingChgGenJet.pt() );
 
  if ( dRByPi <= _dRByPi && TMath::Abs(1.-pTratio) <= _pTratioRange )
    {
      h_eta_chggenjetMatched[iHLTbit]->Fill( theLeadingChgGenJet.eta() );
      h_phi_chggenjetMatched[iHLTbit]->Fill( theLeadingChgGenJet.phi()/TMath::Pi() );
      h_pT_chggenjetMatched[iHLTbit]->Fill( theLeadingChgGenJet.pt() );
    }
}

///___________________________________________________________________________________
///
void UEJetValidation::fillHistogramsCalo( int       iHLTbit,
					  BasicJet &theLeadingTrackJet,
					  CaloJet  &theLeadingCaloJet )
{
  h_dR_tracksjet_calojet[iHLTbit]->Fill( deltaR( theLeadingTrackJet, theLeadingCaloJet )/TMath::Pi() );
  h2d_pTratio_tracksjet_calojet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
						theLeadingTrackJet.pt()/theLeadingCaloJet.pt());
  h2d_nConstituents_tracksjet_calojet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
						      double(theLeadingTrackJet.nConstituents())/double(theLeadingCaloJet.nConstituents()) );
  h2d_maxDistance_tracksjet_calojet[iHLTbit]->Fill( theLeadingTrackJet.pt(),
						    theLeadingTrackJet.maxDistance()/theLeadingCaloJet.maxDistance() );

  h2d_pTRatio_tracksjet_calojet_hadronic[iHLTbit]->Fill( theLeadingTrackJet.pt(),
							 theLeadingTrackJet.pt()/(theLeadingCaloJet.energyFractionHadronic()*theLeadingCaloJet.pt()) );
  h2d_pTRatio_tracksjet_calojet_electromagnetic[iHLTbit]->Fill( theLeadingTrackJet.pt(),
								theLeadingTrackJet.pt()/(theLeadingCaloJet.emEnergyFraction()*theLeadingCaloJet.pt()) );
}


///___________________________________________________________________________________
///
void UEJetValidation::endJob()
{
}

