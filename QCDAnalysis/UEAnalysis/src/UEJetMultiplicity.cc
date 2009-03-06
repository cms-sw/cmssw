// Authors: F. Ambroglini, L. Fano', F. Bechtel
#include <QCDAnalysis/UEAnalysis/interface/UEJetMultiplicity.h>
 
using namespace edm;
using namespace std;
using namespace reco;

typedef vector<string> vstring;

UEJetMultiplicity::UEJetMultiplicity( const ParameterSet& pset )
{
  ChgGenJetsInputTag = pset.getParameter<InputTag>( "ChgGenJetCollectionName" );
  TrackJetsInputTag  = pset.getParameter<InputTag>( "TracksJetCollectionName" );
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

  selectedHLTBits = pset.getParameter<vstring>("selectedHLTBits");
}

void UEJetMultiplicity::beginJob( const EventSetup& )
{
  h2d_nJets_vs_minPtJet_chggenjet = new TH2D*[ selectedHLTBits.size() ];
  h2d_nJets_vs_minPtJet_trackjet  = new TH2D*[ selectedHLTBits.size() ];

  vector<string>::iterator it(selectedHLTBits.begin()),itEnd(selectedHLTBits.end());
  for( unsigned int iHLTbit(0); it != itEnd; ++it, ++iHLTbit) 
    {
      std::string selBit = *it;
      TFileDirectory dir = fs->mkdir( selBit );

      h2d_nJets_vs_minPtJet_chggenjet[iHLTbit] = 
	dir.make<TH2D>("h2d_nJets_vs_minPtJet_chggenjet",
		       "h2d_nJets_vs_minPtJet_chggenjet;Min. p_{T}(chg gen jet) (GeV/c);Number of jets per event",
		       40, 0., 20., 15, -0.5, 15.5 );

      h2d_nJets_vs_minPtJet_trackjet[iHLTbit] = 
	dir.make<TH2D>("h2d_nJets_vs_minPtJet_trackjet",
		       "h2d_nJets_vs_minPtJet_trackjet;Min. p_{T}(track jet) (GeV/c);Number of jets per event",
		       40, 0., 20., 15, -0.5, 15.5 );

    }
}

  
void UEJetMultiplicity::analyze( const Event& e, const EventSetup& es)
{
  ///
  /// ask for event scale (e.g. pThat) and return if it is outside the requested range
  ///
  double genEventScale( -1. );
  if ( e.getByLabel( genEventScaleTag, genEventScaleHandle ) ) genEventScale = *genEventScaleHandle;
  if ( genEventScale <  _eventScaleMin ) return;
  if ( genEventScale >= _eventScaleMax ) return;


  ///
  /// temp histos, to be used to find the jet multiplicity as a function of the jet-pT threshold
  ///
  TH1D* h_pT_trackjet  = new TH1D("h_pT_trackjet" , "", 40, 0., 20. );
  TH1D* h_pT_chggenjet = new TH1D("h_pT_chggenjet", "", 40, 0., 20. );
  h_pT_trackjet ->Reset();
  h_pT_chggenjet->Reset();
      
  ///
  /// find track jet multiplicity as a function of the jet-pT threshold
  ///
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
              if (            it->pt()   >= _PTTHRESHOLD &&
                   TMath::Abs(it->eta()) <= _ETALIMIT )
                {
		  h_pT_trackjet->Fill( it->pt() );
                }
            }
        }
    }

  ///
  /// find chg gen jet multiplicity as a function of the jet-pT threshold
  ///
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
              if (            it->pt()   >= _PTTHRESHOLD &&
                   TMath::Abs(it->eta()) <= _ETALIMIT )
                {
		  h_pT_chggenjet->Fill( it->pt() );
                }
            }
        }
    }


  ///
  /// access trigger bits by TriggerResults
  ///
  if (e.getByLabel( triggerResultsTag, triggerResults ) )
    {
      triggerNames.init( *(triggerResults.product()) );
      
      if ( triggerResults.product()->wasrun() )
    	{
    	  LogDebug("UEJetMultiplicity") << "at least one path out of " << triggerResults.product()->size() 
 				      << " ran? " << triggerResults.product()->wasrun();
	  
    	  if ( triggerResults.product()->accept() ) 
    	    {
    	      LogDebug("UEJetMultiplicity") << "at least one path accepted? " << triggerResults.product()->accept() ;
	      
    	      const unsigned int n_TriggerResults( triggerResults.product()->size() );
    	      for ( unsigned int itrig( 0 ); itrig < n_TriggerResults; ++itrig )
    		{
    		  LogDebug("UEJetMultiplicity") << "path " << triggerNames.triggerName( itrig ) 
						<< ", module index " << triggerResults.product()->index( itrig )
						<< ", state (Ready = 0, Pass = 1, Fail = 2, Exception = 3) " 
						<< triggerResults.product()->state( itrig )
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
			      if ( h_pT_chggenjet->GetEntries() )
				{
				  for ( int ibin(1); ibin<=h_pT_chggenjet->GetNbinsX(); ++ibin )
				    {
				      double conversion( double(h_pT_chggenjet->GetNbinsX())/h_pT_chggenjet->GetXaxis()->GetXmax() );
							 
				      h2d_nJets_vs_minPtJet_chggenjet[iHLTbit]->Fill( double(ibin-1)/conversion, 
										      h_pT_chggenjet->Integral(ibin,h_pT_chggenjet->GetNbinsX()+1) );
				    }
				} 
			      
			      if ( h_pT_trackjet->GetEntries() )
				{
                                  for ( int ibin(1); ibin<=h_pT_trackjet->GetNbinsX(); ++ibin )
                                    {
                                      double conversion( double(h_pT_trackjet->GetNbinsX())/h_pT_trackjet->GetXaxis()->GetXmax() );

                                      h2d_nJets_vs_minPtJet_trackjet[iHLTbit]->Fill( double(ibin-1)/conversion,
										     h_pT_trackjet->Integral(ibin,h_pT_trackjet->GetNbinsX()+1) );
                                    }
				}

			    }
			}
    		    }
    		}
    	    }
    	}
    }
}
 
///
///___________________________________________________________________________________
///
void UEJetMultiplicity::endJob()
{
}

