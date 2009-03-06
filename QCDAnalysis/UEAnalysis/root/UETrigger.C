#include "UETrigger.h"
#include <vector>
#include <math.h>

using std::string;

///
///_______________________________________________________________________
///
UETriggerHistograms::UETriggerHistograms( const char* fileName, string *triggerNames )
{
  ///
  /// Constructor for histogram filler.
  ///
  cout << "[UETriggerHistograms] Create file " << fileName << endl;
  file = TFile::Open( fileName, "recreate" );

  file->mkdir( "UETrigger" );
  file->cd   ( "UETrigger" );

  ///
  /// 11 HLT bits :
  /// 4 Min-Bias (Pixel, Hcal, Ecal, general), Zero-Bias, 6 Jet (30, 50, 80, 110, 180, 250)
  ///
  h_triggerAccepts 
    = new TH1D("h_triggerAccepts", 
	       "h_triggerAccepts;HL Trigger Accepts;Events",
	       12, -0.5, 11.5 );

  unsigned int iHLTbit(0);
  for ( ; iHLTbit<11; ++iHLTbit )
    {
      HLTBitNames[iHLTbit] = triggerNames[iHLTbit];
    }
}

///
///_______________________________________________________________________
///
void
UETriggerHistograms::fill( TClonesArray& acceptedTriggers )
{
  ///
  /// Histo filler for reco-only analysis
  /// HL trigger bits *are* available
  ///

  ///
  /// 11 HLT bits :
  /// 4 Min-Bias (Pixel, Hcal, Ecal, general), Zero-Bias, 6 Jet (30, 50, 80, 110, 180, 250)
  ///

  ///
  /// ask if trigger was accepted and fill corresponding bin
  ///
  bool hltAccept( false );
  unsigned int nAcceptedTriggers( acceptedTriggers.GetSize() );
  for ( unsigned int itrig(0); itrig<nAcceptedTriggers; ++itrig )
    {
      unsigned triggerAccept( 11 );

      unsigned int iHLTbit(0);
      for ( ; iHLTbit<11; ++iHLTbit )
	{
	  std::string filterName( acceptedTriggers.At(itrig)->GetName() );      

	  //cout << "[UETrigger] Compare " << filterName << " with " << HLTBitNames[iHLTbit] << endl;
	  if ( filterName == HLTBitNames[iHLTbit] ) triggerAccept = iHLTbit;
	}

      h_triggerAccepts->Fill( triggerAccept );
    }
}



