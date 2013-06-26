#include "L1Trigger/CSCTrackFinder/test/src/StatisticsFile.h"




namespace csctf_analysis
{
  StatisticsFile::StatisticsFile( )
  {
    // default constructor
  }
  StatisticsFile::StatisticsFile( const std::string fname )
  {
    Create( fname );


  }

  void StatisticsFile::Create( const std::string fname )
  {
    statFileOut.open( fname.c_str() );
    std::cout << "opening StatisticsFile\n";

  }


  StatisticsFile::~StatisticsFile()
  {
    //ensure close of file
    Close();
  }


  void StatisticsFile::WriteStatistics( TrackHistogramList tfHistList, TrackHistogramList refHistList)
  {
    if ( statFileOut.is_open() )
      {

	statFileOut << "CSCTFEfficiency Statistics File \n";

	statFileOut << "\n\nTotal Sim Tracks " << refHistList.Phi->GetEntries();
	statFileOut << "\nQuality >= 1 Efficiency " << tfHistList.PhiQ1->GetEntries() / refHistList.Phi->GetEntries();
	statFileOut << "\nQuality >= 2 Efficiency " << tfHistList.PhiQ2->GetEntries() / refHistList.Phi->GetEntries();
	statFileOut << "\nQuality >= 3 Efficiency " << tfHistList.PhiQ3->GetEntries() / refHistList.Phi->GetEntries();

	statFileOut << "\n\nGhosts and Lost Track Statistics\n";
	statFileOut << "Quality > 0 Ghosts " << tfHistList.ghostPhi->GetEntries();
	statFileOut << "\nGhosts / Sim Tracks " << tfHistList.ghostPhi->GetEntries() / refHistList.Phi->GetEntries();
	statFileOut << "\n";


      }
    
  }

}
