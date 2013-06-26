
#include "L1Trigger/CSCTrackFinder/test/src/MultiplicityHistogramList.h"

namespace csctf_analysis
{
  MultiplicityHistogramList::MultiplicityHistogramList()
  {

	TFileDirectory dir = fs->mkdir("TFMultiplicity");
	
	nTFTracks = dir.make<TH1F>("nTFTracks","TF Track Multiplicity",10,0,10);
	highestTFPt = dir.make<TH1F>("highestTFPt","Highest TF Pt",150,0,150);
	highestTFPtMed = dir.make<TH1F>("highestTFPtMed","Highest TF Pt between 1-50GeV",50,0,50);
	highestTFPtLow = dir.make<TH1F>("highestTFPtLow","Highest TF Pt between 1-10GeV",20,0,10);

  	nTFTracks->GetXaxis()->SetTitle("Number of Track Finder Tracks");
  	nTFTracks->GetYaxis()->SetTitle("Events");
  	highestTFPt->GetXaxis()->SetTitle("Highest Track Finder Pt per Event (GeV)");
  	highestTFPt->GetYaxis()->SetTitle("Events");
  	highestTFPtMed->GetXaxis()->SetTitle("Highest Track Finder Pt per Event (GeV)");
  	highestTFPtMed->GetYaxis()->SetTitle("Events");
  	highestTFPtLow->GetXaxis()->SetTitle("Highest Track Finder Pt per Event (GeV)");
  	highestTFPtLow->GetYaxis()->SetTitle("Events");

  }

  void MultiplicityHistogramList::FillMultiplicityHist( std::vector<TFTrack>* trackFinderTrack )
  {
	float highestPt=0;
	float ntf=0;
  	std::vector<TFTrack>::iterator tfTrack;
  	for(tfTrack=trackFinderTrack->begin(); tfTrack != trackFinderTrack->end(); tfTrack++)  
  	{
		ntf++;
		float pt = tfTrack->getPt();
		if(pt>highestPt)
			highestPt=pt;
  	}
	nTFTracks->Fill(ntf);		
	highestTFPt->Fill(highestPt);		
	highestTFPtMed->Fill(highestPt);		
	highestTFPtLow->Fill(highestPt);		
  }

}
