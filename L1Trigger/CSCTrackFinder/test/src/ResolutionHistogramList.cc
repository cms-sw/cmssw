
#include "L1Trigger/CSCTrackFinder/test/src/TrackHistogramList.h"
#include "L1Trigger/CSCTrackFinder/test/src/EffHistogramList.h"
#include "L1Trigger/CSCTrackFinder/test/src/ResolutionHistogramList.h"

namespace csctf_analysis
{
  ResolutionHistogramList::ResolutionHistogramList(const std::string dirname, const edm::ParameterSet* parameters)
  {

	TFileDirectory dir = fs->mkdir(dirname);
	
	// double maxpt=parameters->getUntrackedParameter<double>("MaxPtHist");
	// double minpt=parameters->getUntrackedParameter<double>("MinPtHist");
	// int ptbins=parameters->getUntrackedParameter<double>("BinsPtHist");

	
	PtQ1Res = dir.make<TH1F>("PtQ1Res","Pt Q>=1 Resolution",300,-1.5,1.5);
	PtQ2Res = dir.make<TH1F>("PtQ2Res","Pt Q>=2 Resolution",300,-1.5,1.5);
	PtQ3Res = dir.make<TH1F>("PtQ3Res","Pt Q>=3 Resolution",300,-1.5,1.5);
	EtaQ1Res = dir.make<TH1F>("EtaQ1Res","Eta Q>=1 Resolution",1000,-1, 1);
	EtaQ2Res = dir.make<TH1F>("EtaQ2Res","Eta Q>=2 Resolution",1000,-1, 1);
	EtaQ3Res = dir.make<TH1F>("EtaQ3Res","Eta Q>=3 Resolution",1000,-1, 1);
	PhiQ1Res = dir.make<TH1F>("PhiQ1Res","Phi Q>=1 Resolution",1000,-1, 1);
	PhiQ2Res = dir.make<TH1F>("PhiQ2Res","Phi Q>=2 Resolution",1000,-1, 1);
	PhiQ3Res = dir.make<TH1F>("PhiQ3Res","Phi Q>=3 Resolution",1000,-1, 1);

	PtQ2ResGolden = dir.make<TH1F>("PtQ2ResGolden","Pt Q>=2 Resolution; 1.2 <= eta <= 2.1",300,-1.5,1.5);
	PhiQ2ResGolden = dir.make<TH1F>("PhiQ2ResGolden","Phi Q>=2 Resolution; 1.2 <= eta <= 2.1",1000,-1, 1);
	PtQ2ResHighEta = dir.make<TH1F>("PtQ2ResHighEta","Pt Q>=2 Resolution; eta >= 2.1",300,-1.5,1.5);
	PhiQ2ResHighEta = dir.make<TH1F>("PhiQ2ResHighEta","Phi Q>=2 Resolution; eta >= 2.1",1000,-1, 1);
	PtQ2ResOverlap = dir.make<TH1F>("PtQ2ResOverlap","Pt Q>=2 Resolution; eta <= 1.2",300,-1.5,1.5);
	PhiQ2ResOverlap = dir.make<TH1F>("PhiQ2ResOverlap","Phi Q>=2 Resolution; eta <= 1.2",1000,-1, 1);

  PtQ2Res->GetXaxis()->SetTitle("Pt_{Sim}/Pt_{TF}-1");
  PtQ2Res->GetYaxis()->SetTitle("Counts");
  PhiQ2Res->GetXaxis()->SetTitle("#phi_{Sim}-#phi_{TF}");
  PhiQ2Res->GetYaxis()->SetTitle("Counts");
  EtaQ2Res->GetXaxis()->SetTitle("#eta_{Sim}-#eta_{TF}");
  EtaQ2Res->GetYaxis()->SetTitle("Counts");

  PtQ2ResGolden->GetXaxis()->SetTitle("Pt_{Sim}/Pt_{TF}-1");
  PtQ2ResGolden->GetYaxis()->SetTitle("Counts");
  PhiQ2ResGolden->GetXaxis()->SetTitle("#phi_{Sim}-#phi_{TF}");
  PhiQ2ResGolden->GetYaxis()->SetTitle("Counts");
  PtQ2ResHighEta->GetXaxis()->SetTitle("Pt_{Sim}/Pt_{TF}-1");
  PtQ2ResHighEta->GetYaxis()->SetTitle("Counts");
  PhiQ2ResHighEta->GetXaxis()->SetTitle("#phi_{Sim}-#phi_{TF}");
  PhiQ2ResHighEta->GetYaxis()->SetTitle("Counts");
  PtQ2ResOverlap->GetXaxis()->SetTitle("Pt_{Sim}/Pt_{TF}-1");
  PtQ2ResOverlap->GetYaxis()->SetTitle("Counts");
  PhiQ2ResOverlap->GetXaxis()->SetTitle("#phi_{Sim}-#phi_{TF}");
  PhiQ2ResOverlap->GetYaxis()->SetTitle("Counts");
  }


  void ResolutionHistogramList::FillResolutionHist( RefTrack refTrk, TFTrack tfTrk )
  {
    double ptResd = (refTrk.getPt() / tfTrk.getPt()) - 1;
    double EtaResd = (refTrk.getEta() - tfTrk.getEta() );
    double PhiResd = (refTrk.getPhi() - tfTrk.getPhi() );

    
    if ( refTrk.getQuality() >= 1 )
      {
	PhiQ1Res->Fill( PhiResd ); 
	EtaQ1Res->Fill( EtaResd );
	PtQ1Res->Fill( ptResd );
      }
    if ( refTrk.getQuality() >= 2 )
      {
	PhiQ2Res->Fill( PhiResd ); 
	EtaQ2Res->Fill( EtaResd );
	PtQ2Res->Fill( ptResd );

	double eta = refTrk.getEta();
	//Golden
	if(eta <= 2.1 && eta >= 1.2)
	{
		PhiQ2ResGolden->Fill( PhiResd ); 
		PtQ2ResGolden->Fill( ptResd );
	}

	//High Eta
	if(eta >= 2.1)
	{
		PhiQ2ResHighEta->Fill( PhiResd ); 
		PtQ2ResHighEta->Fill( ptResd );
	}

	//Overlap
	if(eta <= 1.2)
	{
		PhiQ2ResOverlap->Fill( PhiResd ); 
		PtQ2ResOverlap->Fill( ptResd );
	}


      }    
    if ( refTrk.getQuality() >= 3 )
      {
	PhiQ3Res->Fill( PhiResd ); 
	EtaQ3Res->Fill( EtaResd );
	PtQ3Res->Fill( ptResd );
      }
    
    

    


  }

  void ResolutionHistogramList::Print()
  {
  TCanvas* PtRes = new TCanvas("PtRes");
  PtQ2Res->Draw();
  PtRes->Print("ResPt.png","png");

  TCanvas* PhiRes = new TCanvas("PhiRes");
  PhiQ2Res->Draw();
  PhiRes->Print("ResPhi.png","png");

  TCanvas* EtaRes = new TCanvas("EtaRes");
  EtaQ2Res->Draw();
  EtaRes->Print("ResEta.png","png");
  }
}
 
