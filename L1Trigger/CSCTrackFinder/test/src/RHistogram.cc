#include <L1Trigger/CSCTrackFinder/test/src/RHistogram.h>
#include <iostream>

namespace csctf_analysis
{

  RHistogram::RHistogram(std::string dirName)
  {
	TFileDirectory dir = m_fs->mkdir(dirName);
	m_histR = dir.make<TH1F>("histR","R",26,0,6.5);
	m_histRvEtaHigh = dir.make<TH2F>("histRvEtaHigh","R v. #eta",32,0.9,2.5,26,0,6.5);
	m_histRvEtaLow = dir.make<TH2F>("histRvEtaLow","R v. #eta",32,0.9,2.5,26,0,6.5);
	m_histRvPhiHigh = dir.make<TH2F>("histRvPhiHigh","R v. #phi",65,0,6.5,26,0,6.5);
	m_histRvPhiLow = dir.make<TH2F>("histRvPhiLow","R v. #phi",65,0,6.5,26,0,6.5);
	m_histRvPtHigh = dir.make<TH2F>("histRvPtHigh","R v. Pt",20,0,100,26,0,6.5);
	m_histRvPtLow = dir.make<TH2F>("histRvPtLow","R v. Pt",20,0,100,26,0,6.5);

	m_histR->GetXaxis()->SetTitle("R");
	m_histR->GetYaxis()->SetTitle("Counts");

	m_histRvEtaHigh->GetXaxis()->SetTitle("#eta of Higher #eta Track");
	m_histRvEtaHigh->GetYaxis()->SetTitle("R");
	m_histRvEtaLow->GetXaxis()->SetTitle("#eta of Lower #eta Track");
	m_histRvEtaLow->GetYaxis()->SetTitle("R");

	m_histRvPhiHigh->GetXaxis()->SetTitle("#phi of Higher #phi Track");
	m_histRvPhiHigh->GetYaxis()->SetTitle("R");
	m_histRvPhiLow->GetXaxis()->SetTitle("#phi of Lower #phi Track");
	m_histRvPhiLow->GetYaxis()->SetTitle("R");

	m_histRvPtHigh->GetXaxis()->SetTitle("Pt of Higher Pt Track");
	m_histRvPtHigh->GetYaxis()->SetTitle("R");
	m_histRvPtLow->GetXaxis()->SetTitle("Pt of Lower Pt Track");
	m_histRvPtLow->GetYaxis()->SetTitle("R");
  }

  void RHistogram::fillR(TFTrack track1, TFTrack track2)
  {
	double eta1 = track1.getEta();
	double eta2 = track2.getEta();
	
	double phi1 = track1.getPhi();
	double phi2 = track2.getPhi();

	double pt1 = track1.getPt();
	double pt2 = track2.getPt();

	double R = sqrt((eta1-eta2)*(eta1-eta2) + 
		(phi1-phi2)*(phi1-phi2));

	m_histR->Fill(R);

	if(eta1>=eta2)
	{
	  m_histRvEtaHigh->Fill(eta1,R);
	  m_histRvEtaLow->Fill(eta2,R);
	}
	else
	{
	  m_histRvEtaHigh->Fill(eta2,R);
	  m_histRvEtaLow->Fill(eta1,R);

	}

	if(phi1>=phi2)
	{
	  m_histRvPhiHigh->Fill(phi1,R);
	  m_histRvPhiLow->Fill(phi2,R);
	}
	else
	{
	  m_histRvPhiHigh->Fill(phi2,R);
	  m_histRvPhiLow->Fill(phi1,R);
	}

	if(pt1>=pt2)
	{
	  m_histRvPtHigh->Fill(pt1,R);
	  m_histRvPtLow->Fill(pt2,R);
	}
	else
	{
	  m_histRvPtHigh->Fill(pt2,R);
	  m_histRvPtLow->Fill(pt1,R);
	}

	/*
	if(R < 0.25)
	{
		std::cout << "Debug R" << std::endl;
		std::cout << "eta1: " << eta1 << std::endl;
		std::cout << "eta2: " << eta2 << std::endl;
	
		std::cout << "phi1: " << phi1 << std::endl;
		std::cout << "phi2: " << phi2 << std::endl;
	
		std::cout << "pt1: " << pt1 << std::endl;
		std::cout << "pt2: " << pt2 << std::endl;
	}
	*/
  }

}
