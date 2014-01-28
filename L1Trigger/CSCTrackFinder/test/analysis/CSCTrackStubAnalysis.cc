#include <L1Trigger/CSCTrackFinder/test/analysis/CSCTrackStubAnalysis.h>

#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include <L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h>
#include <L1Trigger/DTUtilities/interface/DTConfig.h>

#include <TMath.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TF1.h>


CSCTrackStubAnalysis::CSCTrackStubAnalysis(edm::ParameterSet const& conf)
{
  cntdtts = 0;
  cntcscts = 0;
  least.first = 4095;
  least.second = 4095;
  greatest.first = 0;
  greatest.second = 0;
}

void CSCTrackStubAnalysis::DeleteHistos()
{
  if(hDTts_phi) delete hDTts_phi;
  if(hCSCts_phi) delete hCSCts_phi;
  if(hDTvsCSC_phi) delete hDTvsCSC_phi;
  if(hDTvsCSC_phi_corr) delete hDTvsCSC_phi_corr;
}

void CSCTrackStubAnalysis::beginJob()
{
  hDTts_phi = new TH1I("hDTts_phi","DT Track Stub Phi",4096,-2047,2047);
  hCSCts_phi  = new TH1I("hCSCts_phi","CSC Track Stub Phi",4096,0,4095);
  hDTvsCSC_phi  = new TH2I("hDTvsCSC_phi","DT Track Stub Phi vs. CSC Track Stub Phi",4096,-2047,2047, 4096, 0, 4095);
  hDTvsCSC_phi_corr = new TH2I("hDTvsCSC_phi","Corrected DT Track Stub Phi vs. CSC Track Stub Phi",4096,0,4095, 4096, 0, 4095);

  hDTts_phi->SetDirectory(0);
  hCSCts_phi->SetDirectory(0);
  hDTvsCSC_phi->SetDirectory(0);
  hDTvsCSC_phi_corr->SetDirectory(0);
}

void CSCTrackStubAnalysis::endJob()
{
  // draw graphs and write out to .ps file
  TCanvas* c1 = new TCanvas("c1");

  hDTts_phi->Draw();

  c1->Print("CSCTrackStubAnalysis.ps(");

  hCSCts_phi->Draw();

  c1->Print("CSCTrackStubAnalysis.ps");

  hDTvsCSC_phi->Draw();

  c1->Print("CSCTrackStubAnalysis.ps");

  hDTvsCSC_phi_corr->Draw();

  c1->Print("CSCTrackStubAnalysis.ps)");

  std::cout << "DT Stubs  : " << cntdtts << std::endl;
  std::cout << "CSC Stubs : " << cntcscts << std::endl;
  std::cout << "Bottom-leftmost point : (" << least.first << ',' << least.second  << ")\n";
  std::cout << "Top-rightmost point   : (" << greatest.first << ',' << greatest.second  << ")\n";

  delete c1;

  DeleteHistos();
}

void CSCTrackStubAnalysis::analyze(edm::Event const& e, edm::EventSetup const& es)
{
  edm::Handle<CSCTriggerContainer<csctf::TrackStub> > cscts;
  edm::Handle<L1MuDTChambPhContainer> dttrig;

  e.getByType(cscts);
  e.getByType(dttrig);

  for(int bx = 3; bx <=9; ++bx)
    {
      std::vector<csctf::TrackStub> temp = cscts->get(bx);
      if(temp.size())
	{
	  std::vector<csctf::TrackStub>::const_iterator itr = temp.begin();
	  for(; itr != temp.end(); itr++)
	    {
	      ++cntcscts;
	      hCSCts_phi->Fill(itr->phiPacked());
	    }
	}
    }

  for(int bx = L1MuDTTFConfig::getBxMin(); bx <= L1MuDTTFConfig::getBxMax(); ++bx)
    {
      for(int e = 1; e <=2; ++e)
	{
	  for(int s = 1; s <= 6; ++s)
	    {
	      int wheel = (e == 1) ? 2 : -2;
	      int sector = 2*s - 1;
//	      int csc_bx = bx - ((L1MuDTTFConfig::getBxMin() + L1MuDTTFConfig::getBxMax())/2) + CSCConstants::TIME_OFFSET + 4;
	      int csc_bx = bx - ((L1MuDTTFConfig::getBxMin() + L1MuDTTFConfig::getBxMax())/2) + 4;

	      for(int is = sector; is <= sector+1; ++is)
		{
		  int iss = (is == 12) ? 0 : is;
		  L1MuDTChambPhDigi *dtts[2];

		  for(int stub = 0; stub < 2; ++stub)
		    {
		      dtts[stub] = (stub == 0) ? dttrig->chPhiSegm1(wheel,1,iss,bx) :
			                         dttrig->chPhiSegm2(wheel,1,iss,bx);

		      if(dtts[stub])
			{
			  //std::cout << "DT STUB BX : " << bx << std::endl;
			  //std::cout << "CSC BX     : " << csc_bx <<std::endl;
			  //std::cout << "DT STUB PHI: " << dtts[stub]->phi() << std::endl;
			  hDTts_phi->Fill(dtts[stub]->phi());
			  ++cntdtts;

			  if(csc_bx <= 9 && 3 <= csc_bx)
			    {
			      std::vector<csctf::TrackStub> temp = cscts->get(e, s, csc_bx);

			      std::vector<csctf::TrackStub>::const_iterator itr = temp.begin();
			      std::vector<csctf::TrackStub>::const_iterator end = temp.end();

			      for(; itr != end; itr++)
				{
				  //std::cout << "DT vs. CSC Phi : " << dtts[stub]->phi() << ' ' << itr->phiPacked()<<std::endl;
				  hDTvsCSC_phi->Fill(dtts[stub]->phi(),itr->phiPacked());
				  int dt_corr_phi = dtts[stub]->phi();
				  dt_corr_phi += 614; // move DTphi lower bound to zero.
				  if(is > sector) dt_corr_phi += 2218; //make [-30,30] -> [0,60]
				  dt_corr_phi = ((double)dt_corr_phi) * 3232./3640.; // scale DT binning to CSC binning

				  dt_corr_phi += 491; // match up DT sector boundary inside of CSC sector

				  hDTvsCSC_phi_corr->Fill(dt_corr_phi, itr->phiPacked());

				  if(dt_corr_phi < least.first && itr->phiPacked() < least.second)
				    {
				      least.first = dt_corr_phi;
				      least.second = itr->phiPacked();
				      std::cout << "(" << least.first << ',' << least.second << ")\n";
				    }

				  if(dt_corr_phi >= greatest.first && itr->phiPacked() >= greatest.second && itr->phiPacked() < 4000)
                                    {
                                      greatest.first = dt_corr_phi;
                                      greatest.second = itr->phiPacked();
				      std::cout << "(" << greatest.first << ',' << greatest.second << ")\n";
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
