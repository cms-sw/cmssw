/*
 * \L1TStage2BMTF.cc
 * \author Esmaeel Eskandari Tadavani
*/

#include "DQM/L1TMonitor/interface/L1TStage2BMTF.h"

L1TStage2BMTF::L1TStage2BMTF(const edm::ParameterSet & ps):
  monitorDir(ps.getUntrackedParameter<std::string>("monitorDir","")),
  stage2bmtfSource(ps.getParameter<edm::InputTag>("stage2bmtfSource")),
  verbose(ps.getUntrackedParameter<bool>("verbose", false))
{
  stage2bmtfToken=consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("stage2bmtfSource"));
}

L1TStage2BMTF::~L1TStage2BMTF()
{
}

void L1TStage2BMTF::dqmBeginRun(const edm::Run& iRrun, const edm::EventSetup& eveSetup)
{
}

void L1TStage2BMTF::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& eveSetup)
{
}


void L1TStage2BMTF::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& iRun, const edm::EventSetup& eveSetup)
{
  ibooker.setCurrentFolder(monitorDir);
  
  // eta ,phi, pt 
  eta_bmtf = ibooker.book1D("eta_bmtf", "#eta of BMTF", 22, -0.5, 21.5);
  phi_bmtf = ibooker.book1D("phi_bmtf", "#phi of BMTF", 18, -0.5, 17.5);
  pt_bmtf = ibooker.book1D("pt_bmtf", "p_{T} of BMTF", 64, -0.5, 63.5);
  bx_bmtf = ibooker.book1D("pt_bmtf", "BX", 5,-2.5, 2.5);
  etaVSphi_bmtf = ibooker.book2D("etaVSphi_bmtf", "#eta VS #phi of BMTF", 22, -0.5, 21.5, 18, -0.5, 17.5);
  phiVSpt_bmtf = ibooker.book2D("phiVSpt_bmtf", "#phi VS p_{T}of BMTF", 18, -0.5, 17.5, 64, -0.5, 63.5);
  etaVSpt_bmtf = ibooker.book2D("etaVSpt_bmtf", "#eta VS p_{T} of BMTF", 22, -0.5, 21.5, 64, -0.5, 63.5);
  etaVSbx_bmtf = ibooker.book2D("etaVSbx_bmtf", "#eta VS bx of BMTF", 22, -0.5, 21.5, 5,-2.5, 2.5);
  phiVSbx_bmtf = ibooker.book2D("phiVSbx_bmtf", "#phi VS bx of BMTF", 18, -0.5, 17.5, 5,-2.5, 2.5);
  ptVSbx_bmtf  = ibooker.book2D("ptVSbx_bmtf", "p_{T} VS bx of BMTF", 64, -0.5, 63.5, 5,-2.5, 2.5);

}

void L1TStage2BMTF::analyze(const edm::Event & eve, const edm::EventSetup & eveSetup)
{
  if (verbose) {
    edm::LogInfo("L1TStage2BMTF") << "L1TStage2BMTF: analyze...." << std::endl;
  }
  // analyze Jet
  edm::Handle<l1t::RegionalMuonCandBxCollection> Muon;
  eve.getByToken(stage2bmtfToken,Muon);


  for(int itBX=Muon->getFirstBX(); itBX<=Muon->getLastBX(); ++itBX)
    {
      for(l1t::RegionalMuonCandBxCollection::const_iterator itMuon = Muon->begin(itBX); itMuon != Muon->end(itBX); ++itMuon)
	{  
	  eta_bmtf->Fill(itMuon->hwEta());
	  phi_bmtf->Fill(itMuon->hwPhi());
	  pt_bmtf->Fill(itMuon->hwPt());
	  bx_bmtf->Fill(itBX);

	  etaVSphi_bmtf->Fill(itMuon->hwEta(),itMuon->hwPhi()); 
	  phiVSpt_bmtf->Fill(itMuon->hwPhi(),itMuon->hwPt()); 
	  etaVSpt_bmtf->Fill(itMuon->hwEta(),itMuon->hwPt());

	  etaVSbx_bmtf->Fill(itMuon->hwEta(),itBX); 
	  phiVSbx_bmtf->Fill(itMuon->hwPhi(),itBX); 
	  ptVSbx_bmtf->Fill(itMuon->hwPt()  ,itBX);  


	}
    }
}



  

