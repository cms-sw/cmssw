/*
 * \L1TStage2mGMT.cc
 * \author Esmaeel Eskandari Tadavani
*/

#include "DQM/L1TMonitor/interface/L1TStage2mGMT.h"

L1TStage2mGMT::L1TStage2mGMT(const edm::ParameterSet & ps):
  monitorDir(ps.getUntrackedParameter<std::string>("monitorDir","")),
  stage2mgmtSource(ps.getParameter<edm::InputTag>("stage2mgmtSource")),
  verbose(ps.getUntrackedParameter<bool>("verbose", false))
{
  stage2mgmtToken=consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("stage2mgmtSource"));
}

L1TStage2mGMT::~L1TStage2mGMT()
{
}

void L1TStage2mGMT::dqmBeginRun(const edm::Run& iRrun, const edm::EventSetup& eveSetup)
{
}

void L1TStage2mGMT::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& eveSetup)
{
}


void L1TStage2mGMT::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& iRun, const edm::EventSetup& eveSetup)
{
  ibooker.setCurrentFolder(monitorDir);
  // eta ,phi, pt 
  eta_mgmt = ibooker.book1D("eta_mgmt", "#eta of MGMT", 229, -114.5, 114.5);
  phi_mgmt = ibooker.book1D("phi_mgmt", "#phi of MGMT", 144, -0.5, 143.5);
  pt_mgmt = ibooker.book1D("pt_mgmt", "p_{T} of MGMT", 300, 0, 300);
  bx_mgmt = ibooker.book1D("pt_mgmt", "BX", 2048, -0.5, 2047.5);
  etaVSphi_mgmt = ibooker.book2D("etaVSphi_mgmt", "#eta VS #phi of MGMT", 229, -114.5, 114.5, 144, -0.5, 143.5);
  phiVSpt_mgmt = ibooker.book2D("phiVSpt_mgmt", "#phi VS p_{T}of MGMT", 144, -0.5, 143.5, 300, 0, 300);
  etaVSpt_mgmt = ibooker.book2D("etaVSpt_mgmt", "#eta VS p_{T} of MGMT", 229, -114.5, 114.5,300, 0, 300);
  etaVSbx_mgmt = ibooker.book2D("etaVSbx_mgmt", "#eta of MGMT VS bx", 229, -114.5, 114.5, 2048, -0.5, 2047.5);
  phiVSbx_mgmt = ibooker.book2D("phiVSbx_mgmt", "#phi of MGMT VS bx", 144, -0.5, 143.5, 2048, -0.5, 2047.5);
  ptVSbx_mgmt  = ibooker.book2D("ptVSbx_mgmt", "#p_{T} of MGMT VS bx", 300, 0, 300 , 2048, -0.5, 2047.5);

  


}


void L1TStage2mGMT::analyze(const edm::Event & eve, const edm::EventSetup & eveSetup)
{
  if (verbose) {
    edm::LogInfo("L1TStage2mGMT") << "L1TStage2mGMT: analyze...." << std::endl;
  }
  // analyze Jet
  edm::Handle<l1t::MuonBxCollection> Muon;
  eve.getByToken(stage2mgmtToken,Muon);


  for(int itBX=Muon->getFirstBX(); itBX<=Muon->getLastBX(); ++itBX)
    {
      for(l1t::MuonBxCollection::const_iterator itMuon = Muon->begin(itBX); itMuon != Muon->end(itBX); ++itMuon)
	//  for(l1t::MuonBxCollection::const_iterator itMuon = Muon->begin(); itMuon != Muon->end(); ++itMuon)
	{  

	  eta_mgmt->Fill(itMuon->hwEta());
	  phi_mgmt->Fill(itMuon->hwPhi());
	  pt_mgmt->Fill(itMuon->hwPt());
	  bx_mgmt->Fill(itBX);

	  etaVSphi_mgmt->Fill(itMuon->hwEta(),itMuon->hwPhi()); 
	  phiVSpt_mgmt->Fill(itMuon->hwPhi(),itMuon->hwPt()); 
	  etaVSpt_mgmt->Fill(itMuon->hwEta(),itMuon->hwPt());

	  etaVSbx_mgmt->Fill(itMuon->hwEta(),itBX); 
	  phiVSbx_mgmt->Fill(itMuon->hwPhi(),itBX); 
	  ptVSbx_mgmt->Fill(itMuon->hwPt()  ,itBX);  

	}
    }
}



  

