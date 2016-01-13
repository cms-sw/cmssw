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
  
  // eta & phi 
  //  eta_bmtf_omtf_emtf= ibooker.book1D("eta_bmtf_omtf_emtf", "Eta of BMTF & OMTF & EMTF", 229, -114.5, 114.5);
  eta_bmtf = ibooker.book1D("eta_bmtf", "Eta of BMTF", 229, -114.5, 114.5);
  eta_omtf = ibooker.book1D("eta_omtf", "Eta of OMTF", 229, -114.5, 114.5);
  eta_emtf = ibooker.book1D("eta_emtf", "Eta of EMTF", 229, -114.5, 114.5);

  //  phi_bmtf_omtf_emtf= ibooker.book1D("phi_bmtf_omtf_emtf", "Phi of BMTF & OMTF & EMTF", 144, -0.5, 143.5);
  phi_bmtf = ibooker.book1D("phi_bmtf", "Phi of BMTF", 144, -0.5, 143.5);
  phi_omtf = ibooker.book1D("phi_omtf", "Phi of OMTF", 144, -0.5, 143.5);
  phi_emtf = ibooker.book1D("phi_emtf", "Phi of EMTF", 144, -0.5, 143.5);

  // etaphi_bmtf_omtf_emtf= ibooker.book2D("etaphi_bmtf_omtf_emtf", "EtaPhi of BMTF & OMTF & EMTF", 229, -114.5, 114.5, 144, -0.5, 143.5);
  // etaphi_bmtf = ibooker.book2D("etaphi_bmtf", "EtaPhi of BMTF", 229, -114.5, 114.5, 144, -0.5, 143.5);
  // etaphi_omtf = ibooker.book2D("etaphi_omtf", "EtaPhi of OMTF", 229, -114.5, 114.5, 144, -0.5, 143.5);
  // etaphi_emtf = ibooker.book2D("etaphi_emtf", "EtaPhi of EMTF", 229, -114.5, 114.5, 144, -0.5, 143.5);

  // eta_bmtf_omtf= ibooker.book1D("eta_bmtf_omtf", "Eta of BMTF & OMTF", 229, -114.5, 114.5);
  // eta_bmtf_emtf= ibooker.book1D("eta_bmtf_emtf", "Eta of BMTF & EMTF", 229, -114.5, 114.5);
  // eta_omtf_emtf= ibooker.book1D("eta_omtf_emtf", "Eta of OMTF & EMTF", 229, -114.5, 114.5);

  // phi_bmtf_omtf= ibooker.book1D("phi_bmtf_omtf", "Phi of BMTF & OMTF", 144, -0.5, 143.5);
  // phi_bmtf_emtf= ibooker.book1D("phi_bmtf_emtf", "Phi of BMTF & EMTF", 144, -0.5, 143.5);
  // phi_omtf_emtf= ibooker.book1D("phi_omtf_emtf", "Phi of OMTF & EMTF", 144, -0.5, 143.5);

  pt_bmtf = ibooker.book1D("pt_bmtf", "Pt of BMTF", 1000, 0, 1000);
  pt_omtf = ibooker.book1D("pt_omtf", "Pt of OMTF", 1000, 0, 1000);
  pt_emtf = ibooker.book1D("pt_emtf", "Pt of EMTF", 1000, 0, 1000);


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
	  const bool endcap  = (itMuon->hwEta() >= 68 || itMuon->hwEta() <= (-68));
	  const bool overlap = ((itMuon->hwEta() > 44 && itMuon->hwEta() < 68) ||(itMuon->hwEta() >= (-68) && itMuon->hwEta() <= (-44)));
	  const bool barrel  = (itMuon->hwEta() >= (-44) && itMuon->hwEta() <= 44);
    
      if (endcap)
	{
	  eta_emtf->Fill(itMuon->hwEta());
	  phi_emtf->Fill(itMuon->hwPhi());
	  pt_emtf->Fill(itMuon->hwPt());
	}
      
      if (overlap)
	{
	  eta_omtf->Fill(itMuon->hwEta());
	  phi_omtf->Fill(itMuon->hwPhi());
	  pt_omtf->Fill(itMuon->hwPt());
	}
      
      if (barrel)
	{
	  eta_bmtf->Fill(itMuon->hwEta());
	  phi_bmtf->Fill(itMuon->hwPhi());
	  pt_bmtf->Fill(itMuon->hwPt());
	}
	}
    }
}



  

