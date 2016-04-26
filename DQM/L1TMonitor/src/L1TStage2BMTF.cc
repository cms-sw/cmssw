/*
 * \L1TStage2BMTF.cc
 * \author Esmaeel Eskandari Tadavani
 * \November 2015
*/

#include "DQM/L1TMonitor/interface/L1TStage2BMTF.h"

L1TStage2BMTF::L1TStage2BMTF(const edm::ParameterSet & ps):
  monitorDir(ps.getUntrackedParameter<std::string>("monitorDir","")),
  bmtfSource(ps.getParameter<edm::InputTag>("bmtfSource")),
  verbose(ps.getUntrackedParameter<bool>("verbose", false)),
  global_phi(-1000)
{
  bmtfToken=consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfSource"));
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
  
  bmtf_hwEta = ibooker.book1D("bmtf_hwEta", "HW #eta", 447, -223.5, 223.5);
  bmtf_hwLocalPhi = ibooker.book1D("bmtf_hwLocalPhi", "HW Local #phi", 76, -10.5, 65.5);
  bmtf_hwGlobalPhi = ibooker.book1D("bmtf_hwGlobalPhi", "HW Global #phi", 411, -10.5, 400.5);
  bmtf_hwPt  = ibooker.book1D("bmtf_hwPt", "HW p_{T}", 511, -0.5, 510.5);
  bmtf_hwQual= ibooker.book1D("bmtf_hwQual" , "HW Quality", 20, -0.5, 19.5);
  bmtf_proc  = ibooker.book1D("bmtf_proc" , "Processor", 12, -0.5, 11.5);

  bmtf_wedge_bx  = ibooker.book2D("bmtf_wedge_bx", "Wedge vs BX", 12, -0.5, 11.5, 5, -2.5, 2.5);
  bmtf_wedge_bx->setTitle(";Wedge; BX");
  for (int bin = 1; bin < 13; ++bin) {
    bmtf_wedge_bx->setBinLabel(bin, std::to_string(bin), 1);
  }

  bmtf_hwEta_hwLocalPhi = ibooker.book2D("bmtf_hwEta_hwLocalPhi", "HW #eta vs HW Local #phi" , 447, -223.5, 223.5, 201, -100.5, 100.5);
  bmtf_hwEta_hwLocalPhi->setTitle(";HW #eta; HW Local #phi");

  bmtf_hwPt_hwEta  = ibooker.book2D("bmtf_hwPt_hwEta" , "HW p_{T} vs HW #eta", 511, -0.5, 510.5, 447, -223.5, 223.5);
  bmtf_hwPt_hwEta->setTitle(";HW p_{T}; HW #eta");

  bmtf_hwPt_hwLocalPhi  = ibooker.book2D("bmtf_hwPt_hwLocalPhi" , "HW p_{T} vs HW Local #phi", 511, -0.5, 510.5, 201, -100.5, 100.5);
  bmtf_hwPt_hwLocalPhi->setTitle(";HW p_{T}; HW Local #phi");

  bmtf_hwEta_bx    = ibooker.book2D("bmtf_hwEta_bx"   , "HW #eta vs BX"      , 447, -223.5, 223.5,  5, -2.5, 2.5);
  bmtf_hwEta_bx->setTitle(";HW #eta; BX");

  bmtf_hwLocalPhi_bx    = ibooker.book2D("bmtf_hwLocalPhi_bx"   , "HW Local #phi vs BX"      , 201, -100.5, 100.5,  5, -2.5, 2.5);
  bmtf_hwLocalPhi_bx->setTitle(";HW Local #phi; BX");

  bmtf_hwPt_bx     = ibooker.book2D("bmtf_hwPt_bx"    , "HW p_{T} vs BX"     , 511,   -0.5, 510.5,  5, -2.5, 2.5);
  bmtf_hwPt_bx->setTitle(";HW p_{T}; BX");

  bmtf_hwQual_bx   = ibooker.book2D("bmtf_hwQual_bx"  , "HW Quality vs BX"      ,  20,   -0.5,  19.5,  5, -2.5, 2.5);
  bmtf_hwQual_bx->setTitle("; HW Quality; BX");
}

void L1TStage2BMTF::analyze(const edm::Event & eve, const edm::EventSetup & eveSetup)
{
  if (verbose) {
    edm::LogInfo("L1TStage2BMTF") << "L1TStage2BMTF: analyze...." << std::endl;
  }

  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfMuon;
  eve.getByToken(bmtfToken, bmtfMuon);

  for(int itBX=bmtfMuon->getFirstBX(); itBX<=bmtfMuon->getLastBX(); ++itBX)
    {
      for(l1t::RegionalMuonCandBxCollection::const_iterator itMuon = bmtfMuon->begin(itBX); itMuon != bmtfMuon->end(itBX); ++itMuon)
	{  

	  bmtf_hwEta->Fill(itMuon->hwEta());
	  bmtf_hwLocalPhi->Fill(itMuon->hwPhi());
	  bmtf_hwPt->Fill(itMuon->hwPt());
	  bmtf_hwQual->Fill(itMuon->hwQual());
	  bmtf_proc->Fill(itMuon->processor());

          bmtf_wedge_bx->Fill(itMuon->processor(), itBX);  
	  bmtf_hwEta_bx->Fill(itMuon->hwEta(), itBX);  
	  bmtf_hwLocalPhi_bx->Fill(itMuon->hwPhi(), itBX);  
	  bmtf_hwPt_bx->Fill(itMuon->hwPt(), itBX);   
	  bmtf_hwQual_bx->Fill(itMuon->hwQual(), itBX);
 
	  bmtf_hwEta_hwLocalPhi->Fill(itMuon->hwEta(),itMuon->hwPhi());
	  bmtf_hwPt_hwEta->Fill(itMuon->hwPt(), itMuon->hwEta());
	  bmtf_hwPt_hwLocalPhi->Fill(itMuon->hwPt(), itMuon->hwPhi());
 
	  if(itMuon->hwPhi()>=0 && itMuon->hwPhi()<=30)
	    global_phi = itMuon->hwPhi() + itMuon->processor()*30.;
	  if(itMuon->hwPhi()<0)
	    global_phi = 30-itMuon->hwPhi() + (itMuon->processor()-1)*30.;
	  if(itMuon->hwPhi()>30)
	    global_phi = itMuon->hwPhi()-30 + (itMuon->processor()+1)*30.;
    	  bmtf_hwGlobalPhi->Fill(global_phi);

	}
    }
}

