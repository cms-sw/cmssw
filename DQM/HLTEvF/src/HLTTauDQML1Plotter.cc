#include "DQM/HLTEvF/interface/HLTTauDQML1Plotter.h"
#include "Math/GenVector/VectorUtil.h"

HLTTauDQML1Plotter::HLTTauDQML1Plotter(const edm::ParameterSet& ps,int etbins,int etabins,int phibins,double maxpt,bool ref,double dr) : 
  triggerTag_(ps.getParameter<std::string>("DQMFolder")),
  l1ExtraTaus_(ps.getParameter<edm::InputTag>("L1Taus")),
  l1ExtraJets_(ps.getParameter<edm::InputTag>("L1Jets")),
  l1ExtraElectrons_(ps.getParameter<edm::InputTag>("L1Electrons")),
  l1ExtraMuons_(ps.getParameter<edm::InputTag>("L1Muons")),
  doRefAnalysis_(ref),
  matchDeltaR_(dr),
  maxEt_(maxpt),
  binsEt_(etbins),
  binsEta_(etabins),
  binsPhi_(phibins)
{
  //Declare DQM Store
  DQMStore* store = &*edm::Service<DQMStore>();

  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(triggerTag_);

      l1tauEt_             = store->book1D("L1TauEt","L1 #tau E_{t}",binsEt_,0,maxEt_);
      l1tauEt_->getTH1F()->GetXaxis()->SetTitle("L1 #tau E_{T}");
      l1tauEt_->getTH1F()->GetYaxis()->SetTitle("entries");
 
      l1tauEta_            = store->book1D("L1TauEta","L1 #tau #eta",binsEta_,-2.5,2.5);
      l1tauEta_->getTH1F()->GetXaxis()->SetTitle("L1 #tau #eta");
      l1tauEta_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1tauPhi_            = store->book1D("L1TauPhi","L1 #tau #phi",binsPhi_,-3.2,3.2);
      l1tauPhi_->getTH1F()->GetXaxis()->SetTitle("L1 #tau #phi");
      l1tauPhi_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1jetEt_             = store->book1D("L1JetEt","L1 jet E_{t}",binsEt_,0,maxEt_);
      l1jetEt_->getTH1F()->GetXaxis()->SetTitle("L1 Central Jet E_{T}");
      l1jetEt_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1jetEta_            = store->book1D("L1JetEta","L1 jet #eta",binsEta_,-2.5,2.5);
      l1jetEta_->getTH1F()->GetXaxis()->SetTitle("L1 Central Jet #eta");
      l1jetEta_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1jetPhi_            = store->book1D("L1JetPhi","L1 jet #phi",binsPhi_,-3.2,3.2);
      l1jetPhi_->getTH1F()->GetXaxis()->SetTitle("L1 Central Jet #phi");
      l1jetPhi_->getTH1F()->GetYaxis()->SetTitle("entries");


      inputEvents_         = store->book1D("InputEvents","Events Read",2,0,2);
      inputEvents_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1electronEt_        = store->book1D("L1ElectronEt","L1 electron E_{t}",binsEt_,0,maxEt_);
      l1electronEt_->getTH1F()->GetXaxis()->SetTitle("L1 e/#gamma  E_{T}");
      l1electronEt_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1electronEta_       = store->book1D("L1ElectronEta","L1 electron #eta",binsEta_,-2.5,2.5);
      l1electronEta_->getTH1F()->GetXaxis()->SetTitle("L1 e/#gamma  #eta");
      l1electronEta_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1electronPhi_       = store->book1D("L1ElectronPhi","L1 electron #phi",binsPhi_,-3.2,3.2);
      l1electronPhi_->getTH1F()->GetXaxis()->SetTitle("L1 e/#gamma  #phi");
      l1electronPhi_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1muonEt_            = store->book1D("L1MuonEt","L1 muon p_{t}",binsEt_,0,maxEt_);
      l1muonEt_->getTH1F()->GetXaxis()->SetTitle("L1 #mu  p_{T}");
      l1muonEt_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1muonEta_           = store->book1D("L1MuonEta","L1 muon #eta",binsEta_,-2.5,2.5);
      l1muonEta_->getTH1F()->GetXaxis()->SetTitle("L1 #mu  #eta");
      l1muonEta_->getTH1F()->GetYaxis()->SetTitle("entries");

      l1muonPhi_           = store->book1D("L1MuonPhi","L1 muon #phi",binsPhi_,-3.2,3.2);
      l1muonPhi_->getTH1F()->GetXaxis()->SetTitle("L1 #mu  #phi");
      l1muonPhi_->getTH1F()->GetYaxis()->SetTitle("entries");


      l1doubleTauPath_     = store->book2D("L1DoubleTau","L1 Double Tau Path Et",binsEt_,0,maxEt_,binsEt_,0,maxEt_);
      l1doubleTauPath_->getTH2F()->GetXaxis()->SetTitle("first L1 #tau p_{T}");
      l1doubleTauPath_->getTH2F()->GetYaxis()->SetTitle("second L1 #tau p_{T}");

      l1electronTauPath_   = store->book2D("L1MuonTau","L1 Muon Tau Path Et",binsEt_,0,maxEt_,binsEt_,0,maxEt_);
      l1electronTauPath_->getTH2F()->GetXaxis()->SetTitle("first L1 #tau p_{T}");
      l1electronTauPath_->getTH2F()->GetYaxis()->SetTitle("first L1 #gamma p_{T}");

      l1muonTauPath_       = store->book2D("L1ElectronTau","L1 Electron Tau Path Et",binsEt_,0,maxEt_,binsEt_,0,maxEt_);
      l1muonTauPath_->getTH2F()->GetXaxis()->SetTitle("first L1 #mu p_{T}");
      l1muonTauPath_->getTH2F()->GetYaxis()->SetTitle("second L1 #mu p_{T}");

      firstTauEt_   = store->book1D("L1LeadTauEt","L1 lead #tau ET",binsEt_,0,maxEt_);
      firstTauEt_->getTH1F()->Sumw2();

      secondTauEt_   = store->book1D("L1SecondTauEt","L1 second #tau ET",binsEt_,0,maxEt_);
      secondTauEt_->getTH1F()->Sumw2();
      
      if(doRefAnalysis_)
	{
	  l1tauEtRes_      = store->book1D("L1TauEtResol","L1 #tau E_{t} resolution",40,-2,2);
	  l1tauEtRes_->getTH1F()->GetXaxis()->SetTitle("[L1 #tau E_{T}-Ref #tau E_{T}]/Ref #tau E_{T}");
	  l1tauEtRes_->getTH1F()->GetYaxis()->SetTitle("entries");

	  store->setCurrentFolder(triggerTag_+"/EfficiencyHelpers");
	  l1tauEtEffNum_   = store->book1D("L1TauEtEffNum","L1 #tau E_{t} Efficiency Numerator",binsEt_,0,maxEt_);
	  l1tauEtEffNum_->getTH1F()->Sumw2();

	  
	  l1tauEtEffDenom_ = store->book1D("L1TauEtEffDenom","L1 #tau E_{t} Denominator",binsEt_,0,maxEt_);
	  l1tauEtEffDenom_->getTH1F()->Sumw2();
	  
	  l1tauEtaEffNum_   = store->book1D("L1TauEtaEffNum","L1 #tau #eta Efficiency",binsEta_,-2.5,2.5);
	  l1tauEtaEffNum_->getTH1F()->Sumw2();
	  
	  l1tauEtaEffDenom_ = store->book1D("L1TauEtaEffDenom","L1 #tau #eta Denominator",binsEta_,-2.5,2.5);
	  l1tauEtaEffDenom_->getTH1F()->Sumw2();

	  l1tauPhiEffNum_   = store->book1D("L1TauPhiEffNum","L1 #tau #phi Efficiency",binsPhi_,-3.2,3.2);
	  l1tauPhiEffNum_->getTH1F()->Sumw2();

	  l1tauPhiEffDenom_ = store->book1D("L1TauPhiEffDenom","L1 #tau #phi Denominator",binsPhi_,-3.2,3.2);
	  l1tauPhiEffDenom_->getTH1F()->Sumw2();

	  l1jetEtEffNum_   = store->book1D("L1JetEtEffNum","L1 jet E_{t} Efficiency",binsEt_,0,maxEt_);
	  l1jetEtEffNum_->getTH1F()->Sumw2();

	  l1jetEtEffDenom_ = store->book1D("L1JetEtEffDenom","L1 jet E_{t} Denominator",binsEt_,0,maxEt_);
	  l1jetEtEffDenom_->getTH1F()->Sumw2();

	  l1jetEtaEffNum_   = store->book1D("L1JetEtaEffNum","L1 jet #eta Efficiency",binsEta_,-2.5,2.5);
	  l1jetEtaEffNum_->getTH1F()->Sumw2();

	  l1jetEtaEffDenom_ = store->book1D("L1JetEtaEffDenom","L1 jet #eta Denominator",binsEta_,-2.5,2.5);
	  l1jetEtaEffDenom_->getTH1F()->Sumw2();

	  l1jetPhiEffNum_   = store->book1D("L1JetPhiEffNum","L1 jet #phi Efficiency",binsPhi_,-3.2,3.2);
	  l1jetPhiEffNum_->getTH1F()->Sumw2();

	  l1jetPhiEffDenom_ = store->book1D("L1JetPhiEffDenom","L1 jet #phi Denominator",binsPhi_,-3.2,3.2);
	  l1jetPhiEffDenom_->getTH1F()->Sumw2();

// 	  l1electronEtEffNum_   = store->book1D("L1ElectronEtEffNum","L1 Electron p_t Efficiency",binsEt_,0,maxEt_);
// 	  l1electronEtEffNum_->getTH1F()->Sumw2();
	  
// 	  l1electronEtEffDenom_ = store->book1D("L1ElectronEtEffDenom","L1 Electron p_{t} Denominator",binsEt_,0,maxEt_);
// 	  l1electronEtEffDenom_->getTH1F()->Sumw2();
	  
// 	  l1electronEtaEffNum_   = store->book1D("L1ElectronEtaEffNum","L1 Electron #eta Efficiency",binsEta_,-2.5,2.5);
// 	  l1electronEtaEffNum_->getTH1F()->Sumw2();

// 	  l1electronEtaEffDenom_ = store->book1D("L1ElectronEtaEffDenom","L1 Electron #eta Denominator",binsEta_,-2.5,2.5);
// 	  l1electronEtaEffDenom_->getTH1F()->Sumw2();
	      
// 	  l1electronPhiEffNum_   = store->book1D("L1ElectronPhiEffNum","L1 Electron #phi Efficiency",binsPhi_,-3.2,3.2);
// 	  l1electronPhiEffNum_->getTH1F()->Sumw2();
	  
// 	  l1electronPhiEffDenom_ = store->book1D("L1ElectronPhiEffDenom","L1 Electron #phi Denominator",binsPhi_,-3.2,3.2);
// 	  l1electronPhiEffDenom_->getTH1F()->Sumw2();
	  
// 	  l1muonEtEffNum_   = store->book1D("L1MuonEtEffNum","L1 Muon E_{t} Efficiency",binsEt_,0,maxEt_);
// 	  l1muonEtEffNum_->getTH1F()->Sumw2();
	  
// 	  l1muonEtEffDenom_ = store->book1D("L1MuonEtEffDenom","L1 Muon E_{t} Denominator",binsEt_,0,maxEt_);
// 	  l1muonEtEffDenom_->getTH1F()->Sumw2();

// 	  l1muonEtaEffNum_   = store->book1D("L1MuonEtaEffNum","L1 Muon #eta Efficiency",binsEta_,-2.5,2.5);
// 	  l1muonEtaEffNum_->getTH1F()->Sumw2();

// 	  l1muonEtaEffDenom_ = store->book1D("L1MuonEtaEffDenom","L1 Muon #eta Denominator",binsEta_,-2.5,2.5);
// 	  l1muonEtaEffDenom_->getTH1F()->Sumw2();

// 	  l1muonPhiEffNum_   = store->book1D("L1MuonPhiEffNum","L1 Muon #phi Efficiency",binsPhi_,-3.2,3.2);
// 	  l1muonPhiEffNum_->getTH1F()->Sumw2();

// 	  l1muonPhiEffDenom_ = store->book1D("L1MuonPhiEffDenom","L1 Muon #phi Denominator",binsPhi_,-3.2,3.2);
// 	  l1muonPhiEffDenom_->getTH1F()->Sumw2();
	}
    }


}

HLTTauDQML1Plotter::~HLTTauDQML1Plotter()
{
}

//
// member functions
//

void
HLTTauDQML1Plotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup,const std::vector<LVColl>& refC)
{

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;

  //  if(!doRefAnalysis)
  //    inputEvents_->Fill(0.5);

  if(doRefAnalysis_)
    {
      if(refC.size()>0)
      for(size_t j = 0;j<(refC[0]).size();++j)
	{
	  l1tauEtEffDenom_->Fill((refC[0])[j].pt());
	  l1jetEtEffDenom_->Fill((refC[0])[j].pt());

	  l1tauEtaEffDenom_->Fill((refC[0])[j].eta());
	  l1jetEtaEffDenom_->Fill((refC[0])[j].eta());

	  l1tauPhiEffDenom_->Fill((refC[0])[j].phi());
	  l1jetPhiEffDenom_->Fill((refC[0])[j].phi());

	}
//       if(refC.size()>1)
// 	for(size_t j = 0;j<(refC)[1].size();++j)
// 	  {
// 	    l1electronEtEffDenom_->Fill((refC[1])[j].pt());
// 	    l1electronEtaEffDenom_->Fill((refC[1])[j].eta());
// 	    l1electronEtaEffDenom_->Fill((refC[1])[j].phi());
// 	  }
//       if(refC.size()>2)
// 	for(size_t j = 0;j<(refC)[2].size();++j)
// 	  {
// 	    l1muonEtEffDenom_->Fill((refC[2])[j].pt());
// 	    l1muonEtaEffDenom_->Fill((refC[2])[j].eta());
// 	    l1muonEtaEffDenom_->Fill((refC[2])[j].phi());
// 	  }

    }


  //Analyze L1 Objects(Tau+Jets)
  edm::Handle<L1JetParticleCollection> taus;
  edm::Handle<L1JetParticleCollection> jets;
  edm::Handle<L1EmParticleCollection>  electrons;
  edm::Handle<L1MuonParticleCollection>  muons;


  LVColl pathTaus;
  LVColl pathMuons;
  LVColl pathElectrons;

  //Set Variables for the threshold plot
  LVColl l1taus;
  LVColl l1electrons;
  LVColl l1muons;
  LVColl l1jets;





  if(iEvent.getByLabel(l1ExtraTaus_,taus))
    if((!taus.failedToGet())&&taus->size()>0)
    {
	  if(!doRefAnalysis_)
	    {
	      firstTauEt_->Fill((*taus)[0].pt());
	      if(taus->size()>1)
		secondTauEt_->Fill((*taus)[0].pt());

	    }
	  for(L1JetParticleCollection::const_iterator i = taus->begin();i!=taus->end();++i)
	    {
	      l1taus.push_back(i->p4());
	      if(!doRefAnalysis_)
		{
		  l1tauEt_->Fill(i->et());
		  l1tauEta_->Fill(i->eta());
		  l1tauPhi_->Fill(i->phi());
		  pathTaus.push_back(i->p4());
		}

	    }
	    
    }
  if(iEvent.getByLabel(l1ExtraJets_,jets))
    if((!jets.failedToGet())&&jets->size()>0)
    for(L1JetParticleCollection::const_iterator i = jets->begin();i!=jets->end();++i)
	  {	
	    l1jets.push_back(i->p4());
	    if(!doRefAnalysis_)
	      {
		l1jetEt_->Fill(i->et());
		l1jetEta_->Fill(i->eta());
		l1jetPhi_->Fill(i->phi());
		pathTaus.push_back(i->p4());
	      }

	  }


  if(iEvent.getByLabel(l1ExtraElectrons_,electrons))
    if((!electrons.failedToGet())&&electrons->size()>0)
    for(L1EmParticleCollection::const_iterator i = electrons->begin();i!=electrons->end();++i)
      {
	l1electrons.push_back(i->p4());
	l1electronEt_->Fill(i->et());
	l1electronEta_->Fill(i->eta());
	l1electronPhi_->Fill(i->phi());
	pathElectrons.push_back(i->p4());
      }


  if(iEvent.getByLabel(l1ExtraMuons_,muons))
    if((!muons.failedToGet())&&muons->size()>0)
    for(L1MuonParticleCollection::const_iterator i = muons->begin();i!=muons->end();++i)
      {
	l1muons.push_back(i->p4());
	l1muonEt_->Fill(i->et());
	l1muonEta_->Fill(i->eta());
	l1muonPhi_->Fill(i->phi());
	pathMuons.push_back(i->p4());
      }




  //Now do the efficiency matching

  if(doRefAnalysis_)
    {

      //printf("Reference Taus = %d\n",refC[0].size());
      if(refC.size()>0)
	if(refC[0].size()>0)
	  for(LVColl::const_iterator i=(refC[0]).begin();i!=(refC[0]).end();++i)
	    {
	      std::pair<bool,LV> m=  match(*i,l1taus,matchDeltaR_);
	      if(m.first)
		{
		  l1tauEt_->Fill(m.second.pt());
		  l1tauEta_->Fill(m.second.eta());
		  l1tauPhi_->Fill(m.second.phi());
		  l1tauEtEffNum_->Fill(i->pt());
		  l1tauEtaEffNum_->Fill(i->eta());
		  l1tauPhiEffNum_->Fill(i->phi());
		  l1tauEtRes_->Fill((m.second.pt()-i->pt())/i->pt());
		  pathTaus.push_back(m.second);
		}
	    }

      if(refC.size()>0)
	if(refC[0].size()>0)
	  for(LVColl::const_iterator i=(refC[0]).begin();i!=(refC[0]).end();++i)
	    {
	      std::pair<bool,LV> m=  match(*i,l1jets,matchDeltaR_);
	      if(m.first)
		{
		  l1jetEt_->Fill(m.second.pt());
		  l1jetEta_->Fill(m.second.eta());
		  l1jetPhi_->Fill(m.second.phi());
		  l1jetEtEffNum_->Fill(i->pt());
		  l1jetEtaEffNum_->Fill(i->eta());
		  l1jetPhiEffNum_->Fill(i->phi());
		}
	}

      if(refC.size()>1)
	if(refC[1].size()>0)
	  for(LVColl::const_iterator i=(refC[1]).begin();i!=(refC[1]).end();++i)
	    {
	      std::pair<bool,LV> m=  match(*i,l1electrons,matchDeltaR_);
	      if(m.first)
		{
		  l1electronEt_->Fill(m.second.pt());
		  l1electronEta_->Fill(m.second.eta());
		  l1electronPhi_->Fill(m.second.phi());
// 		  l1electronEtEffNum_->Fill(i->pt());
// 		  l1electronEtaEffNum_->Fill(i->eta());
// 		  l1electronPhiEffNum_->Fill(i->phi());
		  pathElectrons.push_back(m.second);

		}
	    }
      if(refC.size()>2)
	if(refC[2].size()>0)
	  for(LVColl::const_iterator i=(refC[2]).begin();i!=(refC[2]).end();++i)
	    {
	      std::pair<bool,LV> m=  match(*i,l1muons,matchDeltaR_);
	      if(m.first)
		{
		  l1muonEt_->Fill(m.second.pt());
		  l1muonEta_->Fill(m.second.eta());
		  l1muonPhi_->Fill(m.second.phi());
// 		  l1muonEtEffNum_->Fill(i->pt());
// 		  l1muonEtaEffNum_->Fill(i->eta());
// 		  l1muonPhiEffNum_->Fill(i->phi());
		  pathMuons.push_back(m.second);
		}
	    }
    }
  

  //Fill the Threshold Monitoring


  if(pathTaus.size()>1)
    std::sort(pathTaus.begin(),pathTaus.end(),ptSort);
  if(pathElectrons.size()>1)
    std::sort(pathElectrons.begin(),pathElectrons.end(),ptSort);
  if(pathMuons.size()>1)
    std::sort(pathMuons.begin(),pathMuons.end(),ptSort);


  if(pathTaus.size()>0)
    {
      firstTauEt_->Fill(pathTaus[0].pt());
      inputEvents_->Fill(0.5);
    }
  if(pathTaus.size()>1)
    {
      secondTauEt_->Fill(pathTaus[1].pt());
      inputEvents_->Fill(1.5);
    }

  if(pathTaus.size()>=2)
    {
      
      l1doubleTauPath_->Fill(pathTaus[0].pt(),pathTaus[1].pt());
    }
  if(pathTaus.size()>=1&&pathElectrons.size()>=1)
    {
      l1electronTauPath_->Fill(pathTaus[0].pt(),pathElectrons[0].pt());
    }
  if(pathTaus.size()>=1&&pathMuons.size()>=1)
    {
      l1muonTauPath_->Fill(pathTaus[0].pt(),pathMuons[0].pt());
    }
}
			

std::pair<bool,LV> 
HLTTauDQML1Plotter::match(const LV& jet,const LVColl& McInfo,double dr)
{
  bool matched=false;
  LV out;

  if(&McInfo)
    if(McInfo.size()>0)
      for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
      {
	double delta = ROOT::Math::VectorUtil::DeltaR(jet,*it);
	if(delta<dr)
	  {
	    matched=true;
	    out=*it;
	  }
      }
  std::pair<bool,LV> a =std::make_pair(matched,out);
  return a;
}



