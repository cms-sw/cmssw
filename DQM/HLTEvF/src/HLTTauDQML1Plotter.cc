#include "DQM/HLTEvF/interface/HLTTauDQML1Plotter.h"
#include "Math/GenVector/VectorUtil.h"

HLTTauDQML1Plotter::HLTTauDQML1Plotter(const edm::ParameterSet& ps,int etbins,int etabins,int phibins,double maxpt,bool ref,double dr) : 
  triggerTag_(ps.getParameter<std::string>("DQMFolder")),
  l1ExtraTaus_(ps.getParameter<edm::InputTag>("L1Taus")),
  l1ExtraJets_(ps.getParameter<edm::InputTag>("L1Jets")),
  l1ExtraLeptons_(ps.getParameter<edm::InputTag>("L1Leptons")),
  LeptonType_(ps.getParameter<int>("LeptonType")),
  nTriggeredTaus_(ps.getParameter<unsigned>("NTriggeredTaus")),
  nTriggeredLeptons_(ps.getParameter<unsigned>("NTriggeredLeptons")),
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

      l1tauEt_          = store->book1D("L1TauEt","L1 #tau E_{t}",binsEt_,0,maxEt_);
      l1tauEta_         = store->book1D("L1TauEta","L1 #tau #eta",binsEta_,-2.5,2.5);
      l1tauPhi_         = store->book1D("L1TauPhi","L1 #tau #phi",binsPhi_,-3.2,3.2);
      l1jetEt_          = store->book1D("L1JetEt","L1 jet E_{t}",binsEt_,0,maxEt_);
      l1jetEta_         = store->book1D("L1JetEta","L1 jet #eta",binsEta_,-2.5,2.5);
      l1jetPhi_         = store->book1D("L1JetPhi","L1 jet #phi",binsPhi_,-3.2,3.2);


      inputEvents_      = store->book1D("InputEvents","Events Read",2,0,2);
      if(nTriggeredLeptons_>0)
	{
	  l1leptonEt_          = store->book1D("L1LeptonEt","L1 lepton E_{t}",binsEt_,0,maxEt_);
	  l1leptonEta_         = store->book1D("L1LeptonEta","L1 lepton #eta",binsEta_,-2.5,2.5);
	  l1leptonPhi_         = store->book1D("L1LeptonPhi","L1 lepton #phi",binsPhi_,-3.2,3.2);
	}

      if(nTriggeredTaus_ == 1 && nTriggeredLeptons_ == 0)
	  l1tauPath_ = store->book1D("L1TauPathEt","L1Tau Path Et",binsEt_,0,maxEt_);
      else 
	  l1tauPath_ = store->book2D("L1TauPathEt","L1Tau Path Et",binsEt_,0,maxEt_,binsEt_,0,maxEt_);



      
      if(doRefAnalysis_)
	{
	  inputEvents_    = store->book1D("L1RefEvents","InputRefEvents",1,0,1);
	  l1tauEtRes_      = store->book1D("L1TauEtResol","L1 #tau E_{t} resolution",20,-2,2);

	  l1tauEtEffNum_   = store->book1D("L1TauEtEffNum","L1 #tau E_{t} Efficiency",binsEt_,0,maxEt_);
	  l1tauEtEffNum_->getTH1F()->Sumw2();

	  l1tauEtEffDenom_ = store->book1D("L1TauEtEffDenom","L1 #tau E_{t} Denominator",binsEt_,0,maxEt_);
	  l1tauEtEffDenom_->getTH1F()->Sumw2();

	  l1tauEtaEffNum_   = store->book1D("L1TauEtaEffNum","L1 #tau #eta Efficiency",binsEta_,-2.5,2.5);
	  l1tauEtaEffNum_->getTH1F()->Sumw2();

	  l1tauEtaEffDenom_ = store->book1D("L1TauEtaEffDenom","L1 #tau #eta Denominator",binsEta_,-2.5,2.5);
	  l1tauEtaEffDenom_->getTH1F()->Sumw2();

	  l1tauPhiEffNum_   = store->book1D("L1TauPhiEffNum","L1 #tau #phi Efficiency",binsPhi_,-3.2,3.2);
	  l1tauPhiEffNum_->getTH1F()->Sumw2();

	  l1tauPhiEffDenom_ = store->book1D("l1TauPhiEffDenom","L1 #tau #phi Denominator",binsPhi_,-3.2,3.2);
	  l1tauPhiEffDenom_->getTH1F()->Sumw2();

	  l1jetEtEffNum_   = store->book1D("l1JetEtEffNum","L1 jet E_{t} Efficiency",binsEt_,0,maxEt_);
	  l1jetEtEffNum_->getTH1F()->Sumw2();

	  l1jetEtEffDenom_ = store->book1D("l1JetEtEffDenom","L1 jet E_{t} Denominator",binsEt_,0,maxEt_);
	  l1jetEtEffDenom_->getTH1F()->Sumw2();

	  l1jetEtaEffNum_   = store->book1D("l1JetEtaEffNum","L1 jet #eta Efficiency",binsEta_,-2.5,2.5);
	  l1jetEtaEffNum_->getTH1F()->Sumw2();

	  l1jetEtaEffDenom_ = store->book1D("l1JetEtaEffDenom","L1 jet #eta Denominator",binsEta_,-2.5,2.5);
	  l1jetEtaEffDenom_->getTH1F()->Sumw2();

	  l1jetPhiEffNum_   = store->book1D("l1JetPhiEffNum","L1 jet #phi Efficiency",binsPhi_,-3.2,3.2);
	  l1jetPhiEffNum_->getTH1F()->Sumw2();

	  l1jetPhiEffDenom_ = store->book1D("l1JetPhiEffDenom","L1 jet #phi Denominator",binsPhi_,-3.2,3.2);
	  l1jetPhiEffDenom_->getTH1F()->Sumw2();

	  if(nTriggeredLeptons_>0)
	    {
	      l1leptonEtEffNum_   = store->book1D("l1LeptonEtEffNum","L1 #tau E_{t} Efficiency",binsEt_,0,maxEt_);
	      l1leptonEtEffNum_->getTH1F()->Sumw2();

	      l1leptonEtEffDenom_ = store->book1D("l1LeptonEtEffDenom","L1 #tau E_{t} Denominator",binsEt_,0,maxEt_);
	      l1leptonEtEffDenom_->getTH1F()->Sumw2();

	      l1leptonEtaEffNum_   = store->book1D("l1LeptonEtaEffNum","L1 #tau #eta Efficiency",binsEta_,-2.5,2.5);
	      l1leptonEtaEffNum_->getTH1F()->Sumw2();

	      l1leptonEtaEffDenom_ = store->book1D("l1LeptonEtaEffDenom","L1 #tau #eta Denominator",binsEta_,-2.5,2.5);
 	      l1leptonEtaEffDenom_->getTH1F()->Sumw2();

	      l1leptonPhiEffNum_   = store->book1D("l1LeptonPhiEffNum","L1 #tau #phi Efficiency",binsPhi_,-3.2,3.2);
 	      l1leptonPhiEffNum_->getTH1F()->Sumw2();

	      l1leptonPhiEffDenom_ = store->book1D("l1LeptonPhiEffDenom","L1 #tau #phi Denominator",binsPhi_,-3.2,3.2);
 	      l1leptonPhiEffDenom_->getTH1F()->Sumw2();

	    }



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




  bool tau_ok=true;
  bool lepton_ok=true;

  bool isGoodReferenceEvent=false;

   inputEvents_->Fill(0.5);

  if(doRefAnalysis_)
    {
      for(size_t j = 0;j<(refC[0]).size();++j)
	{
	  l1tauEtEffDenom_->Fill((refC[0])[j].pt());
	  l1jetEtEffDenom_->Fill((refC[0])[j].pt());

	  l1tauEtaEffDenom_->Fill((refC[0])[j].eta());
	  l1jetEtaEffDenom_->Fill((refC[0])[j].eta());

	  l1tauPhiEffDenom_->Fill((refC[0])[j].phi());
	  l1jetPhiEffDenom_->Fill((refC[0])[j].phi());

	}

      if(nTriggeredLeptons_>0)  
	for(size_t j = 0;j<(refC)[1].size();++j)
	  {
	    l1leptonEtEffDenom_->Fill((refC[1])[j].pt());
	    l1leptonEtaEffDenom_->Fill((refC[1])[j].eta());
	    l1leptonEtaEffDenom_->Fill((refC[1])[j].phi());
	  }


      //Tau reference
	if(refC[0].size()<nTriggeredTaus_)
	  {
	    tau_ok = false;
	  }
	
      //lepton reference
	if(refC[1].size()<nTriggeredLeptons_)
	  {
	    lepton_ok = false;
	  }

          
	if(lepton_ok&&tau_ok)
	  {
	    inputEvents_->Fill(1.5);
	    isGoodReferenceEvent=true;
	  }
    }


  //Analyze L1 Objects(Tau+Jets)
  edm::Handle<L1JetParticleCollection> taus;
  edm::Handle<L1JetParticleCollection> jets;


  LVColl pathTaus;
  LVColl pathLeptons;

  //Set Variables for the threshold plot
  LVColl l1taus;
  LVColl l1leptons;
  LVColl l1jets;




  if(iEvent.getByLabel(l1ExtraTaus_,taus))
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

  if(iEvent.getByLabel(l1ExtraJets_,jets))
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

  if(nTriggeredLeptons_>0)
      if(LeptonType_ == 11)
	{
	  Handle<L1EmParticleCollection> muons;
	  if(iEvent.getByLabel(l1ExtraLeptons_,muons))
	     for(L1EmParticleCollection::const_iterator i = muons->begin();i!=muons->end();++i)
	       {
		 l1leptons.push_back(i->p4());
		 l1leptonEt_->Fill(i->et());
		 l1leptonEta_->Fill(i->eta());
		 l1leptonPhi_->Fill(i->phi());
		 pathLeptons.push_back(i->p4());

	       }
	}

  if(nTriggeredLeptons_>0)
      if(LeptonType_ == 13)
	{
	  Handle<L1MuonParticleCollection> muons;
	  if(iEvent.getByLabel(l1ExtraLeptons_,muons))
	     for(L1MuonParticleCollection::const_iterator i = muons->begin();i!=muons->end();++i)
	       {
		 l1leptons.push_back(i->p4());
		 l1leptonEt_->Fill(i->et());
		 l1leptonEta_->Fill(i->eta());
		 l1leptonPhi_->Fill(i->phi());
		 pathLeptons.push_back(i->p4());

	       }
	}


  //Now do the efficiency matching

  if(doRefAnalysis_)
    {
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

      if(nTriggeredLeptons_>0)
      for(LVColl::const_iterator i=(refC[1]).begin();i!=(refC[1]).end();++i)
	{
	  std::pair<bool,LV> m=  match(*i,l1leptons,matchDeltaR_);
	  if(m.first)
	    {
      	      l1leptonEt_->Fill(m.second.pt());
	      l1leptonEta_->Fill(m.second.eta());
	      l1leptonPhi_->Fill(m.second.phi());
	      l1leptonEtEffNum_->Fill(i->pt());
	      l1leptonEtaEffNum_->Fill(i->eta());
	      l1leptonPhiEffNum_->Fill(i->phi());
	      pathTaus.push_back(m.second);

	    }
	}




    }
  



  //Fill the Threshold Monitoring


     if(pathTaus.size()>0)
       std::sort(pathTaus.begin(),pathTaus.end(),ptSort);
     if(pathLeptons.size()>0)
       std::sort(pathLeptons.begin(),pathLeptons.end(),ptSort);


      if(nTriggeredTaus_ >=2)
	{
	  if(pathTaus.size()>=nTriggeredTaus_)
	    {
	      l1tauPath_->Fill(pathTaus[0].pt(),pathTaus[1].pt());
	    }
	}
      else if(nTriggeredTaus_ >= 1 && nTriggeredLeptons_>=1)
	{
	  if(pathTaus.size()>=1)
	    l1tauPath_->Fill(pathTaus[0].pt());
	}
      else if(nTriggeredTaus_ >= 1 && nTriggeredLeptons_ >=1)
	{
	  if(pathTaus.size()>0&&pathLeptons.size()>0)
	    {
	      l1tauPath_->Fill(pathTaus[0].pt(),pathLeptons[0].pt());
	    }
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



