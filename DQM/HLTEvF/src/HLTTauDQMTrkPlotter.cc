#include "DQM/HLTEvF/interface/HLTTauDQMTrkPlotter.h"
#include "Math/GenVector/VectorUtil.h"

HLTTauDQMTrkPlotter::HLTTauDQMTrkPlotter(const edm::ParameterSet& iConfig,int etbins,int etabins,int phibins,double maxpt,bool ref,double dr){
   jetTagSrc_ = iConfig.getParameter<edm::InputTag>("ConeIsolation");
   isolJets_ = iConfig.getParameter<edm::InputTag>("IsolatedJets");
   mcMatch_ = dr;
   doRef_ = ref;
   folder_ = iConfig.getParameter<std::string>("DQMFolder");
   type_ = iConfig.getParameter<std::string>("Type");
   EtMax_ = maxpt;
   NPtBins_ = etbins;
   NEtaBins_ = etabins;
   NPhiBins_ = phibins;


    store = &*edm::Service<DQMStore>();
  
   if(store)
     {		//Create the histograms
      store->setCurrentFolder(folder_);
      jetEt = store->book1D((type_+"TauEt").c_str(), "Tau Et",NPtBins_,0,EtMax_);
      jetEta = store->book1D((type_+"TauEta").c_str(), "Tau Eta", NEtaBins_, -2.5, 2.5);
      jetPhi = store->book1D((type_+"TauPhi").c_str(), "Tau Phi", NPhiBins_, -3.2, 3.2);
      isoJetEt = store->book1D((type_+"IsolJetEt").c_str(), "Selected Jet E_{t}", NPtBins_, 0,EtMax_);
      isoJetEta = store->book1D((type_+"IsolJetEta").c_str(), "Selected Jet #eta", NEtaBins_, -2.5, 2.5);
      isoJetPhi = store->book1D((type_+"IsolJetPhi").c_str(), "Selected jet #phi", NPhiBins_, -3.2, 3.2);

      nPxlTrksInL25Jet  = store->book1D((type_+"nTracks").c_str(), "# RECO Tracks", 30, 0, 30);
      nQPxlTrksInL25Jet = store->book1D((type_+"nQTracks").c_str(),"# Quality RECO tracks", 15, 0, 15);
      signalLeadTrkPt   = store->book1D((type_+"LeadTrackPt").c_str(), "Lead Track p_{t}", 75, 0, 150);
      hasLeadTrack      = store->book1D((type_+"HasLeadTrack").c_str(), "Lead Track ?", 2, 0, 2);


      EtEffNum=store->book1D((type_+"TauEtEffNum").c_str(),"Efficiency vs E_{t}(Numerator)",NPtBins_,0,EtMax_);
      EtEffNum->getTH1F()->Sumw2();

      EtEffDenom=store->book1D((type_+"TauEtEffDenom").c_str(),"Efficiency vs E_{t}(Denominator)",NPtBins_,0,EtMax_);
      EtEffDenom->getTH1F()->Sumw2();

      EtaEffNum=store->book1D((type_+"TauEtaEffNum").c_str(),"Efficiency vs #eta (Numerator)",NEtaBins_,-2.5,2.5);
      EtaEffNum->getTH1F()->Sumw2();

      EtaEffDenom=store->book1D((type_+"TauEtaEffDenom").c_str(),"Efficiency vs #eta(Denominator)",NEtaBins_,-2.5,2.5);
      EtaEffDenom->getTH1F()->Sumw2();

      PhiEffNum=store->book1D((type_+"TauPhiEffNum").c_str(),"Efficiency vs #phi (Numerator)",NPhiBins_,-3.2,3.2);
      PhiEffNum->getTH1F()->Sumw2();

      PhiEffDenom=store->book1D((type_+"TauPhiEffDenom").c_str(),"Efficiency vs #phi(Denominator)",NPhiBins_,-3.2,3.2);
      PhiEffDenom->getTH1F()->Sumw2();

   }
}


HLTTauDQMTrkPlotter::~HLTTauDQMTrkPlotter(){
}



void 
HLTTauDQMTrkPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup,const LVColl& mcInfo){
   using namespace edm;
   using namespace reco;
   
   Handle<IsolatedTauTagInfoCollection> tauTagInfos;
   Handle<CaloJetCollection> isolJets;			   

  
   if(iEvent.getByLabel(jetTagSrc_, tauTagInfos))
     if(tauTagInfos.isValid())
     {
	     for(unsigned int i=0;i<tauTagInfos->size();++i)
	       {
		 IsolatedTauTagInfo tauTagInfo = (*tauTagInfos)[i];
		 if(&(*tauTagInfo.jet()))
		   {
		     LV theJet=tauTagInfo.jet()->p4();  		         

		     std::pair <bool,LV> m = match(theJet,mcInfo);
		 
		     if((doRef_&&m.first)||(!doRef_))
		       {
			 jetEt->Fill(theJet.Et()); 		  							         
			 jetEta->Fill(theJet.Eta());		  						         
			 jetPhi->Fill(theJet.Phi());		  						         
			 nPxlTrksInL25Jet->Fill(tauTagInfo.allTracks().size());								    
			 nQPxlTrksInL25Jet->Fill(tauTagInfo.selectedTracks().size());							    
		     
			 const TrackRef leadTrk = tauTagInfo.leadingSignalTrack();
			 if(!leadTrk)
			   { 
			     hasLeadTrack->Fill(0.5);
			   }
			 else
			   {
			     hasLeadTrack->Fill(1.5);
			     signalLeadTrkPt->Fill(leadTrk->pt());				 
			   }
		     
			 LV refV;
			 if(doRef_) refV = m.second; else refV=theJet; 

			 EtEffDenom->Fill(refV.pt());
			 EtaEffDenom->Fill(refV.eta());
			 PhiEffDenom->Fill(refV.phi());

			 if(iEvent.getByLabel(isolJets_, isolJets))
			   if(matchJet(*(tauTagInfo.jet()),*isolJets))
			     {
			       isoJetEta->Fill(theJet.Eta());
			       isoJetEt->Fill(theJet.Et());
			       isoJetPhi->Fill(theJet.Phi());
			       
			       EtEffNum->Fill(refV.pt());
			       EtaEffNum->Fill(refV.eta());
			       PhiEffNum->Fill(refV.phi());
			     }
		       }
		   }
	       }
     }
}






std::pair<bool,LV>
HLTTauDQMTrkPlotter::match(const LV& jet, const LVColl& matchingObject)
{
   bool matched = false;
   LV mLV;

   if(matchingObject.size() !=0 )
     {
       for(LVColl::const_iterator i = matchingObject.begin();i != matchingObject.end(); ++i)
	 {
	   double deltaR = ROOT::Math::VectorUtil::DeltaR(jet, *i);
	 if(deltaR < mcMatch_)
	   {
	     matched = true;
	     mLV = *i;
	   }
	 }
      }

   std::pair<bool,LV> p =  std::make_pair(matched,mLV);

   return p;
}

bool 
HLTTauDQMTrkPlotter::matchJet(const reco::Jet& jet,const reco::CaloJetCollection& McInfo)
{

  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;

 if(McInfo.size()>0)
  for(reco::CaloJetCollection::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),it->p4().Vect());
	  if(delta<mcMatch_)
	    {
	      matched=true;
	     
	    }
   }



 return matched;
}
