// -*- C++ -*-
//
// Package:    MuonAnalyzerSBSM
// Class:      MuonAnalyzerSBSM
// 
/**\class MuonAnalyzerSBSM MuonAnalyzerSBSM.cc Muon/MuonAnalyzerSBSM/src/MuonAnalyzerSBSM.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Philip Hebda
//         Created:  Thu Jun 25 09:34:50 CEST 2009
// $Id: MuonAnalyzer.cc,v 1.5 2013/04/22 16:36:52 wmtan Exp $
//
//
#include "HLTriggerOffline/SUSYBSM/interface/MuonAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TDirectory.h"

#include "HLTriggerOffline/SUSYBSM/interface/PtSorter.h"
//
// class decleration
//

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;


MuonAnalyzerSBSM::MuonAnalyzerSBSM(edm::InputTag triggerTag_v, edm::InputTag muonTag_v)
{
  triggerTag_ = triggerTag_v;
  muonTag_ = muonTag_v;
}

bool MuonAnalyzerSBSM::find(const std::vector<int>& vec, int element)
{
  for(size_t i=0; i<vec.size(); ++i)
    {
      if(vec[i]==element)
	return true;
    }
  return false;
}

//
// member functions
//

// ------------ method called to for each event  ------------
void
MuonAnalyzerSBSM::FillPlots(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   Handle<reco::MuonCollection> theMuonCollectionHandle;
   iEvent.getByLabel(muonTag_, theMuonCollectionHandle);
   Muons = *theMuonCollectionHandle;
   
   //Find reco muon with highest Pt, fill histos with lead reco muon pt and eta
   int indexOfLeadRecoMuon=-1;
   double LeadRecoMuonPt=-1, LeadRecoMuonEta=-1, LeadRecoMuonPhi=-1;
   for(size_t i=0; i<Muons.size(); ++i)
       if(Muons[i].pt() > LeadRecoMuonPt)
	 {
	   indexOfLeadRecoMuon = i;
	   LeadRecoMuonPt = Muons[i].pt();
	   LeadRecoMuonEta = Muons[i].eta();
	   LeadRecoMuonPhi = Muons[i].phi();
	 }
   //hLeadMuonPt->Fill(Pt);
   //hLeadMuonEta->Fill(Eta);

   //Creating histograms of L3 Muons
   Handle<TriggerEvent> theTriggerCollectionHandle;
   iEvent.getByLabel(triggerTag_,theTriggerCollectionHandle);
   trigger::size_type firstMuon=-1, lastMuon=-2;
   int indexOfCollection=-1, idOfLeadMuon=-1, indexOfFilter=-1;
   double LeadL3MuonPt=-1; // , LeadL3MuonEta=-1, LeadL3MuonPhi=-1; // UNUSED
   vector<int> idsOfFilteredMuons;
   bool LeadRecoMuonAssociation=false;
   const string L3MuonCollection = "hltL3MuonCandidates::HLT";
   const string TriggerFilter = "hltSingleMu9L3Filtered9::HLT";
   const TriggerObjectCollection& TOC(theTriggerCollectionHandle->getObjects());
   if(theTriggerCollectionHandle.isValid())
     {
       for(int i=0; i<theTriggerCollectionHandle->sizeCollections(); ++i)
	 {
	   if(L3MuonCollection.compare(theTriggerCollectionHandle->collectionTag(i).encode())==0)
	     {
	       indexOfCollection = i;
	       break;
	     }
	 }
       for(int i=0; i<theTriggerCollectionHandle->sizeFilters(); ++i)
	 {
	   if(TriggerFilter.compare(theTriggerCollectionHandle->filterTag(i).encode())==0)
	     {
	       indexOfFilter=i;
	       //idsOfFilteredMuons = new int[theTriggerCollectionHandle->filterKeys(i).size()];
	       const Keys& KEYS(theTriggerCollectionHandle->filterKeys(i));
	       for(size_t j=0; j<KEYS.size(); ++j)
		 idsOfFilteredMuons.push_back(KEYS[j]);
	       break;
	     }
	 }
       if(indexOfCollection!=-1 && indexOfFilter!=-1)
	 {
	   if(indexOfCollection==0)
	     firstMuon = 0;
	   else
	     firstMuon = theTriggerCollectionHandle->collectionKey(indexOfCollection-1);
	   lastMuon = theTriggerCollectionHandle->collectionKey(indexOfCollection)-1;     
	   for(int i=firstMuon; i<=lastMuon; ++i)
	     {
	       const TriggerObject& TO(TOC[i]);
		 if(TO.pt()>LeadL3MuonPt && find(idsOfFilteredMuons,i))
		 {
		   LeadL3MuonPt = TO.pt();
		   idOfLeadMuon = i;
		 }
		 if(indexOfLeadRecoMuon!=-1 && find(idsOfFilteredMuons,i) && sqrt(std::pow(LeadRecoMuonEta-TO.eta(),2)+std::pow(LeadRecoMuonPhi-TO.phi(),2))<=0.5)
		   LeadRecoMuonAssociation=true;
	     }
	   const TriggerObject& TO(TOC[idOfLeadMuon]);
	   //hL3LeadMuonPt->Fill(L3LeadMuonPt);
	   //hL3LeadMuonEta->Fill(TO.eta());
	   //hL3LeadMuonPhi->Fill(TO.phi());
	   //hL3LeadMuonMass->Fill(TO.mass());
	   //hNumberL3Muons->Fill(lastMuon-firstMuon+1);
	   LeadL3MuonPt = TO.pt();
	   //	   LeadL3MuonEta = TO.eta(); // UNUSED
	   //	   LeadL3MuonPhi = TO.phi(); // UNUSED
	 }
       //else
       //hNumberL3Muons->Fill(0);
     }
   //else
   //hNumberL3Muons->Fill(0);

   //Fill the histos
   if(indexOfLeadRecoMuon!=-1)
     {
       if(std::abs(LeadRecoMuonEta)<=1.2)
	 hLeadRecoMuonPt_1_ByEvent->Fill(LeadRecoMuonPt);
       else if(std::abs(LeadRecoMuonEta)>1.2 && std::abs(LeadRecoMuonEta)<=2.1)
	 hLeadRecoMuonPt_2_ByEvent->Fill(LeadRecoMuonPt);
       else if(std::abs(LeadRecoMuonEta)>2.1)
	 hLeadRecoMuonPt_3_ByEvent->Fill(LeadRecoMuonPt);
       if(LeadRecoMuonPt>=0)
	 hLeadRecoMuonEta_1_ByEvent->Fill(LeadRecoMuonEta);
       if(LeadRecoMuonPt>=10)
	 hLeadRecoMuonEta_2_ByEvent->Fill(LeadRecoMuonEta);
       if(LeadRecoMuonPt>=20)
	 hLeadRecoMuonEta_3_ByEvent->Fill(LeadRecoMuonEta);
     }
   if(LeadRecoMuonAssociation)
     {
       if(std::abs(LeadRecoMuonEta)<=1.2)
	 hLeadAssocRecoMuonPt_1_ByEvent->Fill(LeadRecoMuonPt);
       else if(std::abs(LeadRecoMuonEta)>1.2 && std::abs(LeadRecoMuonEta)<=2.1)
	 hLeadAssocRecoMuonPt_2_ByEvent->Fill(LeadRecoMuonPt);
       else if(std::abs(LeadRecoMuonEta)>2.1)
	 hLeadAssocRecoMuonPt_3_ByEvent->Fill(LeadRecoMuonPt);
       if(LeadRecoMuonPt>=0)
	 hLeadAssocRecoMuonEta_1_ByEvent->Fill(LeadRecoMuonEta);
       if(LeadRecoMuonPt>=10)
	 hLeadAssocRecoMuonEta_2_ByEvent->Fill(LeadRecoMuonEta);
       if(LeadRecoMuonPt>=20)
	 hLeadAssocRecoMuonEta_3_ByEvent->Fill(LeadRecoMuonEta);
     }

   //Muon by muon

   if(Muons.size()!=0)
     {
       for(size_t i=0; i<Muons.size(); ++i)
	 {
	   double RecoMuonPt=Muons[i].pt(), RecoMuonEta=Muons[i].eta();
	   if(std::abs(RecoMuonEta)<=1.2)
	     hRecoMuonPt_1_ByMuon->Fill(RecoMuonPt);
	   else if(std::abs(RecoMuonEta)>1.2 && std::abs(RecoMuonEta)<=2.1)
	     hRecoMuonPt_2_ByMuon->Fill(RecoMuonPt);
	   else if(std::abs(RecoMuonEta)>2.1)
	     hRecoMuonPt_3_ByMuon->Fill(RecoMuonPt);
	   if(RecoMuonPt>=0)
	     hRecoMuonEta_1_ByMuon->Fill(RecoMuonEta);
	   if(RecoMuonPt>=10)
	     hRecoMuonEta_2_ByMuon->Fill(RecoMuonEta);
	   if(RecoMuonPt>=20)
	     hRecoMuonEta_3_ByMuon->Fill(RecoMuonEta);

	   if(lastMuon-firstMuon+1 > 0)
	     {
	       for(int j=firstMuon; j<=lastMuon; ++j)
		 {
		   const TriggerObject& TO(TOC[j]);
		   if(find(idsOfFilteredMuons,j) && sqrt(std::pow(TO.eta()-Muons[i].eta(), 2)+std::pow(TO.phi()-Muons[i].phi(), 2)) <= 0.5)
		     {
		       RecoMuonPt=Muons[i].pt();
		       RecoMuonEta=Muons[i].eta();
		       if(std::abs(RecoMuonEta)<=1.2)
			 hAssocRecoMuonPt_1_ByMuon->Fill(RecoMuonPt);
		       else if(std::abs(RecoMuonEta)>1.2 && std::abs(RecoMuonEta)<=2.1)
			 hAssocRecoMuonPt_2_ByMuon->Fill(RecoMuonPt);
		       else if(std::abs(RecoMuonEta)>2.1)
			 hAssocRecoMuonPt_3_ByMuon->Fill(RecoMuonPt);
		       if(RecoMuonPt>=0)
			 hAssocRecoMuonEta_1_ByMuon->Fill(RecoMuonEta);
		       if(RecoMuonPt>=10)
			 hAssocRecoMuonEta_2_ByMuon->Fill(RecoMuonEta);
		       if(RecoMuonPt>=20)
			 hAssocRecoMuonEta_3_ByMuon->Fill(RecoMuonEta);

		       break; 
		     }
		 }
	     }
	 }
     }

   //Another approach to Muon by Muon, comparing order in pt



   /*for(size_t i=0; i<Muons.size(); ++i)
     {
       for(int j=0; j<lastMuon-firstMuon+1; ++j)
	 {
	   const TriggerObject& TO(TOC[j]);
	   cout<<(sqrt(std::pow(TO.eta()-Muons[i].eta(),2)+std::pow(TO.phi()-Muons[i].phi(),2))<=0.5)<<'\t';
	 }
       cout<<endl;
       }*/
   


#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
MuonAnalyzerSBSM::InitializePlots(DQMStore* dbe_, string subDir)
{
   //now do what ever initialization is needed
  int pt_bins=100, eta_bins=100;
  double pt_floor=0., pt_ceiling = 200., eta_floor=-3.5, eta_ceiling=3.5;

  dbe_->setCurrentFolder(subDir + "/By_Event");
  hLeadRecoMuonPt_1_ByEvent = dbe_->book1D("LeadRecoMuonPt_1_ByEvent" , "Lead Reco Muon P_{t}, |Eta|<1.2, By Event", pt_bins, pt_floor, pt_ceiling);
  hLeadRecoMuonEta_1_ByEvent= dbe_->book1D("LeadRecoMuonEta_1_ByEvent", "Lead Reco Muon Eta, P_{t}>0, By Event", eta_bins, eta_floor,eta_ceiling);
  hLeadRecoMuonPt_2_ByEvent = dbe_->book1D("LeadRecoMuonPt_2_ByEvent" , "Lead Reco Muon P_{t}, 1.2<|Eta|<2.1, By Event", pt_bins, pt_floor, pt_ceiling);
  hLeadRecoMuonEta_2_ByEvent= dbe_->book1D("LeadRecoMuonEta_2_ByEvent", "Lead Reco Muon Eta, P_{t}>10, By Event", eta_bins, eta_floor,eta_ceiling);
  hLeadRecoMuonPt_3_ByEvent = dbe_->book1D("LeadRecoMuonPt_3_ByEvent" , "Lead Reco Muon P_{t}, |Eta|>2.1, By Event", pt_bins, pt_floor, pt_ceiling);
  hLeadRecoMuonEta_3_ByEvent= dbe_->book1D("LeadRecoMuonEta_3_ByEvent", "Lead Reco Muon Eta, P_{t}>20, By Event", eta_bins, eta_floor,eta_ceiling);
  hLeadAssocRecoMuonPt_1_ByEvent = dbe_->book1D("LeadAssocRecoMuonPt_1_ByEvent" , "Lead Assoc Reco Muon P_{t}, |Eta|<1.2, By Event", pt_bins, pt_floor, pt_ceiling);
  hLeadAssocRecoMuonEta_1_ByEvent= dbe_->book1D("LeadAssocRecoMuonEta_1_ByEvent", "Lead Assoc Muon Eta, P_{t}>0, By Event", eta_bins, eta_floor,eta_ceiling);
  hLeadAssocRecoMuonPt_2_ByEvent = dbe_->book1D("LeadAssocRecoMuonPt_2_ByEvent" , "Lead Assoc Reco Muon P_{t}, 1.2<|Eta|<2.1, By Event", pt_bins, pt_floor, pt_ceiling);
  hLeadAssocRecoMuonEta_2_ByEvent= dbe_->book1D("LeadAssocRecoMuonEta_2_ByEvent", "Lead Assoc Muon Eta, P_{t}>10, By Event", eta_bins, eta_floor,eta_ceiling);
  hLeadAssocRecoMuonPt_3_ByEvent = dbe_->book1D("LeadAssocRecoMuonPt_3_ByEvent" , "Lead Assoc Reco Muon P_{t}, |Eta|>2.1, By Event", pt_bins, pt_floor, pt_ceiling);
  hLeadAssocRecoMuonEta_3_ByEvent= dbe_->book1D("LeadAssocRecoMuonEta_3_ByEvent", "Lead Assoc Muon Eta, P_{t}>20, By Event", eta_bins, eta_floor,eta_ceiling);

  dbe_->setCurrentFolder(subDir + "/By_Muon");
  hRecoMuonPt_1_ByMuon = dbe_->book1D("RecoMuonPt_1_ByMuon" , "Reco Muon P_{t}, |Eta|<1.2, By Muon", pt_bins, pt_floor, pt_ceiling);
  hRecoMuonEta_1_ByMuon= dbe_->book1D("RecoMuonEta_1_ByMuon", "Reco Muon Eta, P_{t}>0, By Muon", eta_bins, eta_floor,eta_ceiling);
  hRecoMuonPt_2_ByMuon = dbe_->book1D("RecoMuonPt_2_ByMuon" , "Reco Muon P_{t}, 1.2<|Eta|<2.1, By Muon", pt_bins, pt_floor, pt_ceiling);
  hRecoMuonEta_2_ByMuon= dbe_->book1D("RecoMuonEta_2_ByMuon", "Reco Muon Eta, P_{t}>10, By Muon", eta_bins, eta_floor,eta_ceiling);
  hRecoMuonPt_3_ByMuon = dbe_->book1D("RecoMuonPt_3_ByMuon" , "Reco Muon P_{t}, |Eta|>2.1, By Muon", pt_bins, pt_floor, pt_ceiling);
  hRecoMuonEta_3_ByMuon= dbe_->book1D("RecoMuonEta_3_ByMuon", "Reco Muon Eta, P_{t}>20, By Muon", eta_bins, eta_floor,eta_ceiling);
  hAssocRecoMuonPt_1_ByMuon = dbe_->book1D("AssocRecoMuonPt_1_ByMuon" , "Assoc Reco Muon P_{t}, |Eta|<1.2, By Muon", pt_bins, pt_floor, pt_ceiling);
  hAssocRecoMuonEta_1_ByMuon= dbe_->book1D("AssocRecoMuonEta_1_ByMuon", "Assoc Muon Eta, P_{t}>0, By Muon", eta_bins, eta_floor,eta_ceiling);
  hAssocRecoMuonPt_2_ByMuon = dbe_->book1D("AssocRecoMuonPt_2_ByMuon" , "Assoc Reco Muon P_{t}, 1.2<|Eta|<2.1, By Muon", pt_bins, pt_floor, pt_ceiling);
  hAssocRecoMuonEta_2_ByMuon= dbe_->book1D("AssocRecoMuonEta_2_ByMuon", "Assoc Muon Eta, P_{t}>10, By Muon", eta_bins, eta_floor,eta_ceiling);
  hAssocRecoMuonPt_3_ByMuon = dbe_->book1D("AssocRecoMuonPt_3_ByMuon" , "Assoc Reco Muon P_{t}, |Eta|>2.1, By Muon", pt_bins, pt_floor, pt_ceiling);
  hAssocRecoMuonEta_3_ByMuon= dbe_->book1D("AssocRecoMuonEta_3_ByMuon", "Assoc Muon Eta, P_{t}>20, By Muon", eta_bins, eta_floor,eta_ceiling);
}
//define this as a plug-in
//DEFINE_FWK_MODULE(MuonAnalyzerSBSM);
