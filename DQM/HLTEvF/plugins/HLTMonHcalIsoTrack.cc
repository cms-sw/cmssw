// -*- C++ -*-
//
// Package:    HLTriggerOffline/special
// Class:      DQMHcalIsoTrackAlCaRaw
// 
/**\class DQMHcalIsoTrackAlCaRaw DQMHcalIsoTrackAlCaRaw.cc HLTriggerOffline/special/src/DQMHcalIsoTrackAlCaRaw.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory SAFRONOV
//         Created:  Mon Oct  6 10:10:22 CEST 2008
// $Id: HLTMonHcalIsoTrack.cc,v 1.5 2010/08/07 14:55:56 wmtan Exp $
//
//


// user include files

#include "DQM/HLTEvF/interface/HLTMonHcalIsoTrack.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/Math/interface/deltaR.h"

double HLTMonHcalIsoTrack::getDist(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = fabs(phi1 - phi2); 
  if(dphi>acos(-1)) dphi = 2*acos(-1)-dphi;
  double dr = sqrt(dphi*dphi + pow(eta1-eta2,2));
  return dr;
}

std::pair<int,int> HLTMonHcalIsoTrack::towerIndex(double eta, double phi) 
{
  int ieta = 0;
  int iphi = 0;
  if (eta!=0)
    {
      for (int i=1; i<21; i++)
	{
	  if (fabs(eta)<(i*0.087)&&fabs(eta)>(i-1)*0.087) ieta=int(fabs(eta)/eta)*i;
	}
      if (fabs(eta)>1.740&&fabs(eta)<=1.830) ieta=int(fabs(eta)/eta)*21;
      if (fabs(eta)>1.830&&fabs(eta)<=1.930) ieta=int(fabs(eta)/eta)*22;
      if (fabs(eta)>1.930&&fabs(eta)<=2.043) ieta=int(fabs(eta)/eta)*23;
      if (fabs(eta)>2.043&&fabs(eta)<=2.172) ieta=int(fabs(eta)/eta)*24;
    }
  
  double delta=phi+0.174532925;
  if (delta<0) delta=delta+2*acos(-1);
  if (fabs(eta)<1.740) 
    {
      for (int i=0; i<72; i++)
	{
	  if (delta<(i+1)*0.087266462&&delta>i*0.087266462) iphi=i;
	}
    }
  else 
    {
      for (int i=0; i<36; i++)
	{
	  if (delta<2*(i+1)*0.087266462&&delta>2*i*0.087266462) iphi=2*i;
	}
    }
  
  return std::pair<int,int>(ieta,iphi);
}


HLTMonHcalIsoTrack::HLTMonHcalIsoTrack(const edm::ParameterSet& iConfig)
{
  folderName_ = iConfig.getParameter<std::string>("folderName");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");
  
  useProducerCollections_=iConfig.getParameter<bool>("useProducerCollections");
  hltRAWEventTag_=iConfig.getParameter<std::string>("hltRAWTriggerEventLabel");
  hltAODEventTag_=iConfig.getParameter<std::string>("hltAODTriggerEventLabel");
  
  hltProcess_=iConfig.getParameter<std::string>("hltProcessName");
  
  saveToRootFile_=iConfig.getParameter<bool>("SaveToRootFile");
  
  triggers = iConfig.getParameter<std::vector<edm::ParameterSet> >("triggers");
  for (std::vector<edm::ParameterSet>::iterator inTrig = triggers.begin(); inTrig != triggers.end(); inTrig++)
    {
      trigNames_.push_back(inTrig->getParameter<std::string>("triggerName"));
      l1filterLabels_.push_back(inTrig->getParameter<std::string>("hltL1filterLabel"));
      l2filterLabels_.push_back(inTrig->getParameter<std::string>("hltL2filterLabel"));
      l3filterLabels_.push_back(inTrig->getParameter<std::string>("hltL3filterLabel"));
      l2collectionLabels_.push_back(inTrig->getParameter<std::string>("l2collectionLabel"));
      l3collectionLabels_.push_back(inTrig->getParameter<std::string>("l3collectionLabel"));
    }
}


HLTMonHcalIsoTrack::~HLTMonHcalIsoTrack()
{}

void HLTMonHcalIsoTrack::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  edm::InputTag toLab=edm::InputTag(hltRAWEventTag_,"",hltProcess_);
  iEvent.getByLabel(toLab,triggerObj); 
  if(!triggerObj.isValid()) return;
  
  for (unsigned int trInd=0; trInd<triggers.size(); trInd++)
    {
      bool l1pass=false;
      std::vector<l1extra::L1JetParticleRef> l1CenJets;
      std::vector<l1extra::L1JetParticleRef> l1ForJets;
      std::vector<l1extra::L1JetParticleRef> l1TauJets;
      edm::InputTag l1Tag = edm::InputTag(l1filterLabels_[trInd], "",hltProcess_);
      trigger::size_type l1filterIndex=triggerObj->filterIndex(l1Tag);

      if (l1filterIndex<triggerObj->size())
	{
	  triggerObj->getObjects(l1filterIndex, trigger::TriggerL1CenJet, l1CenJets);
	  triggerObj->getObjects(l1filterIndex, trigger::TriggerL1ForJet, l1ForJets);
	  triggerObj->getObjects(l1filterIndex, trigger::TriggerL1TauJet, l1TauJets);
	}
      
      if (l1CenJets.size()>0||l1ForJets.size()>0||l1TauJets.size()>0) 
	{
	  hL2L3acc[trInd]->Fill(1+0.001,1);
	  l1pass=true;
	}
      
      if (!l1pass) continue;
      std::vector<reco::IsolatedPixelTrackCandidateRef> l2tracks;
      edm::InputTag l2Tag = edm::InputTag(l2filterLabels_[trInd],"",hltProcess_);
      trigger::size_type l2filterIndex=triggerObj->filterIndex(l2Tag);
      if (l2filterIndex<triggerObj->size()) triggerObj->getObjects(l2filterIndex, trigger::TriggerTrack, l2tracks); 
      
      std::vector<reco::IsolatedPixelTrackCandidateRef> l3tracks;
      edm::InputTag l3Tag = edm::InputTag(l3filterLabels_[trInd], "",hltProcess_);
      trigger::size_type l3filterIndex=triggerObj->filterIndex(l3Tag);
      if (l3filterIndex<triggerObj->size()) triggerObj->getObjects(l3filterIndex, trigger::TriggerTrack, l3tracks);
      
      edm::Handle<reco::IsolatedPixelTrackCandidateCollection> l3col;
      edm::InputTag l3colTag=edm::InputTag(l3collectionLabels_[trInd],"",hltProcess_);
      
      if (l2tracks.size()>0) 
	{
	  hL2L3acc[trInd]->Fill(2+0.0001,1);
	  if (useProducerCollections_)
	    {
	      iEvent.getByLabel(l3collectionLabels_[trInd],l3col);
	      if(!l3col.isValid()) continue;
	      
	      for (reco::IsolatedPixelTrackCandidateCollection::const_iterator l3it=l3col->begin(); l3it!=l3col->end(); ++l3it)
		{
		  double drmin=100;
		  int selL2tr=-1;
		  for (unsigned int il2=0; il2<l2tracks.size(); il2++)
		    {
		      double drl2l3=reco::deltaR(l3it->eta(),l3it->phi(),l2tracks[il2]->eta(),l2tracks[il2]->phi());
		      if (drl2l3<drmin)
			{
			  drmin=drl2l3;
			  selL2tr=il2;
			}
		    }
		  if (selL2tr!=-1&&drmin<0.03&&l2tracks[selL2tr]->p()!=0) hL3candL2rat[trInd]->Fill(l3it->p()/l2tracks[selL2tr]->p(),1);
		  if (selL2tr!=-1) hL3L2trackMatch[trInd]->Fill(drmin,1);
		}
	    }
	}
      
      if (l3tracks.size()>0) hL2L3acc[trInd]->Fill(3+0.0001,1);
      
      l1extra::L1JetParticleRef maxPtJet;
      
      double l1maxPt=-1;
      for (unsigned int i=0; i<l1CenJets.size(); i++)
	{
	  if (l1CenJets[i]->pt()>l1maxPt) 
	    {
	      l1maxPt=l1CenJets[i]->pt();
	      maxPtJet=l1CenJets[i];
	    }
	}
      for (unsigned int i=0; i<l1ForJets.size(); i++)
	{
	  if (l1ForJets[i]->pt()>l1maxPt) 
	    {
	      l1maxPt=l1ForJets[i]->pt();
	      maxPtJet=l1ForJets[i];
	    }
	}
      for (unsigned int i=0; i<l1TauJets.size(); i++)
	{
	  if (l1TauJets[i]->pt()>l1maxPt) 
	    {
	      l1maxPt=l1TauJets[i]->pt();
	      maxPtJet=l1TauJets[i];
	    }
	}
      
      if (maxPtJet.isNonnull()) hL1eta[trInd]->Fill(maxPtJet->eta(),1);
      if (maxPtJet.isNonnull()) hL1phi[trInd]->Fill(maxPtJet->phi(),1);
      
      edm::Handle<reco::IsolatedPixelTrackCandidateCollection> l2col;
      edm::InputTag l2colTag=edm::InputTag(l2collectionLabels_[trInd],"",hltProcess_);
      if (useProducerCollections_)
	{
	  iEvent.getByLabel(l2collectionLabels_[trInd],l2col);
	  if(!l2col.isValid()) continue;
	  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator l2it=l2col->begin(); l2it!=l2col->end(); l2it++)
	    {
	      hL2isolationP[trInd]->Fill(l2it->maxPtPxl(),1);
	    }
	}
      
      for (unsigned int i=0; i<l2tracks.size(); i++)
	{
	  std::pair<int, int> tower=towerIndex(l2tracks[i]->eta(), l2tracks[i]->phi());
	  hL2TowerOccupancy[trInd]->Fill(tower.first,tower.second,1);
	}
      for (unsigned int i=0; i<l3tracks.size(); i++)
	{
	  std::pair<int, int> tower=towerIndex(l3tracks[i]->eta(), l3tracks[i]->phi());
	  hL3TowerOccupancy[trInd]->Fill(tower.first,tower.second,1);
	}
    }
}

void HLTMonHcalIsoTrack::beginJob()
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);

  char tmp1[100];
  char tmp2[100];
  for (unsigned int i=0; i<triggers.size(); i++)
    {
      std::sprintf(tmp1,"hL2L3acc_%s",trigNames_[i].c_str());
      std::sprintf(tmp2,"number of L1, L2 and L3 accepts; [%s]",trigNames_[i].c_str());
      MonitorElement* hL2L3accBuf=dbe_->book1D(tmp1,tmp2,3,1,4);
      hL2L3acc.push_back(hL2L3accBuf);
      hL2L3acc[i]->setTitle(tmp1);
      hL2L3acc[i]->setAxisTitle("trigger level",1);
      
      std::sprintf(tmp1,"hL2L3trackMatch_%s",trigNames_[i].c_str());
      std::sprintf(tmp2,"R from L3 object to L2 object; [%s]",trigNames_[i].c_str());
      MonitorElement* hL3L2trackMatchBuf=dbe_->book1D(tmp1,tmp2,1000,0,1);
      hL3L2trackMatch.push_back(hL3L2trackMatchBuf);
      hL3L2trackMatch[i]->setAxisTitle("R(eta,phi)",1);
      
      std::sprintf(tmp1,"hL2L3rat_%s",trigNames_[i].c_str());
      std::sprintf(tmp2,"ratio of L3 to L2 momentum measurement; [%s]",trigNames_[i].c_str());
      MonitorElement* hL3L2ratBuf=dbe_->book1D(tmp1,tmp2,1000,0,10);
      hL3candL2rat.push_back(hL3L2ratBuf);
      hL3candL2rat[i]->setAxisTitle("P_L3/P_L2",1);
    
      std::sprintf(tmp1,"hL1eta_%s",trigNames_[i].c_str());
      std::sprintf(tmp2,"eta distribution of L1 triggers; [%s]",trigNames_[i].c_str());
      MonitorElement* hL1etaBuf=dbe_->book1D(tmp1,tmp2,1000,-7,7);
      hL1eta.push_back(hL1etaBuf);
      hL1eta[i]->setAxisTitle("eta",1);
      
      std::sprintf(tmp1,"hL1phi_%s",trigNames_[i].c_str());
      std::sprintf(tmp2,"phi distribution of L1 triggers; [%s]",trigNames_[i].c_str());
      MonitorElement* hL1phiBuf=dbe_->book1D(tmp1,tmp2,1000,-4,4);
      hL1phi.push_back(hL1phiBuf);
      hL1phi[i]->setAxisTitle("phi",1);
      
      std::sprintf(tmp1,"hL2isolation_%s",trigNames_[i].c_str());
      std::sprintf(tmp2,"isolation momentum at L2; [%s]",trigNames_[i].c_str());
      MonitorElement* hL2isolationBuf=dbe_->book1D(tmp1,tmp2,1000,0,10);
      hL2isolationP.push_back(hL2isolationBuf);
      hL2isolationP[i]->setAxisTitle("max P (GeV)",1);
      
      std::sprintf(tmp1,"hL2occupancy_%s",trigNames_[i].c_str());
      std::sprintf(tmp2,"tower occupancy at L2; [%s]",trigNames_[i].c_str());
      MonitorElement* hL2TowerOccupancyBuf=dbe_->book2D(tmp1,tmp2,48,-25,25,73,0,73);
      hL2TowerOccupancy.push_back(hL2TowerOccupancyBuf);
      hL2TowerOccupancy[i]->setAxisTitle("ieta",1);
      hL2TowerOccupancy[i]->setAxisTitle("iphi",2);
      hL2TowerOccupancy[i]->getTH2F()->SetOption("colz");
      hL2TowerOccupancy[i]->getTH2F()->SetStats(kFALSE);
  
      
      std::sprintf(tmp1,"hL3occupancy_%s",trigNames_[i].c_str());
      std::sprintf(tmp2,"tower occupancy at L3; [%s]",trigNames_[i].c_str());
      MonitorElement* hL3TowerOccupancyBuf=dbe_->book2D(tmp1,tmp2,48,-25,25,73,0,73);
      hL3TowerOccupancy.push_back(hL3TowerOccupancyBuf);
      hL3TowerOccupancy[i]->setAxisTitle("ieta",1);
      hL3TowerOccupancy[i]->setAxisTitle("iphi",2);
      hL3TowerOccupancy[i]->getTH2F()->SetOption("colz");
      hL3TowerOccupancy[i]->getTH2F()->SetStats(kFALSE);
    }
}

void HLTMonHcalIsoTrack::endJob() {

if(dbe_&&saveToRootFile_) 
  {  
    dbe_->save(outRootFileName_);
  }
}

DEFINE_FWK_MODULE(HLTMonHcalIsoTrack);
