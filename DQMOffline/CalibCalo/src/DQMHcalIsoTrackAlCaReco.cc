// -*- C++ -*-
//
// Package:    DQMOffline/CalibCalo
// Class:      DQMHcalIsoTrackAlCaReco
// 
/**\class DQMHcalIsoTrackAlCaReco DQMHcalIsoTrackAlCaReco.cc DQMOffline/CalibCalo/src/DQMHcalIsoTrackAlCaReco.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory SAFRONOV
//         Created:  Tue Oct  14 16:10:31 CEST 2008
//         Modified: Tue Mar   3 16:10:31 CEST 2015
//
//

//#define DebugLog
// system include files
#include <cmath>

// user include files

#include "DQMOffline/CalibCalo/src/DQMHcalIsoTrackAlCaReco.h"

DQMHcalIsoTrackAlCaReco::DQMHcalIsoTrackAlCaReco(const edm::ParameterSet& iConfig) {
  folderName_  = iConfig.getParameter<std::string>("FolderName");
  l1FilterTag_ = iConfig.getParameter<std::vector<std::string> >("L1FilterLabel");
  hltFilterTag_= iConfig.getParameter<std::vector<std::string> >("HltFilterLabels");
  type_        = iConfig.getParameter<std::vector<int> >("TypeFilter");
  labelTrigger_= iConfig.getParameter<edm::InputTag>("TriggerLabel");
  labelTrack_  = iConfig.getParameter<edm::InputTag>("TracksLabel");
  pThr_        = iConfig.getUntrackedParameter<double>("pThrL3",0);

  nTotal_     = nHLTaccepts_ = 0;
  tokTrigger_ = consumes<trigger::TriggerEvent>(labelTrigger_);
  tokTrack_   = consumes<reco::HcalIsolatedTrackCandidateCollection>(labelTrack_);
#ifdef DebugLog
  std::cout << "Folder " << folderName_ << " Input Tag for Trigger "
	    << labelTrigger_ << " track " << labelTrack_ << " threshold "
	    << pThr_ << " with " << l1FilterTag_.size() << " level 1 and "
	    << hltFilterTag_.size() << " hlt filter tags" << std::endl;
  for (unsigned int k=0; k<l1FilterTag_.size(); ++k) 
    std::cout << "L1FilterTag[" << k << "] " << l1FilterTag_[k] << std::endl;
  for (unsigned int k=0; k<hltFilterTag_.size(); ++k)
    std::cout << "HLTFilterTag[" << k << "] " << hltFilterTag_[k] << std::endl;
#endif
}

DQMHcalIsoTrackAlCaReco::~DQMHcalIsoTrackAlCaReco() {}

void DQMHcalIsoTrackAlCaReco::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

#ifdef DebugLog
  std::cout << "Enters DQMHcalIsoTrackAlCaReco::analyze " << std::endl;
#endif
  nTotal_++;
  bool accept(false);
  edm::Handle<trigger::TriggerEvent> trEv;
  iEvent.getByToken(tokTrigger_,trEv);
  
  edm::Handle<reco::HcalIsolatedTrackCandidateCollection> recoIsoTracks;
  iEvent.getByToken(tokTrack_,recoIsoTracks);

  if (trEv.isValid()) {
    const trigger::TriggerObjectCollection& TOCol(trEv->getObjects());
    const trigger::size_type nFilt(trEv->sizeFilters());

    //plots for L1 trigger
    for (unsigned int k=0; k<l1FilterTag_.size(); ++k) {
      double etaTrigl1(-10000), phiTrigl1(-10000), ptMaxl1(0);
      for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) {
	if ((trEv->filterTag(iFilt).label()).find(l1FilterTag_[k].c_str()) !=
	    std::string::npos) {
	  trigger::Keys KEYSl1=trEv->filterKeys(iFilt); 
	  trigger::size_type nRegl1=KEYSl1.size();
	  for (trigger::size_type iReg=0; iReg<nRegl1; iReg++) {
	    const trigger::TriggerObject& TObj(TOCol[KEYSl1[iReg]]);
	    if (TObj.pt()>ptMaxl1) {
	      etaTrigl1=TObj.eta();
	      phiTrigl1=TObj.phi();
	      ptMaxl1=TObj.pt();
	    }
	  }
	}
      }
#ifdef DebugLog
      std::cout << "For L1 trigger type " << k << " pt " << ptMaxl1 << " eta "
		<< etaTrigl1 << " phi " << phiTrigl1 << std::endl;
#endif
      if (ptMaxl1 > 0) {
	hL1Pt_[k]->Fill(ptMaxl1);
	hL1Eta_[k]->Fill(etaTrigl1);
	hL1phi_[k]->Fill(phiTrigl1);
      }
    }

    //Now make plots for hlt objects
    for (unsigned l=0; l<hltFilterTag_.size(); l++) {
      for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) {
	if ((trEv->filterTag(iFilt).label()).find(hltFilterTag_[l].c_str()) !=
	    std::string::npos) {
	  trigger::Keys KEYS=trEv->filterKeys(iFilt);
	  trigger::size_type nReg=KEYS.size();
	  
	  //checks with IsoTrack trigger results
	  for (trigger::size_type iReg=0; iReg<nReg; iReg++) {
	    const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
#ifdef DebugLog
	    std::cout << "HLT Filter Tag " << l << " trigger " << iFilt
		      << " object " << iReg << " p " << TObj.p() << " pointer "
		      << indexH_[l] << ":" << hHltP_[indexH_[l]] << ":"
		      << hHltEta_[indexH_[l]] << ":" << hHltPhi_[indexH_[l]]
		      << std::endl;
#endif
	    if (TObj.p()>=pThr_) {
	      hHltP_[indexH_[l]]  ->Fill(TObj.p());
	      hHltEta_[indexH_[l]]->Fill(TObj.eta());
	      hHltPhi_[indexH_[l]]->Fill(TObj.phi());
	      if (ifL3_[l]) accept = true;
	      if (recoIsoTracks.isValid() && ifL3_[l]) {
#ifdef DebugLog
		std::cout << "Look for reconstructed object" << std::endl;
#endif
		double minRecoL3dist(1000), pt(1000);
		reco::HcalIsolatedTrackCandidateCollection::const_iterator mrtr;
		for (mrtr=recoIsoTracks->begin(); mrtr!=recoIsoTracks->end(); 
		     mrtr++)  {
		  double R = deltaR(mrtr->eta(),mrtr->phi(),TObj.eta(),TObj.phi()); 
		  if (R<minRecoL3dist) {
		    minRecoL3dist = R;
		    pt            = mrtr->pt();
		  }
		}
#ifdef DebugLog
		std::cout << "Minimum R " << minRecoL3dist << " pt " << pt
			  << ":" << TObj.pt() << std::endl;
#endif
		hL3Dr_->Fill(minRecoL3dist);
		if (minRecoL3dist<0.02) hL3Rat_->Fill(TObj.pt()/pt);
	      }
	    }
	  }
	}
      }
    }
  }
  
  //general distributions
  if (recoIsoTracks.isValid()) {
    for (reco::HcalIsolatedTrackCandidateCollection::const_iterator itr=recoIsoTracks->begin(); itr!=recoIsoTracks->end(); itr++) {
      hMaxP_->Fill(itr->maxP());
      hEnEcal_->Fill(itr->energyEcal());
      std::pair<int,int> etaphi = itr->towerIndex();
      hIeta_->Fill(etaphi.first);
      hIphi_->Fill(etaphi.second);
#ifdef DebugLog
      std::cout << "Reco track p " << itr->p() << " eta|phi " << etaphi.first
		<< "|" << etaphi.second << " maxP " << itr->maxP() << " EcalE "
		<< itr->energyEcal() << " pointers " << hHltP_[3] << ":"
		<< hHltEta_[3] << ":" << hHltPhi_[3] << std::endl;
#endif
      if (itr->p()>=pThr_) {
	hHltP_[3]  ->Fill(itr->p());
	hHltEta_[3]->Fill(itr->eta());
	hHltPhi_[3]->Fill(itr->phi());
      }
      double etaAbs = std::abs(itr->eta());
      hOffP_[0]->Fill(itr->p());
      for (unsigned int l=1; l<etaRange_.size(); l++) {
	if (etaAbs >= etaRange_[l-1] && etaAbs < etaRange_[l]) {
#ifdef DebugLog
	  std::cout << "Range " << l << " p " << itr->p() <<  " pointer "
		    << hOffP_[l] << std::endl;
#endif
	  hOffP_[l]->Fill(itr->p());
	  break;
	}
      }
    }
  }

  if (accept) nHLTaccepts_++;
#ifdef DebugLog
  std::cout << "Accept " << accept << std::endl;
#endif
}

void DQMHcalIsoTrackAlCaReco::bookHistograms(DQMStore::IBooker &iBooker,
					     edm::Run const &, 
					     edm::EventSetup const & ) {

  iBooker.setCurrentFolder(folderName_);
#ifdef DebugLog
  std::cout << "Set the folder to " << folderName_ << std::endl;
#endif
  char name[100], title[200];
  for (unsigned int k=0; k<l1FilterTag_.size(); ++k) {
    sprintf (name, "hp%s", l1FilterTag_[k].c_str());
    sprintf (title, "p_T of L1 object for %s", l1FilterTag_[k].c_str());
    hL1Pt_.push_back(iBooker.book1D(name,title,1000,0,1000));
    hL1Pt_[k]->setAxisTitle("p_T (GeV)", 1);
    sprintf (name, "heta%s", l1FilterTag_[k].c_str());
    sprintf (title, "#eta of L1 object for %s", l1FilterTag_[k].c_str());
    hL1Eta_.push_back(iBooker.book1D(name,title,100,-2.5,2.5));
    hL1Eta_[k]->setAxisTitle("#eta",1);
    sprintf (name, "hphi%s", l1FilterTag_[k].c_str());
    sprintf (title, "#phi of L1 object for %s", l1FilterTag_[k].c_str());
    hL1phi_.push_back(iBooker.book1D(name,title,100,-3.2,3.2));
    hL1phi_[k]->setAxisTitle("#phi",1);
  }
#ifdef DebugLog
  std::cout << "Booked the L1 histos" << std::endl;
#endif
  
  std::string types[4] = {"L2","L2x","L3","Off"};
  for (unsigned int l=0; l<4; l++) {
    sprintf (name, "hp%s", types[l].c_str());
    sprintf (title,"Momentum of %s object", types[l].c_str());
    hHltP_.push_back(iBooker.book1D(name,title,200,0,1000));
    hHltP_[l]->setAxisTitle("p (GeV)", 1);
    sprintf (name, "heta%s", types[l].c_str());
    sprintf (title,"#eta of %s object", types[l].c_str());
    hHltEta_.push_back(iBooker.book1D(name,title,16,-2,2));
    hHltEta_[l]->setAxisTitle("#eta",1);
    sprintf (name, "hphi%s", types[l].c_str());
    sprintf (title,"#phi of %s object", types[l].c_str());
    hHltPhi_.push_back(iBooker.book1D(name,title,16,-3.2,3.2));
    hHltPhi_[l]->setAxisTitle("#phi",1);
  }
#ifdef DebugLog
  std::cout << "Booked the HLT histos" << std::endl;
#endif
  sprintf (title,"Distance of offline track from L3 object");
  hL3Dr_ = (iBooker.book1D("hDRL3",title,40,0,0.2));
  hL3Dr_->setAxisTitle("R(#eta,#phi)",1);
  sprintf (title,"Ratio of p L3/Offline");
  hL3Rat_ = (iBooker.book1D("hRatL3",title,500,0,3));
  indexH_.clear(); ifL3_.clear();
  for (unsigned int l=0; l<hltFilterTag_.size(); l++) {
    unsigned int indx = (type_[l] >= 0 && type_[l] < 3) ? type_[l] : 0;
    indexH_.push_back(indx);
    ifL3_.push_back(indx==2);
#ifdef DebugLog
    std::cout << "Filter[" << l << "] " << hltFilterTag_[l] << " type "
	      << type_[l] << " index " << indexH_[l] << " L3? " << ifL3_[l]
	      << std::endl;
#endif
  }
  
  double etaV[6] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
  for (unsigned int k=0; k<6; ++k) {
    sprintf (name, "hOffP%d", k);
    if (k == 0) {
      sprintf (title, "p of AlCaReco object (All)");
    } else {
      sprintf (title, "p of AlCaReco object (%3.1f < |#eta| < %3.1f)",etaV[k-1],etaV[k]);
    }
    etaRange_.push_back(etaV[k]);
    hOffP_.push_back(iBooker.book1D(name,title,1000,0,1000));
    hOffP_[k]->setAxisTitle("E (GeV)",1);
  }
  hMaxP_ = iBooker.book1D("hChgIsol","Energy for charge isolation",110,-10,100);
  hMaxP_->setAxisTitle("p (GeV)",1);
  hEnEcal_ = iBooker.book1D("hEnEcal","Energy in ECAL",100,0,20);
  hEnEcal_->setAxisTitle("E (GeV)",1);
  hIeta_   = iBooker.book1D("hIEta","i#eta for HCAL tower",90,-45,45);
  hIeta_->setAxisTitle("i#eta",1);
  hIphi_   = iBooker.book1D("hIPhi","i#phi for HCAL tower",72,0,72);
  hIphi_->setAxisTitle("i#phi",1);
#ifdef DebugLog
  std::cout << "End of booking histograms" << std::endl;
#endif
}
