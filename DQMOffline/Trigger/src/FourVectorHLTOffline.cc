// $Id: FourVectorHLTOffline.cc,v 1.2 2008/06/18 19:37:01 berryhil Exp $
// See header file for information. 
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/Trigger/interface/FourVectorHLTOffline.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

FourVectorHLTOffline::FourVectorHLTOffline(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{
  LogDebug("FourVectorHLTOffline") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("FourVectorHLTOffline") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  
  dirname_="HLT/FourVectorHLTOffline" + 
    iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  
  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",40);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);

  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> filters = 
    iConfig.getParameter<std::vector<edm::ParameterSet> >("filters");
  for(std::vector<edm::ParameterSet>::iterator 
	filterconf = filters.begin() ; filterconf != filters.end(); 
      filterconf++) {
    std::string me  = filterconf->getParameter<std::string>("name");
    int objectType = filterconf->getParameter<unsigned int>("type");
    float ptMin = filterconf->getUntrackedParameter<double>("ptMin");
    float ptMax = filterconf->getUntrackedParameter<double>("ptMax");
    hltPaths_.push_back(PathInfo(me, objectType, ptMin, ptMax));
  }
  if ( hltPaths_.size() && plotAll_) {
    // these two ought to be mutually exclusive....
    LogInfo("FourVectorHLTOffline") << "Using both plotAll and a list. "
      "list will be ignored." ;
    hltPaths_.clear();
  }
  triggerSummaryLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
 
  
}


FourVectorHLTOffline::~FourVectorHLTOffline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLTOffline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace trigger;
  ++nev_;
  LogDebug("FourVectorHLTOffline")<< "FourVectorHLTOffline: analyze...." ;
  
  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "Summary HLT objects not found, "
      "skipping event"; 
    return;
  }
  

  

  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());


  if ( plotAll_ ) {
    // loop over all trigger paths, book new ME's for previously unseen paths, 
    // and fill ME's with online and offline 4-vectors

    for ( size_t ia = 0; ia < triggerObj->sizeFilters(); ++ ia) {
      // get the path name to encode ME name
      std::string name = triggerObj->filterTag(ia).encode();
      // KLUDGE:  trigger object type is keyed off of filterId 
      // instead of trigger object pdgId (becuase it is buggy in 2_0)
       Vids  idtype = triggerObj->filterIds(ia);
       int triggertype = idtype[0];
   
      PathInfoCollection::iterator pic =  hltPaths_.find(name);
      if ( pic == hltPaths_.end() ) {
	// doesn't exist - add it
	MonitorElement *etOn(0), *etaOn(0), *phiOn(0), *etavsphiOn(0);
	MonitorElement *etOff(0), *etaOff(0), *phiOff(0), *etavsphiOff(0);
	std::string histoname(name+"_etOn");
	std::string title(name+" E_t online");
	etOn =  dbe_->book1D(histoname.c_str(),
			  title.c_str(),nBins_, 0, 100);
      
	histoname = name+"_etOff";
	title = name+" E_t offline";
	etOff =  dbe_->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 0, 100);

	histoname = name+"_etaOn";
	title = name+" #eta online";
	etaOn =  dbe_->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);
      
	histoname = name+"_etaOff";
	title = name+" #eta offline";
	etaOff =  dbe_->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = name+"_phiOn";
	title = name+" #phi online";
	phiOn =  dbe_->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = name+"_phiOff";
	title = name+" #phi offline";
	phiOff =  dbe_->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);
      
	histoname = name+"_etaphiOn";
	title = name+" #eta vs #phi online";
	etavsphiOn =  dbe_->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = name+"_etaphiOff";
	title = name+" #eta vs #phi offline";
	etavsphiOff =  dbe_->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);
      
	// no idea how to get the bin boundries in this mode
	PathInfo e(name,0, etOn, etaOn, phiOn, etavsphiOn, etOff, etaOff, phiOff, etavsphiOff, 0,100);
	hltPaths_.push_back(e);  
	pic = hltPaths_.begin() + hltPaths_.size()-1;
      }
      const trigger::Keys & k = triggerObj->filterKeys(ia);
      // loop over trigger objects and fill online 4-vectors
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	pic->getEtOnHisto()->Fill(toc[*ki].pt());
	pic->getEtaOnHisto()->Fill(toc[*ki].eta());
	pic->getPhiOnHisto()->Fill(toc[*ki].phi());
	pic->getEtaVsPhiOnHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
      }  
      // for photon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerPhoton)
	{

         edm::Handle<reco::PhotonCollection> photonHandle;
         iEvent.getByLabel("photons",photonHandle);

         if(!photonHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "photonHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::PhotonCollection photonCollection = *(photonHandle.product());

         for (reco::PhotonCollection::const_iterator photonIter=photonCollection.begin(); photonIter!=photonCollection.end(); photonIter++)
         {
	  pic->getEtOffHisto()->Fill((*photonIter).pt());
	  pic->getEtaOffHisto()->Fill((*photonIter).eta());
	  pic->getPhiOffHisto()->Fill((*photonIter).phi());
	  pic->getEtaVsPhiOffHisto()->Fill((*photonIter).eta(),(*photonIter).phi());
         }
	}
      // for electron triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerElectron)
	{
	  //	  std::cout << "Electron trigger" << std::endl;
         edm::Handle<reco::PixelMatchGsfElectronCollection> gsfElectrons;
         iEvent.getByLabel("pixelMatchGsfElectrons",gsfElectrons); 
         if(!gsfElectrons.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "gsfElectrons not found, "
            "skipping event"; 
            return;
         }
         for (reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++)
         {
	  pic->getEtOffHisto()->Fill(gsfIter->pt());
	  pic->getEtaOffHisto()->Fill(gsfIter->eta());
	  pic->getPhiOffHisto()->Fill(gsfIter->phi());
	  pic->getEtaVsPhiOffHisto()->Fill(gsfIter->eta(), gsfIter->phi());
         }
	}
      // for muon triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerMuon)
	{
         edm::Handle<reco::MuonCollection> muonHandle;
         iEvent.getByLabel("muons",muonHandle);

         if(!muonHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "muonHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::MuonCollection muonCollection = *(muonHandle.product());

         for (reco::MuonCollection::const_iterator muonIter=muonCollection.begin(); muonIter!=muonCollection.end(); muonIter++)
         {
	  pic->getEtOffHisto()->Fill((*muonIter).pt());
	  pic->getEtaOffHisto()->Fill((*muonIter).eta());
	  pic->getPhiOffHisto()->Fill((*muonIter).phi());
	  pic->getEtaVsPhiOffHisto()->Fill((*muonIter).eta(),(*muonIter).phi());
         }
	}
      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau)
	{
         edm::Handle<reco::CaloTauCollection> tauHandle;
         iEvent.getByLabel("caloRecoTauProducer",tauHandle);

         if(!tauHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "tauHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::CaloTauCollection tauCollection = *(tauHandle.product());

         for (reco::CaloTauCollection::const_iterator tauIter=tauCollection.begin(); tauIter!=tauCollection.end(); tauIter++)
         {
	  pic->getEtOffHisto()->Fill((*tauIter).pt());
	  pic->getEtaOffHisto()->Fill((*tauIter).eta());
	  pic->getPhiOffHisto()->Fill((*tauIter).phi());
	  pic->getEtaVsPhiOffHisto()->Fill((*tauIter).eta(),(*tauIter).phi());
         }
	}
      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet)
	{
         edm::Handle<reco::CaloJetCollection> jetHandle;
         iEvent.getByLabel("iterativeCone5CaloJets",jetHandle);

         if(!jetHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "jetHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::CaloJetCollection jetCollection = *(jetHandle.product());

         for (reco::CaloJetCollection::const_iterator jetIter=jetCollection.begin(); jetIter!=jetCollection.end(); jetIter++)
         {
	  pic->getEtOffHisto()->Fill((*jetIter).pt());
	  pic->getEtaOffHisto()->Fill((*jetIter).eta());
	  pic->getPhiOffHisto()->Fill((*jetIter).phi());
	  pic->getEtaVsPhiOffHisto()->Fill((*jetIter).eta(),(*jetIter).phi());
         }
	}
      // for bjet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerBJet)
	{
	}
      // for met triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerMET)
	{
         edm::Handle<reco::CaloMETCollection> metHandle;
         iEvent.getByLabel("met",metHandle);

         if(!metHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOffline") << "metHandle not found, "
            "skipping event"; 
            return;
         }
         const reco::CaloMETCollection metCollection = *(metHandle.product());

         for (reco::CaloMETCollection::const_iterator metIter=metCollection.begin(); metIter!=metCollection.end(); metIter++)
         {
	  pic->getEtOffHisto()->Fill((*metIter).pt());
	  pic->getEtaOffHisto()->Fill((*metIter).eta());
	  pic->getPhiOffHisto()->Fill((*metIter).phi());
	  pic->getEtaVsPhiOffHisto()->Fill((*metIter).eta(),(*metIter).phi());
         }
	}
      else
	{
	  //	  std::cout << "Unrecognized trigger" << std::endl;
        }
    }

  }
  else { // not plotAll_
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	v!= hltPaths_.end(); ++v ) {
      const int index = triggerObj->filterIndex(v->getName());
      if ( index >= triggerObj->sizeFilters() ) {
	continue; // not in this event
      }
      LogDebug("FourVectorHLTOffline") << "filling ... " ;
      const trigger::Keys & k = triggerObj->filterKeys(index);
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	v->getEtOnHisto()->Fill(toc[*ki].pt());
	v->getEtaOnHisto()->Fill(toc[*ki].eta());
	v->getPhiOnHisto()->Fill(toc[*ki].phi());
	v->getEtaVsPhiOnHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
      }  
    }
  }
}


// -- method called once each job just before starting event loop  --------
void 
FourVectorHLTOffline::beginJob(const edm::EventSetup&)
{
  nev_ = 0;
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);

    if ( ! plotAll_ ) {
      for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {
	MonitorElement *etOn, *etaOn, *phiOn, *etavsphiOn=0;
	MonitorElement *etOff, *etaOff, *phiOff, *etavsphiOff=0;
	std::string histoname(v->getName()+"_etOn");
	std::string title(v->getName()+" E_t online");
	etOn =  dbe->book1D(histoname.c_str(),
			  title.c_str(),nBins_,
			  v->getPtMin(),
			  v->getPtMax());
      
	histoname = v->getName()+"_etOff";
	title = v->getName()+" E_t offline";
	etOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = v->getName()+"_etaOn";
	title = v->getName()+" #eta online";
	etaOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = v->getName()+"_etaOff";
	title = v->getName()+" #eta offline";
	etaOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = v->getName()+"_phiOn";
	title = v->getName()+" #phi online";
	phiOn =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = v->getName()+"_phiOff";
	title = v->getName()+" #phi offline";
	phiOff =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);
 

	histoname = v->getName()+"_etaphiOn";
	title = v->getName()+" #eta vs #phi online";
	etavsphiOn =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = v->getName()+"_etaphiOff";
	title = v->getName()+" #eta vs #phi offline";
	etavsphiOff =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	v->setHistos( etOn, etaOn, phiOn, etavsphiOn, etOff, etaOff, phiOff, etavsphiOff);
      } 
    } // ! plotAll_ - for plotAll we discover it during the event
  }
}

// - method called once each job just after ending the event loop  ------------
void 
FourVectorHLTOffline::endJob() 
{
   LogInfo("FourVectorHLTOffline") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLTOffline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOffline") << "beginRun, run " << run.id();
}

/// EndRun
void FourVectorHLTOffline::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOffline") << "endRun, run " << run.id();
}
