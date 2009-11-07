// $Id: FourVectorHLTOnline.cc,v 1.8 2009/11/05 13:31:52 rekovic Exp $
// See header file for information. 
#include "TMath.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/FourVectorHLTOnline.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include <vector>

using namespace edm;
using namespace std;

FourVectorHLTOnline::FourVectorHLTOnline(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{
  LogDebug("FourVectorHLTOnline") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("FourVectorHLTOnline") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  dirname_ = iConfig.getUntrackedParameter("dirname",
					   std::string("HLT/FourVector/"));
  //dirname_ +=  iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  processname_ = iConfig.getParameter<std::string>("processname");

  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",40);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);
     // this is the list of paths to look at.
     std::vector<edm::ParameterSet> paths = 
     iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");
     for(std::vector<edm::ParameterSet>::iterator 
	pathconf = paths.begin() ; pathconf != paths.end(); 
      pathconf++) {
       std::pair<std::string, std::string> custompathnamepair;
       custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
       custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
       custompathnamepairs_.push_back(custompathnamepair);
       //    customdenompathnames_.push_back(pathconf->getParameter<std::string>("denompathname"));  
       // custompathnames_.push_back(pathconf->getParameter<std::string>("pathname"));  
    }

  if (hltPaths_.size() > 0)
    {
      // book a histogram of scalers
     scalersSelect = dbe_->book1D("selectedScalers","Selected Scalers", hltPaths_.size(), 0.0, (double)hltPaths_.size());
    }

 
  triggerSummaryLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerResultsLabel");

  electronEtaMax_ = iConfig.getUntrackedParameter<double>("electronEtaMax",2.5);
  electronEtMin_ = iConfig.getUntrackedParameter<double>("electronEtMin",3.0);
  electronDRMatch_  =iConfig.getUntrackedParameter<double>("electronDRMatch",0.3); 

  muonEtaMax_ = iConfig.getUntrackedParameter<double>("muonEtaMax",2.5);
  muonEtMin_ = iConfig.getUntrackedParameter<double>("muonEtMin",3.0);
  muonDRMatch_  =iConfig.getUntrackedParameter<double>("muonDRMatch",0.3); 

  tauEtaMax_ = iConfig.getUntrackedParameter<double>("tauEtaMax",2.5);
  tauEtMin_ = iConfig.getUntrackedParameter<double>("tauEtMin",3.0);
  tauDRMatch_  =iConfig.getUntrackedParameter<double>("tauDRMatch",0.3); 

  jetEtaMax_ = iConfig.getUntrackedParameter<double>("jetEtaMax",5.0);
  jetEtMin_ = iConfig.getUntrackedParameter<double>("jetEtMin",10.0);
  jetDRMatch_  =iConfig.getUntrackedParameter<double>("jetDRMatch",0.3); 

  bjetEtaMax_ = iConfig.getUntrackedParameter<double>("bjetEtaMax",2.5);
  bjetEtMin_ = iConfig.getUntrackedParameter<double>("bjetEtMin",10.0);
  bjetDRMatch_  =iConfig.getUntrackedParameter<double>("bjetDRMatch",0.3); 

  photonEtaMax_ = iConfig.getUntrackedParameter<double>("photonEtaMax",2.5);
  photonEtMin_ = iConfig.getUntrackedParameter<double>("photonEtMin",3.0);
  photonDRMatch_  =iConfig.getUntrackedParameter<double>("photonDRMatch",0.3); 

  trackEtaMax_ = iConfig.getUntrackedParameter<double>("trackEtaMax",2.5);
  trackEtMin_ = iConfig.getUntrackedParameter<double>("trackEtMin",3.0);
  trackDRMatch_  =iConfig.getUntrackedParameter<double>("trackDRMatch",0.3); 

  metMin_ = iConfig.getUntrackedParameter<double>("metMin",10.0);
  htMin_ = iConfig.getUntrackedParameter<double>("htMin",10.0);
  sumEtMin_ = iConfig.getUntrackedParameter<double>("sumEtMin",10.0);
  
}


FourVectorHLTOnline::~FourVectorHLTOnline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLTOnline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace trigger;
  ++nev_;
  // LogDebug("FourVectorHLTOnline")<< "FourVectorHLTOnline: analyze...." ;
  
  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
   iEvent.getByLabel(triggerResultsLabelFU,triggerResults);
  if(!triggerResults.isValid()) {
    edm::LogInfo("FourVectorHLTOnline") << "TriggerResults not found, "
      "skipping event"; 
    return;
   }
  }
  TriggerNames triggerNames(*triggerResults);  
  int npath = triggerResults->size();

  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) {
    edm::InputTag triggerSummaryLabelFU(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(), "FU");
   iEvent.getByLabel(triggerSummaryLabelFU,triggerObj);
  if(!triggerObj.isValid()) {
    edm::LogInfo("FourVectorHLTOnline") << "TriggerEvent not found, "
      "skipping event"; 
    return;
   }
  }
  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());

    for(PathInfoCollection::iterator v = hltPaths_.begin();
	v!= hltPaths_.end(); ++v ) 
{ 

  int NOn = 0;
  int NL1 = 0;
  int NL1On = 0;
  int NL1OnUM = 0;

  // did we pass the denomPath?
  bool denompassed = false;  
  for(int i = 0; i < npath; ++i) {
     if (triggerNames.triggerName(i).find(v->getDenomPath()) != std::string::npos && triggerResults->accept(i))
       { 
        denompassed = true;
        break;
       }
  }

  if (denompassed)
    {  


      int triggertype = 0;     
      triggertype = v->getObjectType();

      bool l1accept = false;
      edm::InputTag l1testTag(v->getl1Path(),"",processname_);
      const int l1index = triggerObj->filterIndex(l1testTag);
      if ( l1index >= triggerObj->sizeFilters() ) {
        edm::LogInfo("FourVectorHLTOnline") << "no index "<< l1index << " of that name " << v->getl1Path() << "\t" << "\t" << l1testTag;
	continue; // not in this event
      }

      const trigger::Vids & idtype = triggerObj->filterIds(l1index);
      const trigger::Keys & l1k = triggerObj->filterKeys(l1index);
      l1accept = l1k.size() > 0;
      //if (l1k.size() == 0) cout << v->getl1Path() << endl;
      //l1accept = true;

      // for muon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu){

        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
	    if (*idtypeiter == trigger::TriggerL1Mu)
	      {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (fabs(toc[*ki].eta()) <= muonEtaMax_ && toc[*ki].pt() >= muonEtMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }


	   }
            ++idtypeiter;
	   }
         }
      }

      // for electron triggers, loop over and fill offline 4-vectors
     else if (triggertype == trigger::TriggerElectron || triggertype == trigger::TriggerL1NoIsoEG || triggertype == trigger::TriggerL1IsoEG)
	{

	  //	  std::cout << "Electron trigger" << std::endl;

        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
            if ( *idtypeiter == trigger::TriggerL1IsoEG || *idtypeiter == trigger::TriggerL1NoIsoEG ) {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (fabs(toc[*ki].eta()) <= electronEtaMax_ && toc[*ki].pt() >= electronEtMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }


	   }
            ++idtypeiter;
	   }
         }
	}
    

      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau || triggertype == trigger::TriggerL1TauJet)
	{


        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
	    if (*idtypeiter == trigger::TriggerL1TauJet || *idtypeiter == trigger::TriggerL1ForJet)
	      {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (fabs(toc[*ki].eta()) <= tauEtaMax_ && toc[*ki].pt() >= tauEtMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }


	   }
            ++idtypeiter;
	   }
         }

    }



      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet || triggertype == trigger::TriggerL1CenJet || triggertype == trigger::TriggerL1ForJet)
	{

        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
	    if (*idtypeiter == trigger::TriggerL1TauJet || *idtypeiter == trigger::TriggerL1ForJet || *idtypeiter == trigger::TriggerL1CenJet)
	      {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (fabs(toc[*ki].eta()) <= jetEtaMax_ && toc[*ki].pt() >= jetEtMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }


	   }
            ++idtypeiter;
	   }
         }

	}

      // for bjet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerBJet)
	{ 
	
        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
	    if (*idtypeiter == trigger::TriggerL1TauJet || *idtypeiter == trigger::TriggerL1ForJet || *idtypeiter == trigger::TriggerL1CenJet)
	      {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (fabs(toc[*ki].eta()) <= bjetEtaMax_ && toc[*ki].pt() >= bjetEtMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }


	   }
            ++idtypeiter;
	   }
         }

	}
      // for met triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM)
	{


        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
	    if (*idtypeiter == trigger::TriggerL1ETM)
	      {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (toc[*ki].pt() >= metMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }


	      }

            ++idtypeiter;
	   }
         }

	}
      else if (triggertype == trigger::TriggerTET || triggertype == trigger::TriggerL1ETT)
	{


        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
	    if (*idtypeiter == trigger::TriggerL1ETT)
	      {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (toc[*ki].pt() >= sumEtMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }


	   }
            ++idtypeiter;
	   }
         }

	}
      // for photon triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerPhoton)
	{

        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
	    if (*idtypeiter == trigger::TriggerL1IsoEG || *idtypeiter == trigger::TriggerL1NoIsoEG)
	      {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (fabs(toc[*ki].eta()) <= photonEtaMax_ && toc[*ki].pt() >= photonEtMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }


	   }
            ++idtypeiter;
	   }
         }
	}

      // for IsoTrack triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerTrack)
	{

        if (l1accept)
         {
          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator ki = l1k.begin(); ki !=l1k.end(); ++ki ) {
	    if (*idtypeiter == trigger::TriggerL1CenJet || *idtypeiter == trigger::TriggerL1ForJet || *idtypeiter == trigger::TriggerL1TauJet)
	      {
	    //	cout << v->getl1Path() << "\t" << *idtypeiter << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
  	    if (fabs(toc[*ki].eta()) <= trackEtaMax_ && toc[*ki].pt() >= trackEtMin_)
             { 
	      NL1++;    
              v->getL1EtL1Histo()->Fill(toc[*ki].pt());
	      v->getL1EtaVsL1PhiL1Histo()->Fill(toc[*ki].eta(), toc[*ki].phi());
	     }

	   }
            ++idtypeiter;
	   }
         }

	}

    // did we pass the numerator path?
  bool numpassed = false;
  for(int i = 0; i < npath; ++i) {
     if (triggerNames.triggerName(i) == v->getPath() && triggerResults->accept(i)) numpassed = true;
  }

  if (numpassed)
    { 
 
      if (!l1accept) {
            edm::LogInfo("FourVectorHLTOnline") << "l1 seed path not accepted for hlt path "<< v->getPath() << "\t" << v->getl1Path();
      }
    // ok plot On, L1On, OnOff, and OnMc objects

    // fill scaler histograms
      edm::InputTag filterTag = v->getTag();

	// loop through indices and see if the filter is on the list of filters used by this path
      
    if (v->getLabel() == "dummy"){
        const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	//loop over labels
        for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++)          
	 {
	   //cout << v->getPath() << "\t" << *labelIter << endl;
           // last match wins...
	   edm::InputTag testTag(*labelIter,"",processname_);
	   //           cout << v->getPath() << "\t" << testTag.label() << "\t" << testTag.process() << endl;
           int testindex = triggerObj->filterIndex(testTag);
           if ( !(testindex >= triggerObj->sizeFilters()) ) {
	     //cout << "found one! " << v->getPath() << "\t" << testTag.label() << endl; 
            filterTag = testTag; v->setLabel(*labelIter);}
	 }
         }
	
      const int index = triggerObj->filterIndex(filterTag);
      if ( index >= triggerObj->sizeFilters() ) {
	//        cout << "WTF no index "<< index << " of that name "
	//	     << filterTag << endl;
	continue; // not in this event
      }
      //LogDebug("FourVectorHLTOnline") << "filling ... " ;
      const trigger::Keys & k = triggerObj->filterKeys(index);
      //      const trigger::Vids & idtype = triggerObj->filterIds(index);
      // assume for now the first object type is the same as all objects in the collection
      //    cout << filterTag << "\t" << idtype.size() << "\t" << k.size() << endl;
      //     cout << "path " << v->getPath() << " trigger type "<<triggertype << endl;
      //if (k.size() > 0) v->getNOnHisto()->Fill(k.size());
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
         
        double tocEtaMax = 2.5;
        double tocEtMin = 3.0;
        if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu) 
	  {
	    tocEtaMax = muonEtaMax_; tocEtMin = muonEtMin_;
	  }
        else if (triggertype == trigger::TriggerElectron || triggertype == trigger::TriggerL1NoIsoEG || triggertype == trigger::TriggerL1IsoEG )
	  {
	    tocEtaMax = electronEtaMax_; tocEtMin = electronEtMin_;
	  }
        else if (triggertype == trigger::TriggerTau || triggertype == trigger::TriggerL1TauJet )
	  {
	    tocEtaMax = tauEtaMax_; tocEtMin = tauEtMin_;
	  }
        else if (triggertype == trigger::TriggerJet || triggertype == trigger::TriggerL1CenJet || triggertype == trigger::TriggerL1ForJet )
	  {
	    tocEtaMax = jetEtaMax_; tocEtMin = jetEtMin_;
	  }
        else if (triggertype == trigger::TriggerBJet)
	  {
	    tocEtaMax = bjetEtaMax_; tocEtMin = bjetEtMin_;
	  }
        else if (triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM )
	  {
	    tocEtaMax = 999.0; tocEtMin = metMin_;
	  }
        else if (triggertype == trigger::TriggerPhoton)
	  {
	    tocEtaMax = photonEtaMax_; tocEtMin = photonEtMin_;
	  }
        else if (triggertype == trigger::TriggerTrack)
	  {
	    tocEtaMax = trackEtaMax_; tocEtMin = trackEtMin_;
	  }

        if (fabs(toc[*ki].eta()) <= tocEtaMax && toc[*ki].pt() >= tocEtMin)
	  {
	NOn++;    
        v->getOnEtOnHisto()->Fill(toc[*ki].pt());
	v->getOnEtaVsOnPhiOnHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	  }
	//	  cout << "pdgId "<<toc[*ki].id() << endl;
      // for muon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu)
	{


          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1Mu)
	      {
	   if (reco::deltaR(toc[*l1ki].eta(),toc[*l1ki].phi(),toc[*ki].eta(),toc[*ki].phi()) < muonDRMatch_ && fabs(toc[*l1ki].eta()) <= muonEtaMax_ && toc[*l1ki].pt() >= muonEtMin_ )
            {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
	   if (NOn==1 && fabs(toc[*l1ki].eta()) <= muonEtaMax_ && toc[*l1ki].pt() >= muonEtMin_ )
            {
	     NL1OnUM++;
	     v->getL1EtL1OnUMHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnUMHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
              }
	    ++idtypeiter;
	  }

      }

      // for electron triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerElectron || triggertype == trigger::TriggerL1IsoEG || triggertype == trigger::TriggerL1NoIsoEG )
	{
	  //	  std::cout << "Electron trigger" << std::endl;

          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1IsoEG || *idtypeiter == trigger::TriggerL1NoIsoEG)
	      {
	   if (reco::deltaR(toc[*l1ki].eta(),toc[*l1ki].phi(),toc[*ki].eta(),toc[*ki].phi()) < electronDRMatch_ && fabs(toc[*l1ki].eta()) <= electronEtaMax_ && toc[*l1ki].pt() >= electronEtMin_ )
            {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
	   if (NOn==1 && fabs(toc[*l1ki].eta()) <= electronEtaMax_ && toc[*l1ki].pt() >= electronEtMin_ )
            {
	     NL1OnUM++;
	     v->getL1EtL1OnUMHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnUMHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
              }
	    ++idtypeiter;
	  }


      }


      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau || triggertype == trigger::TriggerL1TauJet)
	{


          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1TauJet || *idtypeiter == trigger::TriggerL1ForJet)
	      {
	   if (reco::deltaR(toc[*l1ki].eta(),toc[*l1ki].phi(),toc[*ki].eta(),toc[*ki].phi()) < tauDRMatch_ && fabs(toc[*l1ki].eta()) <= tauEtaMax_ && toc[*l1ki].pt() >= tauEtMin_ )
            {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
	   if (NOn==1 && fabs(toc[*l1ki].eta()) <= tauEtaMax_ && toc[*l1ki].pt() >= tauEtMin_ )
            {
	     NL1OnUM++;
	     v->getL1EtL1OnUMHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnUMHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
              }
	    ++idtypeiter;
	  }


      }


      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet || triggertype == trigger::TriggerL1CenJet || triggertype == trigger::TriggerL1ForJet )
	{

          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1TauJet || *idtypeiter == trigger::TriggerL1ForJet || *idtypeiter == trigger::TriggerL1CenJet)
	      {
	   if (reco::deltaR(toc[*l1ki].eta(),toc[*l1ki].phi(),toc[*ki].eta(),toc[*ki].phi()) < jetDRMatch_ && fabs(toc[*l1ki].eta()) <= jetEtaMax_ && toc[*l1ki].pt() >= jetEtMin_ )
            {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
	   if (NOn==1 && fabs(toc[*l1ki].eta()) <= jetEtaMax_ && toc[*l1ki].pt() >= jetEtMin_ )
            {
	     NL1OnUM++;
	     v->getL1EtL1OnUMHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnUMHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
              }
	    ++idtypeiter;
	  }


      }

      // for bjet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerBJet)
	{

          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1ForJet || *idtypeiter == trigger::TriggerL1CenJet)
	      {
	   if (reco::deltaR(toc[*l1ki].eta(),toc[*l1ki].phi(),toc[*ki].eta(),toc[*ki].phi()) < bjetDRMatch_ && fabs(toc[*l1ki].eta()) <= bjetEtaMax_ && toc[*l1ki].pt() >= bjetEtMin_ )
            {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
	   if (NOn==1 && fabs(toc[*l1ki].eta()) <= bjetEtaMax_ && toc[*l1ki].pt() >= bjetEtMin_ )
            {
	     NL1OnUM++;
	     v->getL1EtL1OnUMHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnUMHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
              }
	    ++idtypeiter;
	  }


	}
      // for met triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM )
	{

          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1ETM)
	      {
		//   if (reco::deltaR(toc[*l1ki].eta(),toc[*l1ki].phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3 && toc[*l1ki].pt() >= metMin_ )
		// {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	     // }
              }
	    ++idtypeiter;
	  }


      }
      // for sumet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTET || triggertype == trigger::TriggerL1ETT )
	{

          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1ETT)
	      {
	   if (toc[*l1ki].pt() >= sumEtMin_ )
            {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
              }
	    ++idtypeiter;
	  }



      }


      // for photon triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerPhoton)
	{

          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1IsoEG || *idtypeiter == trigger::TriggerL1NoIsoEG)
	      {
	   if (reco::deltaR(toc[*l1ki].eta(),toc[*l1ki].phi(),toc[*ki].eta(),toc[*ki].phi()) < photonDRMatch_ && fabs(toc[*l1ki].eta()) <= photonEtaMax_ && toc[*l1ki].pt() >= photonEtMin_ )
            {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
	   if (NOn==1 && fabs(toc[*l1ki].eta()) <= photonEtaMax_ && toc[*l1ki].pt() >= photonEtMin_ )
            {
	     NL1OnUM++;
	     v->getL1EtL1OnUMHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnUMHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
              }
	    ++idtypeiter;
	  }


	}// photon trigger type


      // for track triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerTrack)
	{

          trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
          for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	    if (*idtypeiter == trigger::TriggerL1CenJet || *idtypeiter == trigger::TriggerL1ForJet || *idtypeiter == trigger::TriggerL1TauJet)
	      {
	   if (reco::deltaR(toc[*l1ki].eta(),toc[*l1ki].phi(),toc[*ki].eta(),toc[*ki].phi()) < trackDRMatch_ && fabs(toc[*l1ki].eta()) <= trackEtaMax_ && toc[*l1ki].pt() >= trackEtMin_ )
            {
	     NL1On++;
	     v->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
	   if (NOn==1 && fabs(toc[*l1ki].eta()) <= trackEtaMax_ && toc[*l1ki].pt() >= trackEtMin_ )
            {
	     NL1OnUM++;
	     v->getL1EtL1OnUMHisto()->Fill(toc[*l1ki].pt());
	     v->getL1EtaVsL1PhiL1OnUMHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
	    }
              }
	    ++idtypeiter;
	  }


	}// track trigger type

      } //online object loop

      v->getNOnHisto()->Fill(NOn);      
      v->getNL1OnHisto()->Fill(NL1On);      
      v->getNL1OnUMHisto()->Fill(NL1OnUM);      
  

    } //numpassed
    
      v->getNL1Histo()->Fill(NL1);

    } //denompassed

   // fill histogram of filter ocupancy for each HLT path
   unsigned int lastModule = 0;

   unsigned int pathByIndex = triggerNames.triggerIndex(v->getPath());
   lastModule = triggerResults->index(pathByIndex);

    //go through the list of filters
    for(unsigned int filt = 0; filt < v->filtersAndIndices.size(); filt++){
      
      int binNumber = v->getFiltersHisto()->getTH1()->GetXaxis()->FindBin(v->filtersAndIndices[filt].first.c_str());      
      
      //check if filter passed
      if(triggerResults->accept(pathByIndex)){
        v->getFiltersHisto()->Fill(binNumber-1);//binNumber1 = 0 = first filter
      }
      //otherwise the module that issued the decision is the first fail
      //so that all the ones before it passed
      else if(v->filtersAndIndices[filt].second < lastModule){
        v->getFiltersHisto()->Fill(binNumber-1);//binNumber1 = 0 = first filter
      }

    } // end for filt

  } //pathinfo loop

}



// -- method called once each job just before starting event loop  --------
void 
FourVectorHLTOnline::beginJob()
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
    }  
}

// - method called once each job just after ending the event loop  ------------
void 
FourVectorHLTOnline::endJob() 
{
   LogInfo("FourVectorHLTOnline") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLTOnline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOnline") << "beginRun, run " << run.id();
// HLT config does not change within runs!
 
  if (!hltConfig_.init(processname_)) {
    processname_ = "FU";
    if (!hltConfig_.init(processname_)){
  LogDebug("FourVectorHLTOnline") << "HLTConfigProvider failed to initialize.";
    }
    // check if trigger name in (new) config
    //	cout << "Available TriggerNames are: " << endl;
    //	hltConfig_.dump("Triggers");
      }


  if (1)
 {
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
  }


    const unsigned int n(hltConfig_.size());
    if (plotAll_){
    for (unsigned int j=0; j!=n; ++j) {
    std::string pathname = hltConfig_.triggerName(j);  
    std::string l1pathname = "dummy";
    for (unsigned int i=0; i!=n; ++i) {
      // cout << hltConfig_.triggerName(i) << endl;
    
    std::string denompathname = hltConfig_.triggerName(i);  
    int objectType = 0;
    int denomobjectType = 0;
    //parse pathname to guess object type
    if (pathname.find("MET") != std::string::npos) 
      objectType = trigger::TriggerMET;    
    if (pathname.find("SumET") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("BTag") != std::string::npos) 
      objectType = trigger::TriggerBJet;    
    if (pathname.find("Mu") != std::string::npos) 
      objectType = trigger::TriggerMuon;    
    if (pathname.find("Ele") != std::string::npos) 
      objectType = trigger::TriggerElectron;    
    if (pathname.find("Photon") != std::string::npos) 
      objectType = trigger::TriggerPhoton;    
    if (pathname.find("Tau") != std::string::npos) 
      objectType = trigger::TriggerTau;    
    if (pathname.find("IsoTrack") != std::string::npos) 
      objectType = trigger::TriggerTrack;    

    //parse denompathname to guess denomobject type
    if (denompathname.find("MET") != std::string::npos) 
      denomobjectType = trigger::TriggerMET;    
    if (denompathname.find("SumET") != std::string::npos) 
      denomobjectType = trigger::TriggerTET;    
    if (denompathname.find("Jet") != std::string::npos) 
      denomobjectType = trigger::TriggerJet;    
    if (denompathname.find("BTag") != std::string::npos) 
      denomobjectType = trigger::TriggerBJet;    
    if (denompathname.find("Mu") != std::string::npos) 
      denomobjectType = trigger::TriggerMuon;    
    if (denompathname.find("Ele") != std::string::npos) 
      denomobjectType = trigger::TriggerElectron;    
    if (denompathname.find("Photon") != std::string::npos) 
      denomobjectType = trigger::TriggerPhoton;    
    if (denompathname.find("Tau") != std::string::npos) 
      denomobjectType = trigger::TriggerTau;    
    if (denompathname.find("IsoTrack") != std::string::npos) 
      denomobjectType = trigger::TriggerTrack;    

    // find L1 condition for numpath with numpath objecttype 

    // find PSet for L1 global seed for numpath, 
    // list module labels for numpath
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

            for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
		  //  l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
                  l1pathname = *numpathmodule; 
                  break; 
		}
    	} 
   
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (plotAll_ && denomobjectType == objectType && objectType != 0)
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));

    }
    }

    }
    else
    {
     // plot all diagonal combinations plus any other specified pairs
    for (unsigned int i=0; i!=n; ++i) {
      std::string denompathname = "";  
      std::string pathname = hltConfig_.triggerName(i);  
      std::string l1pathname = "dummy";
      int objectType = 0;
      int denomobjectType = 0;
    //parse pathname to guess object type
    if (pathname.find("MET") != std::string::npos) 
      objectType = trigger::TriggerMET;    
    if (pathname.find("SumET") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("BTag") != std::string::npos) 
      objectType = trigger::TriggerBJet;    
    if (pathname.find("Mu") != std::string::npos) 
      objectType = trigger::TriggerMuon;    
    if (pathname.find("Ele") != std::string::npos) 
      objectType = trigger::TriggerElectron;    
    if (pathname.find("Photon") != std::string::npos) 
      objectType = trigger::TriggerPhoton;    
    if (pathname.find("Tau") != std::string::npos) 
      objectType = trigger::TriggerTau;    
    if (pathname.find("IsoTrack") != std::string::npos) 
      objectType = trigger::TriggerTrack;    

    //parse denompathname to guess denomobject type
    if (denompathname.find("MET") != std::string::npos) 
      denomobjectType = trigger::TriggerMET;    
    if (denompathname.find("SumET") != std::string::npos) 
      denomobjectType = trigger::TriggerTET;    
    if (denompathname.find("Jet") != std::string::npos) 
      denomobjectType = trigger::TriggerJet;    
    if (denompathname.find("BTag") != std::string::npos) 
      denomobjectType = trigger::TriggerBJet;    
    if (denompathname.find("Mu") != std::string::npos) 
      denomobjectType = trigger::TriggerMuon;    
    if (denompathname.find("Ele") != std::string::npos) 
      denomobjectType = trigger::TriggerElectron;    
    if (denompathname.find("Photon") != std::string::npos) 
      denomobjectType = trigger::TriggerPhoton;    
    if (denompathname.find("Tau") != std::string::npos) 
      denomobjectType = trigger::TriggerTau;    
    if (denompathname.find("IsoTrack") != std::string::npos) 
      denomobjectType = trigger::TriggerTrack;    
    // find L1 condition for numpath with numpath objecttype 

    // find PSet for L1 global seed for numpath, 
    // list module labels for numpath
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
                  //l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression"); 
                  l1pathname = *numpathmodule;
                  break; 
		}
    } 
   
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (objectType == trigger::TriggerPhoton) ptMax = 100.0;
    if (objectType == trigger::TriggerElectron) ptMax = 100.0;
    if (objectType == trigger::TriggerMuon) ptMax = 100.0;
    if (objectType == trigger::TriggerTau) ptMax = 100.0;
    if (objectType == trigger::TriggerJet) ptMax = 300.0;
    if (objectType == trigger::TriggerBJet) ptMax = 300.0;
    if (objectType == trigger::TriggerMET) ptMax = 300.0;
    if (objectType == trigger::TriggerTET) ptMax = 600.0;
    if (objectType == trigger::TriggerTrack) ptMax = 100.0;

    if (objectType != 0){
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
      //create folder for pathname
     }
    }
    // now loop over denom/num path pairs specified in cfg, 
    // recording the off-diagonal ones
    for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair)
    {
      if (custompathnamepair->first != custompathnamepair->second)
	{

      std::string denompathname = custompathnamepair->second;  
      std::string pathname = custompathnamepair->first;  
     
      // check that these exist
      bool foundfirst = false;
      bool foundsecond = false;
      for (unsigned int i=0; i!=n; ++i) {
	if (hltConfig_.triggerName(i) == denompathname) foundsecond = true;
	if (hltConfig_.triggerName(i) == pathname) foundfirst = true;
      } 
      if (!foundfirst)
	{
	  edm::LogInfo("FourVectorHLTOnline") << "pathname not found, ignoring " << pathname;
          continue;
	}
      if (!foundsecond)
	{
	  edm::LogInfo("FourVectorHLTOnline") << "denompathname not found, ignoring " << pathname;
          continue;
	}

     //cout << pathname << "\t" << denompathname << endl;
      std::string l1pathname = "dummy";
      int objectType = 0;
      //int denomobjectType = 0;
    //parse pathname to guess object type
    if (pathname.find("MET") != std::string::npos) 
      objectType = trigger::TriggerMET;    
    if (pathname.find("SumET") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("BTag") != std::string::npos) 
      objectType = trigger::TriggerBJet;    
    if (pathname.find("Mu") != std::string::npos) 
      objectType = trigger::TriggerMuon;    
    if (pathname.find("Ele") != std::string::npos) 
      objectType = trigger::TriggerElectron;    
    if (pathname.find("Photon") != std::string::npos) 
      objectType = trigger::TriggerPhoton;    
    if (pathname.find("Tau") != std::string::npos) 
      objectType = trigger::TriggerTau;    
    if (pathname.find("IsoTrack") != std::string::npos) 
      objectType = trigger::TriggerTrack;    
    // find L1 condition for numpath with numpath objecttype 

    // find PSet for L1 global seed for numpath, 
    // list module labels for numpath
  
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
    
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
		  // l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
                  l1pathname = *numpathmodule;
                  //cout << *numpathmodule << endl; 
                  break; 
		}
    }
    
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (objectType == trigger::TriggerPhoton) ptMax = 100.0;
    if (objectType == trigger::TriggerElectron) ptMax = 100.0;
    if (objectType == trigger::TriggerMuon) ptMax = 100.0;
    if (objectType == trigger::TriggerTau) ptMax = 100.0;
    if (objectType == trigger::TriggerJet) ptMax = 300.0;
    if (objectType == trigger::TriggerBJet) ptMax = 300.0;
    if (objectType == trigger::TriggerMET) ptMax = 300.0;
    if (objectType == trigger::TriggerTET) ptMax = 300.0;
    if (objectType == trigger::TriggerTrack) ptMax = 100.0;

    if (objectType != 0)
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
    
	}
    }

    }



    // now set up all of the histos for each path
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {
    	MonitorElement *NOn, *onEtOn, *onEtavsonPhiOn=0;
	MonitorElement *NL1, *l1EtL1, *l1Etavsl1PhiL1=0;
    	MonitorElement *NL1On, *l1EtL1On, *l1Etavsl1PhiL1On=0;
    	MonitorElement *NL1OnUM, *l1EtL1OnUM, *l1Etavsl1PhiL1OnUM=0;
      MonitorElement *filters=0;
	std::string labelname("dummy");
        labelname = v->getPath() + "_wrt_" + v->getDenomPath();
	std::string histoname(labelname+"_NOn");
	std::string title(labelname+" N online");



        double histEtaMax = 2.5;
        if (v->getObjectType() == trigger::TriggerMuon || v->getObjectType() == trigger::TriggerL1Mu) 
	  {
	    histEtaMax = muonEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerElectron || v->getObjectType() == trigger::TriggerL1NoIsoEG || v->getObjectType() == trigger::TriggerL1IsoEG )
	  {
	    histEtaMax = electronEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerTau || v->getObjectType() == trigger::TriggerL1TauJet )
	  {
	    histEtaMax = tauEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerJet || v->getObjectType() == trigger::TriggerL1CenJet || v->getObjectType() == trigger::TriggerL1ForJet )
	  {
	    histEtaMax = jetEtaMax_; 
	  }
        else if (v->getObjectType() == trigger::TriggerBJet)
	  {
	    histEtaMax = bjetEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerMET || v->getObjectType() == trigger::TriggerL1ETM )
	  {
	    histEtaMax = 5.0; 
	  }
        else if (v->getObjectType() == trigger::TriggerPhoton)
	  {
	    histEtaMax = photonEtaMax_; 
	  }
        else if (v->getObjectType() == trigger::TriggerTrack)
	  {
	    histEtaMax = trackEtaMax_; 
	  }

        TString pathfolder = dirname_ + TString("/") + v->getPath();
        dbe_->setCurrentFolder(pathfolder.Data());

	NOn =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1";
	title = labelname+" N L1";
	NL1 =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1On";
	title = labelname+" N L1On";
	NL1On =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1OnUM";
	title = labelname+" N L1OnUM";
	NL1OnUM =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

        histoname = labelname+"_onEtOn";
	title = labelname+" onE_t online";
	onEtOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1EtL1";
	title = labelname+" l1E_t L1";
	l1EtL1 =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

        int nBins2D = 10;

	histoname = labelname+"_onEtaonPhiOn";
	title = labelname+" on#eta vs on#phi online";
	onEtavsonPhiOn =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_l1Etal1PhiL1";
	title = labelname+" l1#eta vs l1#phi L1";
	l1Etavsl1PhiL1 =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_l1EtL1On";
	title = labelname+" l1E_t L1+online";
	l1EtL1On =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1Etal1PhiL1On";
	title = labelname+" l1#eta vs l1#phi L1+online";
	l1Etavsl1PhiL1On =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_l1EtL1OnUM";
	title = labelname+" l1E_t L1+onlineUM";
	l1EtL1OnUM =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1Etal1PhiL1OnUM";
	title = labelname+" l1#eta vs l1#phi L1+onlineUM";
	l1Etavsl1PhiL1OnUM =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

       // -------------------------
       //
       //  Filters for each path
       //
       // -------------------------
       
       // get all modules in this HLT path
       vector<string> moduleNames = hltConfig_.moduleLabels( v->getPath() ); 
       
       int numModule = 0;
       string moduleName, moduleType;
       unsigned int moduleIndex;
       
       //print module name
       vector<string>::const_iterator iDumpModName;
       for (iDumpModName = moduleNames.begin();iDumpModName != moduleNames.end();iDumpModName++) {

         moduleName = *iDumpModName;
         moduleType = hltConfig_.moduleType(moduleName);
         moduleIndex = hltConfig_.moduleIndex(v->getPath(), moduleName);

         LogTrace ("FourVectorHLTOffline") << "Module "      << numModule
             << " is called " << moduleName
             << " , type = "  << moduleType
             << " , index = " << moduleIndex
             << endl;

         numModule++;

         if((moduleType.find("Filter") != string::npos && moduleType.find("HLTTriggerTypeFilter") == string::npos && moduleType.find("HLTL1NumberFilterXXX") == string::npos ) || 
            (moduleType.find("Associator") != string::npos) || 
            (moduleType.find("HLTLevel1GTSeed") != string::npos) || 
            (moduleType.find("HLTPrescaler") != string::npos) ||
            (moduleType.find("HLTGlobalSumsCaloMET") != string::npos) 
	    ) {

           std::pair<std::string, int> filterIndexPair;
           filterIndexPair.first   = moduleName;
           filterIndexPair.second  = moduleIndex;
           v->filtersAndIndices.push_back(filterIndexPair);

         }


       }//end for modulesName

       TString pathsummary = TString("HLT/FourVector/PathsSummary");
       dbe_->setCurrentFolder(pathsummary.Data()); 

       //int nbin_sub = 5;
       int nbin_sub = 10;
    
       // count plots for subfilter
       filters = dbe_->book1D("Filters_" + v->getPath(), 
                              "Filters_" + v->getPath(),
                              nbin_sub+1, -0.5, 0.5+(double)nbin_sub);
       
       for(unsigned int filt = 0; filt < v->filtersAndIndices.size(); filt++){

         filters->setBinLabel(filt+1, (v->filtersAndIndices[filt]).first);

       }

	v->setHistos( NOn, onEtOn, onEtavsonPhiOn, NL1, l1EtL1, l1Etavsl1PhiL1, NL1On, l1EtL1On, l1Etavsl1PhiL1On, NL1OnUM, l1EtL1OnUM, l1Etavsl1PhiL1OnUM, filters
);


    }
 }
 return;



}

/// EndRun
void FourVectorHLTOnline::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOnline") << "endRun, run " << run.id();
}
