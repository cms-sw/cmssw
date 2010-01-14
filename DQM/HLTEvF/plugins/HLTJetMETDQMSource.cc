
#include "TMath.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/HLTJetMETDQMSource.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "math.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TH2F.h"
#include "TPRegexp.h"

using namespace edm;
using namespace std;

HLTJetMETDQMSource::HLTJetMETDQMSource(const edm::ParameterSet& iConfig):
  resetMe_(true),
  currentRun_(-99) {
  LogDebug("HLTJetMETDQMSource") << "constructor....";

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogDebug("HLTJetMETDQMSource") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  dirname_ = iConfig.getUntrackedParameter("dirname",
					   std::string("HLT/JetMET/"));
  //dirname_ +=  iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  processname_ = iConfig.getParameter<std::string>("processname");
  verbose_     = iConfig.getUntrackedParameter< bool >("verbose", false);
  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",50);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);
  plotwrtMu_ = iConfig.getUntrackedParameter<bool>("plotwrtMu", false);
  plotEff_ = iConfig.getUntrackedParameter<bool>("plotEff", false);
  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths =  iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");

 
 for(std::vector<edm::ParameterSet>::iterator pathconf = paths.begin() ; pathconf != paths.end();  pathconf++) {
    std::pair<std::string, std::string> custompathnamepair;
    int prescaleused;
    custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
    custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
    custompathnamepairs_.push_back(custompathnamepair);
    //prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  prescaleused = pathconf->getParameter<int>("prescaleused");  
  prescUsed_.push_back(prescaleused);
    //    customdenompathnames_.push_back(pathconf->getParameter<std::string>("denompathname"));  
    // custompathnames_.push_back(pathconf->getParameter<std::string>("pathname"));  
  }

  custompathnamemu_ = iConfig.getUntrackedParameter("pathnameMuon",
						    std::string("HLT_L1Mu"));
  
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");

  muonEtaMax_ = iConfig.getUntrackedParameter<double>("muonEtaMax",2.5);
  muonEtMin_ = iConfig.getUntrackedParameter<double>("muonEtMin",0.0);
  muonDRMatch_  =iConfig.getUntrackedParameter<double>("muonDRMatch",0.3); 

  jetEtaMax_ = iConfig.getUntrackedParameter<double>("jetEtaMax",5.0);
  jetEtMin_ = iConfig.getUntrackedParameter<double>("jetEtMin",0.0);
  jetDRMatch_  =iConfig.getUntrackedParameter<double>("jetDRMatch",0.3); 

  metMin_ = iConfig.getUntrackedParameter<double>("metMin",0.0);
  htMin_ = iConfig.getUntrackedParameter<double>("htMin",0.0);
  sumEtMin_ = iConfig.getUntrackedParameter<double>("sumEtMin",0.0);	
  


}


HLTJetMETDQMSource::~HLTJetMETDQMSource() {
 
 
  //
 // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


void
HLTJetMETDQMSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
  using namespace trigger;
   
  ++nev_;
  if (verbose_) std::cout << " --------------- N Event ------------" << nev_ << std::endl; // LogDebug("HLTJetMETDQMSource")<< "HLTJetMETDQMSource: analyze...." ;
  
  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
    iEvent.getByLabel(triggerResultsLabelFU,triggerResults);
    if(!triggerResults.isValid()) {
      if (verbose_) std::cout << "TriggerResults not found, skipping event" << std::endl; 
      return;
    }
  }
  TriggerNames triggerNames(*triggerResults);  
  unsigned int npath = triggerResults->size();

  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) {
    edm::InputTag triggerSummaryLabelFU(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(), "FU");
    iEvent.getByLabel(triggerSummaryLabelFU,triggerObj);
    if(!triggerObj.isValid()) {
      if (verbose_) std::cout << "TriggerEvent not found, skipping event" << std::endl; 
      return;
    }
  }
  //----plot all the ditributions for JET and MET -----------------------


  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());


  int testint=0;
  if (plotAll_) testint=1;
  if (verbose_) std::cout << " plots all  " <<  testint  << std::endl; 
  if (plotAll_) {
    if (verbose_) std::cout << " Look at basic distributions " << std::endl; 
    
    int N =0;
    int NL1=0;
    
  
    for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){ 
      NL1++;
      N++;
      int triggertype = 0;     
      triggertype = v->getObjectType();
      
      bool l1accept = false;
      edm::InputTag l1testTag(v->getl1Path(),"",processname_);
      const int l1index = triggerObj->filterIndex(l1testTag);
      if ( l1index >= triggerObj->sizeFilters() ) {
	if (verbose_) std::cout<< "no index "<< l1index << " of that name " << v->getl1Path() << "\t" << "\t" << l1testTag << endl;
	continue;
      }

      const trigger::Vids & idtype = triggerObj->filterIds(l1index);
      const trigger::Keys & l1k = triggerObj->filterKeys(l1index);
      //if (verbose_) std::cout << "filterID "<< idtype << " keys " << l1k << std::endl;
      //if (verbose_) std::cout << " keys " << l1k << std::endl;
      l1accept = l1k.size() > 0;
      //if (l1accept) 
      //l1accept = true ;
     
      //
      //-----------------------------------------------------

      bool passed = false;
      for(unsigned int i = 0; i < npath; ++i) {
	if ( triggerNames.triggerName(i).find(v->getPath()) != std::string::npos  
	     && triggerResults->accept(i)){
	  passed = true;
	  if (verbose_) cout << " i " << i << "  trigger name " << v->getPath() << endl;
	  break;
	}
	
      }
      if(passed){
	rate_All->Fill(N-0.5);
	
	if (verbose_) cout <<  "  N " << N << "  trigger name " << v->getPath() << endl;
		
	if (!l1accept) {
            edm::LogInfo("HLTJetMETDQMSource") << "l1 seed path not accepted for hlt path "<< v->getPath() << "\t" << v->getl1Path();
      }
    
	
	edm::InputTag filterTag = v->getTag();

	
	//const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	
	if (v->getLabel() == "dummy"){
	  const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	  //loop over labels
	  

	for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++)  {	  	
	  edm::InputTag testTag(*labelIter,"",processname_);
	  if (verbose_) cout << v->getPath() << "\t" << testTag.label() << "\t" << testTag.process() << endl;
	  
	  int testindex = triggerObj->filterIndex(testTag);
	  
	  if ( !(testindex >= triggerObj->sizeFilters()) ) {
	    if (verbose_) cout << "found one! " << v->getPath() << "\t" << testTag.label() << endl; 
	    
	    
            filterTag = testTag; 
	    v->setLabel(*labelIter);
	  }
	}
	}
	const int index = triggerObj->filterIndex(filterTag);
	if (verbose_)      cout << "filter index "<< index << " of that name " << filterTag << endl;
	if ( index >= triggerObj->sizeFilters() ) {
	  if (verbose_)      cout << "WTF no index "<< index << " of that name " << filterTag << endl;
	  continue; // 
	}
	const trigger::Keys & k = triggerObj->filterKeys(index);
	
	for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	  
	  
	  
	  
	  if (verbose_)	cout << " filling HLT " <<  v->getPath() << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
	  if ( triggertype == trigger::TriggerJet 
	       || triggertype == trigger::TriggerL1TauJet 
	       || triggertype == trigger::TriggerL1CenJet 
	       || triggertype == trigger::TriggerL1ForJet)
	    {
	      if (verbose_)	cout << " filling HLT " <<  v->getPath() << "\t" << "  *ki  " << *ki << " Nki   "<< N<< " " << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
	      
	      
	      v->getEtHisto()->Fill(toc[*ki].pt());
	      v->getEtaHisto()->Fill(toc[*ki].eta());
	      v->getPhiHisto()->Fill(toc[*ki].phi());
	      v->getEtaPhiHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	      
	      //---L1 Histograms
	      trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
	      for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
		if (*idtypeiter == trigger::TriggerL1TauJet || *idtypeiter == trigger::TriggerL1ForJet || *idtypeiter == trigger::TriggerL1CenJet)
		  {
		     if (verbose_)	cout << " filling L1 HLT " <<  v->getPath() << "\t" << "  *l1ki  " << *l1ki << " NL1   "<< NL1<< " " << toc[*l1ki].pt() << "\t" << toc[*l1ki].eta() << "\t" << toc[*l1ki].phi() << endl;
		     
		    rate_All_L1->Fill(NL1-0.5);
		    v->getL1EtHisto()->Fill(toc[*l1ki].pt());
		    v->getL1PhiHisto()->Fill(toc[*l1ki].phi());
		    v->getL1PhiHisto()->Fill(toc[*l1ki].eta());
		    v->getL1EtaPhiHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
		  }
		++idtypeiter;
	      }//end L1
	      
	    }
	  
	  //----------------------------
	  else if (triggertype == trigger::TriggerMuon ||triggertype == trigger::TriggerL1Mu  )
	    {
	    if (verbose_)	cout << " filling HLT " <<  v->getPath() << "\t" << "  *ki  " << *ki << " Nki   "<< N << " " << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
	  
	    
	    v->getEtHisto()->Fill(toc[*ki].pt());
	    //v->getEtaHisto()->Fill(toc[*ki].eta());
	    //v->getPhiHisto()->Fill(toc[*ki].phi());
	    //v->getEtaPhiHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	 
   //---L1 Histograms
	    trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
	    for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	      if (*idtypeiter == trigger::TriggerL1Mu)
		{
		  //NL1[npth]++;
		  rate_All_L1->Fill(NL1-0.5);
		  v->getL1EtHisto()->Fill(toc[*l1ki].pt());
		  //v->getL1PhiHisto()->Fill(toc[*l1ki].phi()());
		  //v->getL1PhiHisto()->Fill(toc[*l1ki].eta()());
		  //v->getL1EtaPhiHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
		}
	      ++idtypeiter;
	    }//end L1
	    
	    
	    }
	  //--------------------------------------
	  else if ( triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM
	       || triggertype == trigger::TriggerTET){
	    
	    v->getEtHisto()->Fill(toc[*ki].pt());
	    v->getPhiHisto()->Fill(toc[*ki].phi());
	   
	    //---L1 Histograms
	    trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
	    for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	      if (*idtypeiter == trigger::TriggerL1ETM)
		{
		  //NL1[npth]++;
		  rate_All_L1->Fill(NL1-0.5);
		  v->getL1EtHisto()->Fill(toc[*l1ki].pt());
		  //v->getL1PhiHisto()->Fill(toc[*l1ki].phi()());
		  //v->getL1PhiHisto()->Fill(toc[*l1ki].eta()());
		  //v->getL1EtaPhiHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
		}
	      ++idtypeiter;
	    }//end L1
	   
	  }
	  //-------------------------------------------
      else if ( triggertype == trigger::TriggerTET || triggertype == trigger::TriggerL1ETT){
	    
	    v->getEtHisto()->Fill(toc[*ki].pt());
	    v->getPhiHisto()->Fill(toc[*ki].phi());
	   //---L1 Histograms
	    trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
	    for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {
	      if (*idtypeiter == TriggerL1ETT)
		{
		  //NL1[npth]++;
		  rate_All_L1->Fill(NL1-0.5);
		  v->getL1EtHisto()->Fill(toc[*l1ki].pt());
		  //v->getL1PhiHisto()->Fill(toc[*l1ki].phi()());
		  //v->getL1PhiHisto()->Fill(toc[*l1ki].eta()());
		  //v->getL1EtaPhiHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());
		}
	      ++idtypeiter;
	    }//end L1
	    
	     
	  }
      
	}
	
      }//ifpassed 
 

   } 
  }//----------------- end if plotAll


	    
  //----plot all the ditributions for JET and MET with respect to a muon trigger -----------------------
  if (plotwrtMu_) {

 if (verbose_) std::cout << " Look at basic distributions with respect to muon " << std::endl; 
    
    int N =0;
    int NL1=0;
    
  
    for(PathInfoCollection::iterator v = hltPathswrtMu_.begin(); v!= hltPathswrtMu_.end(); ++v ){ 
      NL1++;
      N++;
      int triggertype = 0;     
      triggertype = v->getObjectType();
      
      bool l1accept = false;
      edm::InputTag l1testTag(v->getl1Path(),"",processname_);
      const int l1index = triggerObj->filterIndex(l1testTag);
      if ( l1index >= triggerObj->sizeFilters() ) {
	if (verbose_) std::cout<< "no index "<< l1index << " of that name (wrtMu)" << v->getl1Path() << "\t" << "\t" << l1testTag << endl;
	continue;
      }

      //const trigger::Vids & idtype = triggerObj->filterIds(l1index);
      const trigger::Keys & l1k = triggerObj->filterKeys(l1index);
      //if (verbose_) std::cout << "filterID "<< idtype << " keys " << l1k << std::endl;
      //if (verbose_) std::cout << " keys " << l1k << std::endl;
      l1accept = l1k.size() > 0;
      //if (l1accept) rate_All_L1->Fill(NL1-0.5);
      //l1accept = true ;
     
      //
      //-----------------------------------------------------
      bool denompassed = false;
      bool passed = false;
      for(unsigned int i = 0; i < npath; ++i) {
	if ( triggerNames.triggerName(i).find(v->getPath()) != std::string::npos  && triggerResults->accept(i)){
	  passed = true; break ;
	}
      }
      for(unsigned int i = 0; i < npath; ++i) {
	if ( triggerNames.triggerName(i).find(v->getDenomPath()) != std::string::npos  && triggerResults->accept(i)){
	denompassed = true; break ;
	}
      }
      if(denompassed){

	if (verbose_) cout <<  "  N " << N << "  trigger wrt mu demom name " << v->getDenomPath() << endl;
	if(passed){

	rate_wrtMu->Fill(N-0.5);
	
	if (verbose_) cout <<  "  N " << N << "  trigger name " << v->getPath() << endl;
		
	if (!l1accept) {
            edm::LogInfo("HLTJetMETDQMSource") << "l1 seed path not accepted for hlt path "<< v->getPath() << "\t" << v->getl1Path();
      }
    
	
	edm::InputTag filterTag = v->getTag();

	
	//const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	
	if (v->getLabel() == "dummy"){
	  const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	  //loop over labels
	  

	for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++)  {	  	
	  edm::InputTag testTag(*labelIter,"",processname_);
	  if (verbose_) cout << v->getPath() << "\t" << testTag.label() << "\t" << testTag.process() << endl;
	  
	  int testindex = triggerObj->filterIndex(testTag);
	  
	  if ( !(testindex >= triggerObj->sizeFilters()) ) {
	    if (verbose_) cout << "found one! " << v->getPath() << "\t" << testTag.label() << endl; 
	    
	    
            filterTag = testTag; 
	    v->setLabel(*labelIter);
	  }
	}
	}
	const int index = triggerObj->filterIndex(filterTag);
	if (verbose_)      cout << "filter index "<< index << " of that name (wrtMu)" << filterTag << endl;
	if ( index >= triggerObj->sizeFilters() ) {
	  if (verbose_)      cout << "WTF no index "<< index << " of that name (wrtMu)" << filterTag << endl;
	  continue; // 
	}
	const trigger::Keys & k = triggerObj->filterKeys(index);
	
	for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	  
	  	  
	  if (verbose_)	cout << " filling HLT (wrtMu)" <<  v->getPath() << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
	  if ( triggertype == trigger::TriggerJet 
	       || triggertype == trigger::TriggerL1TauJet 
	       || triggertype == trigger::TriggerL1CenJet 
	       || triggertype == trigger::TriggerL1ForJet)
	    {
	      if (verbose_)	cout << " filling HLT (wrtMu)" <<  v->getPath() << "\t" << "  *ki  " << *ki << " Nki   "<< N<< " " << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
	      
	      
	      v->getEtwrtMuHisto()->Fill(toc[*ki].pt());
	      //v->getEtawrtMuHisto()->Fill(toc[*ki].eta());
	      //v->getPhiwrtMuHisto()->Fill(toc[*ki].phi());
	      // v->getEtaPhiwrtMuHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	      
	    }
	  
	  //----------------------------
	  else if (triggertype == trigger::TriggerMuon ||triggertype == trigger::TriggerL1Mu  )
	    {
	    if (verbose_)	cout << " filling HLT (wrtMu)" <<  v->getPath() << "\t" << "  *ki  " << *ki << " Nki   "<< N << " " << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
	  
	    
	    v->getEtwrtMuHisto()->Fill(toc[*ki].pt());
	    //v->getEtaHisto()->Fill(toc[*ki].eta());
	    //v->getPhiHisto()->Fill(toc[*ki].phi());
	    //v->getEtaPhiHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	    
	   
	    }
	  //--------------------------------------
	  else if ( triggertype == trigger::TriggerMET 
		    || triggertype == trigger::TriggerL1ETM
	  ){
	    
	    v->getEtwrtMuHisto()->Fill(toc[*ki].pt());
	    
	    
	  }
	  //-------------------------------------------
      else if ( triggertype == trigger::TriggerTET || triggertype == trigger::TriggerL1ETT){
	    
	    v->getEtwrtMuHisto()->Fill(toc[*ki].pt());
	    
	  }
      
	}
	
      }//ifpassed 
      }//if denom passed

   } 

    
  }//----------------- end if plowrtMu





  //----------plot efficiency---------------
  
  if (plotEff_) {
 if (verbose_) std::cout << " Look at basic distributions for Eff " << std::endl; 
    
     unsigned int N = 0;
     unsigned int Ndenom = 0;
     unsigned int Nnum = 0;
      
  
    for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ){ 
      Ndenom++;
      Nnum++;
      N++;

      int triggertype = 0;     
      triggertype = v->getObjectType();
      
      bool l1accept = false;
      edm::InputTag l1testTag(v->getl1Path(),"",processname_);
      const int l1index = triggerObj->filterIndex(l1testTag);
      if ( l1index >= triggerObj->sizeFilters() ) {
	if (verbose_) std::cout<< "no index "<< l1index << " of that name " << v->getl1Path() << "\t" << "\t" << l1testTag << endl;
	continue;
      }
      
      //const trigger::Vids & idtype = triggerObj->filterIds(l1index);
      const trigger::Keys & l1k = triggerObj->filterKeys(l1index);
      //if (verbose_) std::cout << "filterID "<< idtype << " keys " << l1k << std::endl;
      //if (verbose_) std::cout << " keys " << l1k << std::endl;
      l1accept = l1k.size() > 0;
      bool passed = false;
      for(unsigned int i = 0; i < npath; ++i) {
	if ( triggerNames.triggerName(i).find(v->getPath()) != std::string::npos  
	     && triggerResults->accept(i)){
	  passed = true;
	  if (verbose_) cout << " i " << i << "  trigger name " << v->getPath() << endl;
	  break;
	}
	
      }
      bool denompassed = false;
      for(unsigned int i = 0; i < npath; ++i) {
	if ( triggerNames.triggerName(i).find(v->getDenomPath()) != std::string::npos  
	     && triggerResults->accept(i)){
	  denompassed = true;
	  if (verbose_) cout << " i " << i << "  trigger name " << v->getDenomPath() << endl;
	  break;
	}
	
      }

      if (denompassed){
	rate_Denom->Fill(Ndenom-0.5);
	if (verbose_) cout <<  "  N " << N << "  trigger name " << v->getDenomPath() << endl;
	
	if (!l1accept) {
	  edm::LogInfo("HLTJetMETDQMSource") << "l1 seed path not accepted for hlt path "<< v->getPath() << "\t" << v->getl1Path();
	}
	
		edm::InputTag filterTag = v->getDenomTag();
	
	
	//const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
		if (verbose_) cout << v->getDenomPath() << "\t" << v->getDenomLabel() << "\t"  << endl;
		    
	        
		if (v->getDenomLabel() == "denomdummy"){
		  const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getDenomPath());
		  //loop over labels
		  
		  
		  for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++)  {	  	
		    edm::InputTag testTag(*labelIter,"",processname_);
		    if (verbose_) cout << v->getDenomPath() << "\t" << testTag.label() << "\t" << testTag.process() << endl;
		    
		    int testindex = triggerObj->filterIndex(testTag);
		    
		    if ( !(testindex >= triggerObj->sizeFilters()) ) {
		      if (verbose_) cout << "found one! " << v->getDenomPath() << "\t" << testTag.label() << endl; 
		      
		      
		      filterTag = testTag; 
		      v->setDenomLabel(*labelIter);
		    }
		  }
		}
		const int index = triggerObj->filterIndex(filterTag);
		if (verbose_)      cout << "filter index "<< index << " of that name " << filterTag << endl;
		if ( index >= triggerObj->sizeFilters() ) {
		  if (verbose_)      cout << "WTF no index "<< index << " of that name " << filterTag << endl;
		  continue; // 
		}
		const trigger::Keys & k = triggerObj->filterKeys(index);
		
		int numobj=-1;
		for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
		  if (verbose_)	cout << " filling HLT Denom " <<  v->getDenomPath() << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
		  if ( triggertype == trigger::TriggerJet 
		       || triggertype == trigger::TriggerL1TauJet 
		       || triggertype == trigger::TriggerL1CenJet 
		       || triggertype == trigger::TriggerL1ForJet)
		    {
		      if (verbose_)	cout << " filling HLT Denom" <<  v->getDenomPath() << "\t" << "  *kiDeno  " << *ki << " NkiDenom   "<< Ndenom<< " " << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
		      
		      numobj=*ki;
		      v->getEtDenomHisto()->Fill(toc[*ki].pt());
		      v->getEtaDenomHisto()->Fill(toc[*ki].eta());
		      v->getPhiDenomHisto()->Fill(toc[*ki].phi());
		    }
		  //--------------------------------------
		  else if ( triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM
			    || triggertype == trigger::TriggerTET){
		    
		    v->getEtDenomHisto()->Fill(toc[*ki].pt());
		    v->getPhiDenomHisto()->Fill(toc[*ki].phi());
		  }
		  //-------------------------------------------
		  else if ( triggertype == trigger::TriggerTET || triggertype == trigger::TriggerL1ETT){
		    v->getEtDenomHisto()->Fill(toc[*ki].pt());
		    v->getPhiDenomHisto()->Fill(toc[*ki].phi());
		    
		  }
		  if ( numobj != -1) break;
		}
		
      }//if denom passed
      

      if(denompassed){

	if (passed){

 rate_Num->Fill(Nnum-0.5);
 
if (verbose_) cout <<  "  N " << N << "  trigger name " << v->getPath() << endl;
		
	if (!l1accept) {
            edm::LogInfo("HLTJetMETDQMSource") << "l1 seed path not accepted for hlt path "<< v->getPath() << "\t" << v->getl1Path();
      }
    
	
	edm::InputTag filterTag = v->getTag();

	
	//const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	if (verbose_) cout << v->getPath() << "\t" << v->getLabel()  << endl;
	if (v->getLabel() == "dummy"){
	  const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	  //loop over labels
	  

	for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++)  {	  	
	  edm::InputTag testTag(*labelIter,"",processname_);
	  if (verbose_) cout << v->getPath() << "\t" << testTag.label() << "\t" << testTag.process() << endl;
	  
	  int testindex = triggerObj->filterIndex(testTag);
	  
	  if ( !(testindex >= triggerObj->sizeFilters()) ) {
	    if (verbose_) cout << "found one! " << v->getPath() << "\t" << testTag.label() << endl; 
	    
	    
            filterTag = testTag; 
	    v->setLabel(*labelIter);
	  }
	}
	}
	const int index = triggerObj->filterIndex(filterTag);
	if (verbose_)      cout << "filter index "<< index << " of that name " << filterTag << endl;
	if ( index >= triggerObj->sizeFilters() ) {
	  if (verbose_)      cout << "WTF no index "<< index << " of that name " << filterTag << endl;
	  continue; // 
	}
	const trigger::Keys & k = triggerObj->filterKeys(index);
	int numobj = -1;
	for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	 	  
	  if (verbose_)	cout << " filling HLT  " <<  v->getPath() << "\t" << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
	  if ( triggertype == trigger::TriggerJet 
	       || triggertype == trigger::TriggerL1TauJet 
	       || triggertype == trigger::TriggerL1CenJet 
	       || triggertype == trigger::TriggerL1ForJet)
	    {
	      if (verbose_)	cout << " filling HLT " <<  v->getPath() << "\t" << "  *ki  " << *ki << " Nki   "<< N<< " " << toc[*ki].pt() << "\t" << toc[*ki].eta() << "\t" << toc[*ki].phi() << endl;
	      numobj=*ki;
	      
	      v->getEtNumHisto()->Fill(toc[*ki].pt());
	      v->getEtaNumHisto()->Fill(toc[*ki].eta());
	      v->getPhiNumHisto()->Fill(toc[*ki].phi());
	    }
	  //--------------------------------------
	  else if ( triggertype == trigger::TriggerMET || triggertype == trigger::TriggerL1ETM
		    || triggertype == trigger::TriggerTET){
	    
	    v->getEtNumHisto()->Fill(toc[*ki].pt());
	    v->getPhiNumHisto()->Fill(toc[*ki].phi());
	  }
	  //-------------------------------------------
	  else if ( triggertype == trigger::TriggerTET || triggertype == trigger::TriggerL1ETT){
	    v->getEtNumHisto()->Fill(toc[*ki].pt());
	    v->getPhiNumHisto()->Fill(toc[*ki].phi());
	    
	  }
	   if ( numobj != -1) break;
	}

	}//end if passed
      } //if denompassed
	
    }

    //----------plot efficiency---------------
 
    if ( (nev_ % 1000) == 0 ){
      if (verbose_) cout << " Calculating Eff.... " << nev_ << endl;   

      //if (nev_ % 1000)
      //if (verbose_) cout << "starting the luminosity block -----" << endl;
   
      TH1F *rate_denom=NULL;
      TH1F *rate_num=NULL;
      rate_denom = rate_Denom->getTH1F(); 
      rate_num   = rate_Num->getTH1F(); 
      //rate_denom->Sumw2();rate_num->Sumw2();
      if ((rate_denom->Integral() != 0.)  && (rate_num->Integral() != 0.) ) {
	    if (verbose_) cout << " Nonzero rate summary  -----" << endl;
	    for(int j=1; j <= rate_denom->GetXaxis()->GetNbins();j++ ){
	      double y1 = rate_num->GetBinContent(j);
	      double y2 = rate_denom->GetBinContent(j);
	      double eff = y2 > 0. ? y1/y2 : 0.;
	      rate_Eff->setBinContent(j, eff);
	      double y1err = rate_num->GetBinError(j);
	      double y2err = rate_denom->GetBinError(j);
	      double efferr = 0.0;
	      
	      if (y2 && y1  > 0.) efferr =  (y1/y2)* sqrt ((y1err/y1)*(y1err/y1) + (y2err/y2)*(y2err/y2)) ;
	      rate_Eff->setBinError(j, efferr);

	    }
	  }

      for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ) { 
	 int triggertype = 0;     
	triggertype = v->getObjectType();
	if (verbose_) cout << " triggertype -----  " << triggertype << endl;
	if ( triggertype == trigger::TriggerJet 
	     || triggertype == trigger::TriggerL1TauJet
	     || triggertype == trigger::TriggerL1CenJet
	     || triggertype == trigger::TriggerL1ForJet
	     || triggertype == trigger::TriggerL1Mu
	     || triggertype == trigger::TriggerMuon ){	
      
	
	  TH1F *EtNum=NULL;
	  TH1F *EtaNum=NULL;
	  TH1F *PhiNum=NULL;
	  TH1F *EtDenom=NULL;
	  TH1F *EtaDenom=NULL;
	  TH1F *PhiDenom=NULL;
	
	  EtNum= ( v->getEtNumHisto())->getTH1F() ;
	  EtaNum= ( v->getEtaNumHisto())->getTH1F() ;
	  PhiNum= ( v->getPhiNumHisto())->getTH1F() ;
	  EtDenom= ( v->getEtDenomHisto())->getTH1F() ;
	  EtaDenom= ( v->getEtaDenomHisto())->getTH1F() ;
	  PhiDenom= ( v->getPhiDenomHisto())->getTH1F() ;
	  	  
	  if ((EtNum->Integral() != 0.)  && (EtDenom->Integral() != 0.) ) {
	    if (verbose_) cout << " Nonzero Jet Et  -----" << endl;
	    for(int j=1; j <= EtNum->GetXaxis()->GetNbins();j++ ){
	      double y1 = EtNum->GetBinContent(j);
	      double y2 = EtDenom->GetBinContent(j);
	      double eff = y2 > 0. ? y1/y2 : 0.;
	      
	      v->getEtEffHisto()->setBinContent(j, eff);
	      double y1err = EtNum->GetBinError(j);
	      double y2err = EtDenom->GetBinError(j);
	      double efferr = 0.0;
	      
	      if (y2 && y1  > 0.) efferr =  (y1/y2)* sqrt ((y1err/y1)*(y1err/y1) + (y2err/y2)*(y2err/y2)) ;
	      v->getEtEffHisto()->setBinError(j, efferr);

	      if(verbose_) cout << eff << " "<<  efferr << " "<< y1 << " " << y2 << " "<< y1err << " " << y2err << endl;
	      
	    }
	  }

	  if (EtaNum->Integral() != 0.  && EtaDenom->Integral() != 0. ) {
	    for(int j=1; j <= EtaNum->GetXaxis()->GetNbins();j++ ){
	      double y1 = EtaNum->GetBinContent(j);
	      double y2 = EtaDenom->GetBinContent(j);
	      double eff = y2 > 0. ? y1/y2 : 0.;
	      v->getEtaEffHisto()->setBinContent(j, eff);
	      double y1err = EtaNum->GetBinError(j);
	      double y2err = EtaDenom->GetBinError(j);
	      double efferr = 0.0;
	      
	      if (y2 && y1  > 0.) efferr =  (y1/y2)* sqrt ((y1err/y1)*(y1err/y1) + (y2err/y2)*(y2err/y2)) ;
	      v->getEtaEffHisto()->setBinError(j, efferr);
//	    
	  
	    }
	  }
	  if (PhiNum->Integral() != 0.  && PhiDenom->Integral() != 0. ) {
	    for(int j=1; j <= PhiNum->GetXaxis()->GetNbins();j++ ){
	      double y1 = PhiNum->GetBinContent(j);
	      double y2 = PhiDenom->GetBinContent(j);
	      double eff = y2 > 0. ? y1/y2 : 0.;
	      v->getPhiEffHisto()->setBinContent(j, eff);
	      double y1err = PhiNum->GetBinError(j);
	      double y2err = PhiDenom->GetBinError(j);
	      double efferr = 0.0;
	      
	      if (y2 && y1  > 0.) efferr =  (y1/y2)* sqrt ((y1err/y1)*(y1err/y1) + (y2err/y2)*(y2err/y2)) ;
	      v->getPhiEffHisto()->setBinError(j, efferr);
//	    
	    }
	
	  }
	}
	if ( triggertype == trigger::TriggerMET
	     || triggertype == trigger::TriggerTET ){
	  TH1F *EtNum=NULL;
	  TH1F *PhiNum=NULL;
	  TH1F *EtDenom=NULL;
	  TH1F *PhiDenom=NULL;
	  
	  EtNum= ( v->getEtNumHisto())->getTH1F() ;
	  PhiNum= ( v->getPhiNumHisto())->getTH1F() ;
	  EtDenom= ( v->getEtDenomHisto())->getTH1F() ;
	  PhiDenom= ( v->getPhiDenomHisto())->getTH1F() ;
	  


	  if (EtNum->Integral() != 0.  && EtDenom->Integral() != 0. ) {
	    if (verbose_) cout << " Nonzero Met Et  -----" << endl;
	    for(int j=1; j <= EtNum->GetXaxis()->GetNbins();j++ ){
	      double y1 = EtNum->GetBinContent(j);
	      double y2 = EtDenom->GetBinContent(j);
	      double eff = y2 > 0. ? y1/y2 : 0.;
	      v->getEtEffHisto()->setBinContent(j, eff);
	      double y1err = EtNum->GetBinError(j);
	      double y2err = EtDenom->GetBinError(j);
	      double efferr = 0.0;
	      
	      if (y2 && y1  > 0.) efferr =  (y1/y2)* sqrt ((y1err/y1)*(y1err/y1) + (y2err/y2)*(y2err/y2)) ;
	      v->getEtEffHisto()->setBinError(j, efferr);
//	      
	    }
	  }
	  if (PhiNum->Integral() != 0.  && PhiDenom->Integral() != 0. ) {

	    for(int j=1; j <= PhiNum->GetXaxis()->GetNbins();j++ ){
	      double y1 = PhiNum->GetBinContent(j);
	      double y2 = PhiDenom->GetBinContent(j);
	      double eff = y2 > 0. ? y1/y2 : 0.;
	      v->getPhiEffHisto()->setBinContent(j, eff);
	      double y1err = PhiNum->GetBinError(j);
	      double y2err = PhiDenom->GetBinError(j);
	      double efferr = 0.0;
	      
	      if (y2 && y1  > 0.) efferr =  (y1/y2)* sqrt ((y1err/y1)*(y1err/y1) + (y2err/y2)*(y2err/y2)) ;
	      v->getPhiEffHisto()->setBinError(j, efferr);
//	      
	    }
	  }
	}// met triggers

      }
    }

  }
}

// -- method called once each job just before starting event loop  --------
void 
HLTJetMETDQMSource::beginJob(){
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

// BeginRun
void HLTJetMETDQMSource::beginRun(const edm::Run& run, const edm::EventSetup& c){

  if (verbose_) std::cout << "beginRun, run " << run.id() << std::endl;
  // HLT config does not change within runs!
 
  if (!hltConfig_.init(processname_)) {
    processname_ = "FU";
    if (!hltConfig_.init(processname_)){
      if (verbose_) std::cout << "HLTConfigProvider failed to initialize." << std::endl;
    }
    // check if trigger name in (new) config
    //	cout << "Available TriggerNames are: " << endl;
    //	hltConfig_.dump("Triggers");
  }

  //hltConfig_.dump("Triggers");

  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
      

  const unsigned int n(hltConfig_.size());

  

  //-------------if plot all jetmet trigger pt,eta, phi-------------------
  if (plotAll_){
    if (verbose_) cout << " booking histos All " << endl;
    std::string foldernm = "/All/";
    
    if (dbe) {
    
      dbe->setCurrentFolder(dirname_ + foldernm);
    }
    for (unsigned int j=0; j!=n; ++j) {
      std::string pathname = hltConfig_.triggerName(j);  
      std::string l1pathname = "dummy";
      //for (unsigned int i=0; i!=n; ++i) {
      // cout << hltConfig_.triggerName(i) << endl;
    
      //std::string denompathname = hltConfig_.triggerName(i);  
      std::string denompathname = "";  
      unsigned int usedPresscale = 1;  
      unsigned int objectType = 0;
      //int denomobjectType = 0;
      //parse pathname to guess object type

      if ( (pathname.find("BTag") != std::string::npos) 
	   || pathname.find("Ele") != std::string::npos
	   || pathname.find("Photon") != std::string::npos
	   || pathname.find("IsoTrack") != std::string::npos
	   //|| pathname.find("Mu") != std::string::npos 
	   ) continue;
	    
      if (pathname.find("MET") != std::string::npos) objectType = trigger::TriggerMET;    
      if (pathname.find("L1MET") != std::string::npos) objectType = trigger::TriggerL1ETM;    
      if (pathname.find("SumET") != std::string::npos) objectType = trigger::TriggerTET;    
      if (pathname.find("Jet") != std::string::npos) objectType = trigger::TriggerJet;    
      //if (pathname.find("HLT_Jet30") != std::string::npos) objectType = trigger::TriggerJet;    
      //if (pathname.find("HLT_Jet50") != std::string::npos) objectType = trigger::TriggerJet;    
      //if ((pathname.find("HLT_Mu3") != std::string::npos) || (pathname.find("HLT_L2Mu9") != std::string::npos)  ) objectType = trigger::TriggerMuon;    
	if ((pathname.find("HLT_L1MuOpen") != std::string::npos)  ) objectType = trigger::TriggerMuon;    
	  
	   
      //std::cout << "objecttye " << objectType << std::endl;
      // find L1 condition for numpath with numpath objecttype 

      // find PSet for L1 global seed for numpath, 
      // list module labels for numpath
      std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
    
      for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	  numpathmodule!= numpathmodules.end(); 
	  ++numpathmodule ) {
     
	//  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed") {
	  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
	  //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
	  //  l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
	  l1pathname = *numpathmodule; 
	  break; 
	}
      } 
    
      std::string filtername("dummy");
      std::string Denomfiltername("denomdummy");
      float ptMin = 0.0;
      float ptMax = 300.0;
      if (objectType != 0 )
	hltPathsAll_.push_back(PathInfo(usedPresscale, denompathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, ptMin, ptMax));

	  
    }


    std::string histonm="JetMET_rate_All";
    std::string histonmL1="JetMET_rate_All_L1";
    std::string histot="JetMET Rate Summary";
    std::string histoL1t="JetMET L1 Rate Summary";
    rate_All = dbe->book1D(histonm.c_str(),histot.c_str(),
			   hltPathsAll_.size(),0,hltPathsAll_.size());

    rate_All_L1 = dbe->book1D(histonmL1.c_str(),histoL1t.c_str(),
			      hltPathsAll_.size(),0,hltPathsAll_.size());
    
    //rate_All->setBinLabel(hltPathsAll_.size()+1,"Rate",1);
    unsigned int nname=0;
    unsigned int nnameL1=0;
    for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){
      std::string labelnm("dummy");
      labelnm = v->getPath();
      rate_All->setBinLabel(nname+1,labelnm); 
      nname++;

      std::string labelnml1("dummyl1");
      labelnml1 = v->getl1Path();
      rate_All_L1->setBinLabel(nnameL1+1,labelnml1); 
      nnameL1++;
     }
 
    
    
 
    // now set up all of the histos for each path
    for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ) {
      MonitorElement *N = 0;
      MonitorElement *Et = 0;
      MonitorElement *EtaPhi = 0;
      MonitorElement *Eta = 0;
      MonitorElement  *Phi = 0;
      MonitorElement *NL1 = 0;
      MonitorElement *l1Et = 0;
      MonitorElement *l1EtaPhi = 0;
      MonitorElement *l1Eta = 0;
      MonitorElement *l1Phi = 0;
    	
      std::string labelname("dummy");
      labelname = v->getPath();
      std::string histoname(labelname+"");
      std::string title(labelname+"");



      double histEtaMax = 2.5;
      if (v->getObjectType() == trigger::TriggerMuon  
	  || v->getObjectType() == trigger::TriggerL1Mu)  {
	histEtaMax = muonEtaMax_;
	nBins_ = 20 ;
      }
        
      else if (v->getObjectType() == trigger::TriggerJet 
	       || v->getObjectType() == trigger::TriggerL1CenJet 
	       || v->getObjectType() == trigger::TriggerL1ForJet ) {
	histEtaMax = jetEtaMax_; 
	nBins_ = 60 ;
      }
        
      else if (v->getObjectType() == trigger::TriggerMET 
	       || v->getObjectType() == trigger::TriggerL1ETM ) {
	histEtaMax = 5.0; 
	nBins_ = 60 ;
      }
        
      TString pathfolder = dirname_ + foldernm + v->getPath();
      dbe_->setCurrentFolder(pathfolder.Data());
      if (verbose_) cout << "Booking Histos in Directory " << pathfolder.Data() << endl;
      int nBins2D = 10;
      

      histoname = labelname+"_Et";
      title = labelname+" E_t";
      Et =  dbe->book1D(histoname.c_str(),
			title.c_str(),nBins_, 
			v->getPtMin(),
			v->getPtMax());

      histoname = labelname+"_l1Et";
      title = labelname+" L1 E_t";
      l1Et =  dbe->book1D(histoname.c_str(),
			  title.c_str(),nBins_, 
			  v->getPtMin(),
			  v->getPtMax());
 
      if (labelname.find("Jet") != std::string::npos
	  || labelname.find("Mu") != std::string::npos) {
      
	histoname = labelname+"_EtaPhi";
	title = labelname+" #eta vs #phi";
	EtaPhi =  dbe->book2D(histoname.c_str(),
			      title.c_str(),
			      nBins2D,-histEtaMax,histEtaMax,
			      nBins2D,-TMath::Pi(), TMath::Pi());
	
	histoname = labelname+"_l1EtaPhi";
	title = labelname+"L1 #eta vs L1 #phi";
	l1EtaPhi =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_Phi";
	title = labelname+" #phi";
	Phi =  dbe->book1D(histoname.c_str(),
			   title.c_str(),
			   nBins_,-TMath::Pi(), TMath::Pi());
	
	histoname = labelname+"_l1Phi";
	title = labelname+"L1 #phi";
	l1Phi =  dbe->book1D(histoname.c_str(),
			     title.c_str(),
			     nBins_,-TMath::Pi(), TMath::Pi());
	histoname = labelname+"_Eta";
	title = labelname+" #eta";
	Eta =  dbe->book1D(histoname.c_str(),
			   title.c_str(),
			   nBins_,-histEtaMax,histEtaMax
			   );
	
	histoname = labelname+"_l1Eta";
	title = labelname+"L1 #eta";
	l1Eta =  dbe->book1D(histoname.c_str(),
			     title.c_str(),
			     nBins_,-histEtaMax,histEtaMax);
      } 
      else if( (labelname.find("MET") != std::string::npos)
	       || (labelname.find("SumET") != std::string::npos)    ){
	histoname = labelname+"_phi";
	title = labelname+" #phi";
	Phi =  dbe->book1D(histoname.c_str(),
			   title.c_str(),
			   nBins_,-TMath::Pi(), TMath::Pi());
	
	histoname = labelname+"_l1Phi";
	title = labelname+"L1 #phi";
	l1Phi =  dbe->book1D(histoname.c_str(),
			     title.c_str(),
			     nBins_,-TMath::Pi(), TMath::Pi());
	
      }
      v->setHistos( N, Et, EtaPhi, Eta, Phi, NL1, l1Et, l1EtaPhi,l1Eta, l1Phi);


    }
    if (verbose_) cout << "Done  booking histos All  " << endl;
  }

      
  //----------------plot all jetmet trigger wrt some muon trigger-----
  if  (plotwrtMu_){
    if (verbose_) cout << " booking histos wrt Muon " << endl;
    std::string foldernm = "/wrtMuon/";
    if (dbe)   {
      dbe->setCurrentFolder(dirname_ + foldernm);
    }
 
    for (unsigned int j=0; j!=n; ++j)  {
      std::string pathname = hltConfig_.triggerName(j);  
      std::string l1pathname = "dummy";
      //for (unsigned int i=0; i!=n; ++i) 
      //{
      if (verbose_) cout << hltConfig_.triggerName(j) << endl;
      std::string denompathname = custompathnamemu_ ;  
      int objectType = 0;
      int usedPresscale = 1;
      int denomobjectType = 0;
      //parse pathname to guess object type
      if ( (pathname.find("BTag") != std::string::npos) 
	   || pathname.find("Ele") != std::string::npos
	   || pathname.find("Photon") != std::string::npos
	   || pathname.find("IsoTrack") != std::string::npos
	   ) continue;
	    
      if (pathname.find("MET") != std::string::npos) objectType = trigger::TriggerMET;    
      if (pathname.find("L1MET") != std::string::npos) objectType = trigger::TriggerL1ETM;    
      if (pathname.find("SumET") != std::string::npos) objectType = trigger::TriggerTET;    
      if (pathname.find("Jet") != std::string::npos) objectType = trigger::TriggerJet;    
      //if (pathname.find("HLT_Jet30") != std::string::npos) objectType = trigger::TriggerJet;    
      //if (pathname.find("HLT_Jet50") != std::string::npos) objectType = trigger::TriggerJet;    
      if ((pathname.find("HLT_L1MuOpen") != std::string::npos) ) objectType = trigger::TriggerMuon;    
      if (denompathname.find("HLT_L1MuOpen") != std::string::npos) denomobjectType = trigger::TriggerMuon;    
	    
      // find L1 condition for numpath with numpath objecttype 

      // find PSet for L1 global seed for numpath, 
      // list module labels for numpath
      std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

      for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	//  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed") {
	  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
	  //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
	  //  l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
	  l1pathname = *numpathmodule; 
	  break; 
	}
      } 
	      
      std::string filtername("dummy");
      std::string Denomfiltername("denomdummy");
      float ptMin = 0.0;
      float ptMax = 300.0;
      if ( objectType != 0){
	if (verbose_) cout << " wrt muon PathInfo(denompathname, pathname, l1pathname, filtername,  denomfiltername, processname_, objectType, ptMin, ptMax  " << denompathname << " "<< pathname << " "<< l1pathname << " " << filtername  << " " << Denomfiltername << " " <<  processname_ << " " <<  objectType << " " <<  ptMin << " " <<  ptMax<< endl;
	
	hltPathswrtMu_.push_back(PathInfo(usedPresscale, denompathname, pathname, l1pathname, filtername,  Denomfiltername, processname_, objectType, ptMin, ptMax));
      }
    }
    




    std::string histonm="JetMET_rate_wrt_" + custompathnamemu_ + "_Summary";
    std::string histt="JetMET Rate wrt " + custompathnamemu_ + "Summary";
    rate_wrtMu = dbe->book1D(histonm.c_str(),histt.c_str(),
			   hltPathswrtMu_.size()+1,0,hltPathswrtMu_.size()+1);

    
    int nname=0;
    for(PathInfoCollection::iterator v = hltPathswrtMu_.begin(); v!= hltPathswrtMu_.end(); ++v ){
      std::string labelnm("dummy");
      labelnm = v->getPath();
      
      rate_wrtMu->setBinLabel(nname+1,labelnm);
            
      nname++;
    }
 




    // now set up all of the histos for each path
    for(PathInfoCollection::iterator v = hltPathswrtMu_.begin(); v!= hltPathswrtMu_.end(); ++v )  {
      MonitorElement *NwrtMu = 0;
      MonitorElement *EtwrtMu = 0;
      MonitorElement *EtaPhiwrtMu = 0;
      MonitorElement *PhiwrtMu = 0;
      
      std::string labelname("dummy");
      labelname = v->getPath() + "_wrt_" + v->getDenomPath();
      std::string histoname(labelname+"");
      std::string title(labelname+"");



      double histEtaMax = 2.5;
      if (v->getObjectType() == trigger::TriggerMuon || v->getObjectType() == trigger::TriggerL1Mu)  {
	histEtaMax = muonEtaMax_;nBins_ = 20 ;
      }
        
      else if (v->getObjectType() == trigger::TriggerJet || v->getObjectType() == trigger::TriggerL1CenJet || v->getObjectType() == trigger::TriggerL1ForJet ){
	histEtaMax = jetEtaMax_; nBins_ = 60 ;
      }
        
      else if (v->getObjectType() == trigger::TriggerMET || v->getObjectType() == trigger::TriggerL1ETM )  {
	histEtaMax = 5.0; nBins_ = 60 ;
      }
        
      TString pathfolder = dirname_ + foldernm + v->getPath();
      dbe_->setCurrentFolder(pathfolder.Data());
      if (verbose_) cout << "Booking Histos in Directory " << pathfolder.Data() << endl;
      int nBins2D = 10;
	 
      //pathfolder = dirname_ + TString("/wrtMuon/") + v->getPath();
      

      histoname = labelname+"_Et";
      title = labelname+" E_t";
      EtwrtMu =  dbe->book1D(histoname.c_str(),
			     title.c_str(),nBins_, 
			     v->getPtMin(),
			     v->getPtMax());




      if ((v->getPath()).find("Jet") != std::string::npos
	  || (v->getPath()).find("Mu") != std::string::npos) {
	histoname = labelname+"_EtaPhi";
	title = labelname+" #eta vs #phi";
	EtaPhiwrtMu =  dbe->book2D(histoname.c_str(),
				   title.c_str(),
				   nBins2D,-histEtaMax,histEtaMax,
				   nBins2D,-TMath::Pi(), TMath::Pi());
      }
      else if( ((v->getPath()).find("MET") != std::string::npos) 
	       || ((v->getPath()).find("SumET") != std::string::npos)    ){
	histoname = labelname+"_phi";
	title = labelname+" #phi";
	PhiwrtMu =  dbe->book1D(histoname.c_str(),
				title.c_str(),
				nBins_,-TMath::Pi(), TMath::Pi());
      }

      v->setHistoswrtMu( NwrtMu, EtwrtMu, EtaPhiwrtMu, PhiwrtMu);
      
    }
    if (verbose_) cout << "Done  booking histos wrt Muon " << endl;
  }
  //////////////////////////////////////////////////////////////////

  if (plotEff_)	{
    // plot efficiency for specified HLT path pairs
    if (verbose_) cout << " booking histos for Efficiency " << endl;

    std::string foldernm = "/Efficiency/";
    if (dbe) {
	
      dbe->setCurrentFolder(dirname_ + foldernm);
    }
    // now loop over denom/num path pairs specified in cfg, 
    int countN = 0;  
    for (std::vector<std::pair<std::string, std::string> >::iterator
	   custompathnamepair = custompathnamepairs_.begin(); 
	 custompathnamepair != custompathnamepairs_.end(); 
	 ++custompathnamepair) {
	     
      std::string denompathname = custompathnamepair->second;  
      std::string pathname = custompathnamepair->first;  
      int usedPrescale = prescUsed_[countN];
      if (verbose_) std::cout << " ------prescale used ----------" << usedPrescale << std::endl; 
      // check that these exist
      bool foundfirst = false;
      bool foundsecond = false;
      for (unsigned int i=0; i!=n; ++i) {
	if (hltConfig_.triggerName(i) == denompathname) foundsecond = true;
	if (hltConfig_.triggerName(i) == pathname) foundfirst = true;
      } 
      if (!foundfirst) {
	edm::LogInfo("HLTJetMETDQMSource") 
	  << "pathname not found, ignoring "
	  << pathname;
	continue;
      }
      if (!foundsecond) {
	edm::LogInfo("HLTJetMETDQMSource") 
	  << "denompathname not found, ignoring "
	  << pathname;
	continue;
      }

      //if (verbose_) cout << pathname << "\t" << denompathname << endl;
      std::string l1pathname = "dummy";
      int objectType = 0;
      //int denomobjectType = 0;
      //parse pathname to guess object type
      if (pathname.find("MET") != std::string::npos) objectType = trigger::TriggerMET;  
      if (pathname.find("L1MET") != std::string::npos) objectType = trigger::TriggerL1ETM;  
      if (pathname.find("SumET") != std::string::npos) objectType = trigger::TriggerTET;
      if (pathname.find("Jet") != std::string::npos)  objectType = trigger::TriggerJet;
      if ((pathname.find("HLT_Mu3") != std::string::npos) || (pathname.find("HLT_Mu9") != std::string::npos)  || (pathname.find("HLT_L1MuOpen") != std::string::npos)  )  objectType = trigger::TriggerMuon;
      // find L1 condition for numpath with numpath objecttype 

      // find PSet for L1 global seed for numpath, 
      // list module labels for numpath
  
      std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
    
      for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	  numpathmodule!= numpathmodules.end(); 
	  ++numpathmodule ) {
	//  if (verbose_) cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
	if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed"){
	  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
	  //                if (verbose_)   cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
	  // l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
	  l1pathname = *numpathmodule;
	  //if (verbose_) cout << *numpathmodule << endl; 
	  break; 
	}
      }
    
      std::string filtername("dummy");
      std::string Denomfiltername("denomdummy");
      float ptMin = 0.0;
      float ptMax = 300.0;
      if (objectType == trigger::TriggerMuon) ptMax = 300.0;
      if (objectType == trigger::TriggerJet) ptMax = 300.0;
      if (objectType == trigger::TriggerMET) ptMax = 300.0;
      if (objectType == trigger::TriggerTET) ptMax = 300.0;
		 
      
      if (objectType != 0){
	if (verbose_) cout << " PathInfo(denompathname, pathname, l1pathname, filtername,  Denomfiltername, processname_, objectType, ptMin, ptMax  " << denompathname << " "<< pathname << " "<< l1pathname << " " << filtername << " " << Denomfiltername << " " <<  processname_ << " " <<  objectType << " " <<  ptMin << " " <<  ptMax<< endl;
	
	
	hltPathsEff_.push_back(PathInfo(usedPrescale, denompathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, ptMin, ptMax));
      }

      countN++;
    }
    
    std::string histonm="JetMET_Efficiency_Summary";
    std::string histonmDenom="Denom_passed_Summary";
    std::string histonmNum="Num_passed_Summary";
    rate_Denom = dbe->book1D(histonmDenom.c_str(),histonmDenom.c_str(),
    			   hltPathsEff_.size(),0,hltPathsEff_.size());
    rate_Num = dbe->book1D(histonmNum.c_str(),histonmNum.c_str(),
    			   hltPathsEff_.size(),0,hltPathsEff_.size());
    rate_Eff = dbe->book1D(histonm.c_str(),histonm.c_str(),
    			   hltPathsEff_.size(),0,hltPathsEff_.size());

    //rate_Eff = dbe_->bookProfile("Efficiency_Summary","Efficiency_Summary", hltPathsEff_.size(), -0.5, hltPathsEff_.size()-0.5, 1000, 0.0, 1.0);
    int nname=0;
    for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ){
      std::string labelnm("dummy");
      std::string labeldenom("dummy");
      labelnm = v->getPath();
      labeldenom = v->getDenomPath();
      //rate_Eff->getTProfile()->GetXaxis()->SetBinLabel(nname+1,labelnm.c_str());
 
      rate_Eff->setBinLabel(nname+1,labelnm);
      rate_Denom->setBinLabel(nname+1,labeldenom);
      rate_Num->setBinLabel(nname+1,labelnm);
      

      nname++;
    }



    for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ) {
      MonitorElement *NEff=0;
      MonitorElement *EtEff=0;
      MonitorElement *EtaEff=0;
      MonitorElement *PhiEff=0;
      MonitorElement *NNum =0;
      MonitorElement *EtNum =0;
      MonitorElement *EtaNum =0;
      MonitorElement *PhiNum =0; 
      MonitorElement *NDenom=0;
      MonitorElement *EtDenom=0;
      MonitorElement *EtaDenom=0;
      MonitorElement *PhiDenom=0;
      std::string labelname("dummy");
      labelname = "Eff_" + v->getPath() + "_wrt_" + v->getDenomPath();
      std::string histoname(labelname+"");
      std::string title(labelname+"");
      
      

      double histEtaMax = 5.0;
      if (v->getObjectType() == trigger::TriggerMuon || v->getObjectType() == trigger::TriggerL1Mu) {
	histEtaMax = muonEtaMax_; nBins_ = 20 ;
      }
        
      else if (v->getObjectType() == trigger::TriggerJet || v->getObjectType() == trigger::TriggerL1CenJet || v->getObjectType() == trigger::TriggerL1ForJet ){
	histEtaMax = jetEtaMax_; nBins_ = 60 ;
      }
        
      else if (v->getObjectType() == trigger::TriggerMET || v->getObjectType() == trigger::TriggerL1ETM ) {
	histEtaMax = 5.0; nBins_ = 60 ;
      }
        
      TString pathfolder = dirname_ + foldernm + v->getPath();
      dbe_->setCurrentFolder(pathfolder.Data());
      if (verbose_) cout << "Booking Histos in Directory " << pathfolder.Data() << endl;
	    
      //pathfolder = dirname_ + TString("/Eff/") + v->getPath(); 
      //int nBins2D = 10;
      histoname = labelname+"_Et_Eff";
      title = labelname+" E_t Eff";
      EtEff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
			   v->getPtMin(),
			   v->getPtMax());

      histoname = labelname+"_Et_Num";
      title = labelname+" E_t Num";
      EtNum =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
			   v->getPtMin(),
			   v->getPtMax());


      histoname = labelname+"_Et_Denom";
      title = labelname+" E_t Denom";
      EtDenom =  dbe->book1D(histoname.c_str(),
			     title.c_str(),nBins_, 
			     v->getPtMin(),
			     v->getPtMax());
      
      if ((v->getPath()).find("Jet") != std::string::npos
	  || (v->getPath()).find("Mu") != std::string::npos) {
	histoname = labelname+"_Eta_Eff";
	title = labelname+" #eta  Eff";
	EtaEff =  dbe->book1D(histoname.c_str(),
			      title.c_str(),
			      nBins_,-histEtaMax,histEtaMax);
	histoname = labelname+"_Phi_Eff";
	title = labelname+" #phi Eff";
	PhiEff =  dbe->book1D(histoname.c_str(),
			      title.c_str(),
			      nBins_,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_Eta_Num";
	title = labelname+" #eta  Num";
	EtaNum =  dbe->book1D(histoname.c_str(),
			      title.c_str(),
			      nBins_,-histEtaMax,histEtaMax);
	histoname = labelname+"_Phi_Num";
	title = labelname+" #phi Num";
	PhiNum =  dbe->book1D(histoname.c_str(),
			      title.c_str(),
			      nBins_,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_Eta_Denom";
	title = labelname+" #eta  Denom";
	EtaDenom =  dbe->book1D(histoname.c_str(),
				title.c_str(),
				nBins_,-histEtaMax,histEtaMax);
	histoname = labelname+"_Phi_Denom";
	title = labelname+" #phi Denom";
	PhiDenom =  dbe->book1D(histoname.c_str(),
				title.c_str(),
				nBins_,-TMath::Pi(), TMath::Pi());
      }


      else if( ((v->getPath()).find("MET") != std::string::npos) 
	       || ((v->getPath()).find("SumET") != std::string::npos)    ){
	
     	histoname = labelname+"_Phi_Eff";
	title = labelname+" #phi Eff";
	PhiEff =  dbe->book1D(histoname.c_str(),
			      title.c_str(),
			      nBins_,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_Phi_Num";
	title = labelname+" #phi Num";
	PhiNum =  dbe->book1D(histoname.c_str(),
			      title.c_str(),
			      nBins_,-TMath::Pi(), TMath::Pi());
	
	histoname = labelname+"_Phi_Denom";
	title = labelname+" #phi Denom";
	PhiDenom =  dbe->book1D(histoname.c_str(),
				title.c_str(),
				nBins_,-TMath::Pi(), TMath::Pi());
	
      }
     
      v->setHistosEff( NEff, EtEff, EtaEff, PhiEff, NNum, EtNum, EtaNum, PhiNum, NDenom, EtDenom, EtaDenom, PhiDenom);
      
    }
    if (verbose_) cout << "Done  booking histos for Efficiency " << endl;
  }
  if (verbose_) cout << "End BeginRun ---------------- " << endl;
  
}



//--------------------------------------------------------
void HLTJetMETDQMSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
					      const EventSetup& context) {


}
//--------------------------------------------------------
void HLTJetMETDQMSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
					    const EventSetup& context) {
}

// - method called once each job just after ending the event loop  ------------
void 
HLTJetMETDQMSource::endJob() {
  LogInfo("HLTJetMETDQMSource") << "analyzed " << nev_ << " events";
  return;
}



/// EndRun
void HLTJetMETDQMSource::endRun(const edm::Run& run, const edm::EventSetup& c){
  if (verbose_) std::cout << "endRun, run " << run.id() << std::endl;
}
