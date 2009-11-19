#include "TString.h"
#include "DQM/Physics/interface/JetChecker.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

JetChecker::JetChecker(const edm::ParameterSet& iConfig, const std::string& relativePath, const std::string& label )
{
  dqmStore_ = edm::Service<DQMStore>().operator->();

  relativePath_ = relativePath;
  label_ = label;
  nBins_=50;
  checkBtaggingSet_=iConfig.getParameter<bool>("includeBtagInfo");
  if(checkBtaggingSet_){
    btaggingalgonames_ = iConfig.getParameter<std::vector<std::string> >("btaggingAlgoLabels");
    btaggingMatchDr_= iConfig.getParameter<double>("btaggingMatchDr");
    btaggingcuts_ = iConfig.getParameter<edm::ParameterSet>("btaggingSelectors");
    //get the information on the b-tagging cuts from the PSet
    std::vector<double> cutsworkertemp= btaggingcuts_.getParameter<std::vector<double> >("cuts");
    std::vector<std::string> algosworkertemp = btaggingcuts_.getParameter<std::vector<std::string> >("taggers");
    // move these into the container:
    if(cutsworkertemp.size()!=algosworkertemp.size())
      edm::LogError("InputUnknown") <<"WARNING: you are passing a different number of cuts as you are algos!" << std::endl;

    // loop where the copy into workingpoints_ is done.
    for(size_t ii=0; ii<cutsworkertemp.size() && ii<algosworkertemp.size(); ++ii){
      std::pair<std::string, double> temppair(algosworkertemp[ii],cutsworkertemp[ii]);
      edm::LogInfo("Debug|JetChecker") << "creating tagging point for tagger " << temppair.first<< " at cut " << temppair.second << std::endl;
      workingpoints_.push_back(temppair);
    }
    // just to be sure, resize the temp workers:
    cutsworkertemp.resize(0);
    algosworkertemp.resize(0);
  }
}

JetChecker::~JetChecker()
{
   delete dqmStore_;
}


void
JetChecker::analyze(const std::vector<reco::CaloJet>& jets, bool useJES, const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::vector <CaloTowerPtr> jettowers;
  double sumtwrpt; 
  double corrJES = 1;
  for( unsigned int i=0;i<jets.size();i++) { 
    if(useJES){
     corrJES = corrector->correction((jets)[i], iEvent, iSetup);
    }

    edm::LogInfo("Debug|JetChecker") << "looping over jets, now at jet " << i << " pT:" << (jets)[i].pt() << ", eta:" << (jets)[i].eta() << std::endl;
    hists_["Jetn90"]->Fill((jets)[i].n90());
    hists_["JetTowersArea"]->Fill((jets)[i].towersArea());
    hists_["emEnergyFraction"]->Fill((jets)[i].emEnergyFraction());
    hists_["hcalEnergyFraction"]->Fill((jets)[i].energyFractionHadronic());
    hists_["maxEInEmTowers"]->Fill((jets)[i].maxEInEmTowers());
    hists_["maxEInHadTowers"]->Fill((jets)[i].maxEInHadTowers());
    
    //towers infos
    sumtwrpt   =0;
    jettowers = (jets)[i].getCaloConstituents(); 
    std::vector <CaloTowerPtr>::const_iterator caloiter;
    for(caloiter=jettowers.begin();caloiter!=jettowers.end();caloiter++){
      //       double caloet=(*caloiter)->et();
      hists_["JetTwrEt"]->Fill((*caloiter)->et());
      hists_["JetTwrPhi"]->Fill((*caloiter)->phi());
      hists_["JetTwrEta"]->Fill((*caloiter)->eta());
      sumtwrpt   +=(*caloiter)->pt()*corrJES;
    }
     hists_["JetTwrSumPt"]->Fill( sumtwrpt);
     hists_["JetdiffTwrSumPt"]->Fill( fabs(sumtwrpt- (jets)[i].pt()));
 
     double diff =  jettowers.size() - (jets)[i].n90();
     hists_["Jetn90vsE-n90"]->Fill((jets)[i].n90(), diff  );

  }
  
  //Fill October exercise quantities:
  if (jets.size() > 0) {
    hists_["ptHighest_pt"]->Fill(jets.at(0).pt()*((useJES) ? corrector->correction(jets.at(0), iEvent, iSetup) : 1.));
    hists_["ptHighest_eta"]->Fill(jets.at(0).eta());
  }
  if (jets.size() > 1) {
    hists_["ptSecond_pt"]->Fill(jets.at(1).pt()*((useJES) ? corrector->correction(jets.at(1), iEvent, iSetup) : 1.));
    hists_["ptSecond_eta"]->Fill(jets.at(1).eta());
  }
  if (jets.size() > 2) {
    hists_["ptThird_pt"]->Fill(jets.at(2).pt()*((useJES) ? corrector->correction(jets.at(2), iEvent, iSetup) : 1.));
    hists_["ptThird_eta"]->Fill(jets.at(2).eta());
  }
  if (jets.size() > 3) {
    hists_["ptFourth_pt"]->Fill(jets.at(3).pt()*((useJES) ? corrector->correction(jets.at(3), iEvent, iSetup) : 1.));
    hists_["ptFourth_eta"]->Fill(jets.at(3).eta());
  }
  
  // check b-tagging:
  analyzeWithBjets(jets,useJES,iEvent,iSetup);
}

void 
JetChecker::begin(const edm::EventSetup& iSetup, const std::string& jetCorrector)
{
  corrector = JetCorrector::getJetCorrector(jetCorrector,iSetup);
  dqmStore_->setCurrentFolder( relativePath_+"/CaloJets_"+label_ );
  
  hists_["Jetn90"]            = dqmStore_->book1D("Jetn90" ,"n90 ",10,0,10);
  hists_["Jetn90"]            ->setAxisTitle("n90",1);
  hists_["JetTowersArea"]     = dqmStore_->book1D("JetTowersArea" ," Jet Towers Area ",nBins_,0,1);
  hists_["JetTowersArea"]     ->setAxisTitle("Jet Towers Area",1);
  hists_["emEnergyFraction"]  = dqmStore_->book1D("emEnergyFraction" ,"jet electomagnetic energy fraction ",nBins_,0, 1);
  hists_["emEnergyFraction"]  ->setAxisTitle("em Energy Fraction",1);
  hists_["hcalEnergyFraction"]= dqmStore_->book1D("hcalEnergyFraction" ,"jet hadronic energy fraction ",nBins_,0, 1);
  hists_["hcalEnergyFraction"]->setAxisTitle("hcal Energy Fraction",1);
  hists_["maxEInEmTowers"]    = dqmStore_->book1D("maxEInEmTowers" ,"maximum energy deposited in ECAL towers ",nBins_,0, 100);
  hists_["maxEInEmTowers"]    ->setAxisTitle("Max Energy in ECAL towers",1);
  hists_["maxEInHadTowers"]   = dqmStore_->book1D("maxEInHadTowers" ,"maximum energy deposited in HCAL towers ",nBins_,0, 100);
  hists_["maxEInHadTowers"]   ->setAxisTitle("Max Energy in HCAL towers",1);
  hists_["Jetn90vsE-n90"]     = dqmStore_->book2D("Jetn90vsE-n90" ,"n90 vs jet n-n90 ",10,0, 10, 10, 0, 10);
  hists_["Jetn90vsE-n90"]     ->setAxisTitle("n90 vs E-n90",1);
  hists_["JetTwrEt"]          = dqmStore_->book1D("JetTwrEt" ,"jet towers Et ",nBins_,0,100);
  hists_["JetTwrEt"]          ->setAxisTitle("Et from Towers",1);
  hists_["JetTwrEta"]         = dqmStore_->book1D("JetTwrEta" ,"jet towers Eta ",nBins_,-6, 6);
  hists_["JetTwrEta"]         ->setAxisTitle("Eta from Towers",1);
  hists_["JetTwrPhi"]         = dqmStore_->book1D("JetTwrPhi" ,"jet towers Phi ",25,-3.2, 3.2);
  hists_["JetTwrPhi"]         ->setAxisTitle("Phi from Towers",1);
  hists_["JetTwrSumPt"]       = dqmStore_->book1D("JetTwrSumPt" ,"Sum of Pt towers",nBins_,0, 100);
  hists_["JetTwrSumPt"]       ->setAxisTitle("Sum Pt Towers",1);
  hists_["JetdiffTwrSumPt"]   = dqmStore_->book1D("JetdiffTwrSumPt" ,"Diff between Sum of Pt towers and Pt Jet",nBins_,0, 50);
  hists_["JetdiffTwrSumPt"]   ->setAxisTitle("SumPtTowers - JetPt",1);
  
  //October exercise quantities:
  hists_["ptHighest_pt"]   = dqmStore_->book1D("ptHighest_pt" ,"pt of jet with highest pt", 100 ,0.0, 200.0);
  hists_["ptHighest_pt"]   ->setAxisTitle("p_{T} of jet with highest p_{T}",1);
  hists_["ptSecond_pt"]   = dqmStore_->book1D("ptSecond_pt" ,"pt of jet with second highest pt", 100 ,0.0, 200.0);
  hists_["ptSecond_pt"]   ->setAxisTitle("p_{T} of jet with second highest p_{T}",1);
  hists_["ptThird_pt"]   = dqmStore_->book1D("ptThird_pt" ,"pt of jet with third highest pt", 100 ,0.0, 200.0);
  hists_["ptThird_pt"]   ->setAxisTitle("p_{T} of jet with third highest p_{T}",1);
  hists_["ptFourth_pt"]   = dqmStore_->book1D("ptFourth_pt" ,"pt of jet with fourth highest pt", 100 ,0.0, 200.0);
  hists_["ptFourth_pt"]   ->setAxisTitle("p_{T} of jet with fourth highest p_{T}",1);
  
  hists_["ptHighest_eta"]   = dqmStore_->book1D("ptHighest_eta" ,"#eta of jet with highest pt", 100 ,-3.0, 3.0);
  hists_["ptHighest_eta"]   ->setAxisTitle("#eta of jet with highest p_{T}",1);
  hists_["ptSecond_eta"]   = dqmStore_->book1D("ptSecond_eta" ,"#eta of jet with second highest pt", 100 , -3.0, 3.0);
  hists_["ptSecond_eta"]   ->setAxisTitle("#eta of jet with second highest p_{T}",1);
  hists_["ptThird_eta"]   = dqmStore_->book1D("ptThird_eta" ,"#eta of jet with third highest pt", 100 , -3.0, 3.0);
  hists_["ptThird_eta"]   ->setAxisTitle("#eta of jet with third highest p_{T}",1);
  hists_["ptFourth_eta"]   = dqmStore_->book1D("ptFourth_eta" ,"#eta of jet with fourth highest pt", 100 , -3.0, 3.0);
  hists_["ptFourth_eta"]   ->setAxisTitle("#eta of jet with fourth highest p_{T}",1);
  
  // b-tagging info:
  if(checkBtaggingSet_)
    beginJobBtagging(iSetup);
}

void 
JetChecker::end() 
{
}

std::string 
JetChecker::makeBtagCutHistName(const size_t &iwp)
{
  if(iwp<workingpoints_.size()){
    std::string result;
    // use root TStrings as they are less painful to deal with replacements.
    TString numbercut;
    numbercut+=workingpoints_[iwp].second;
    numbercut.ReplaceAll(".","");
    numbercut.ReplaceAll(" ","");
    result+="JetBtag_NtaggedJets_";
    result+= workingpoints_[iwp].first;
    result+= "_at_";
    result+=numbercut;

    return result;
  }
  return "";
}

std::string
JetChecker::makeBtagHistName(const size_t &index)
{ 
  if(index<btaggingalgonames_.size()){ 
    std::string result="JetBtag_";
    result+=btaggingalgonames_[index];
    edm::LogInfo("Debug|JetChecker") << "adding b-tag algo " << result << std::endl;
    return result;
  }
 
  edm::LogError("InputUnknown") << "JetChecker::makeBtagHistName(): trying to get b-tag algo number: " << index << ", while only " << btaggingalgonames_.size() << " are available!. Returning empty string!";
  return "";
}

void 
JetChecker::beginJobBtagging(const edm::EventSetup & iSetup)
{
  // check
  if(!checkBtaggingSet_)
    return;
  // book histos:
  dqmStore_->setCurrentFolder( relativePath_+"/CaloJets_"+label_ );

  // loop over all available algos, passed via config file
  for(size_t ii=0; ii<btaggingalgonames_.size(); ++ii){
    std::string name = makeBtagHistName(ii);
    edm::LogInfo("Debug|JetChecker") << "created new histogram: " << name << " in folder " <<  relativePath_+"/CaloJets_"+label_  << std::endl;
    hists_[name]        = dqmStore_->book1D(name,name,nBins_,-10,30);
    hists_[name]        ->setAxisTitle(name,1);
  }
  for(size_t ii=0; ii<workingpoints_.size(); ++ii){
    // check that the string makes at least some sense...
    if(workingpoints_[ii].first=="")
      return;
    // and make a name to fill the ME's with
    std::string name=makeBtagCutHistName(ii);
    edm::LogInfo("Debug|JetChecker") << "created new histogram for working point " << name << " in folder  " <<  relativePath_+"/CaloJets_"+label_  << std::endl;
    Int_t nbinstouse=nBins_;
    if(nbinstouse>10)
      nbinstouse=10;
    hists_[name]= dqmStore_->book1D(name,name,nbinstouse,-0.5,nbinstouse-0.5);
    hists_[name] -> setAxisTitle("number of tagged jets",1);
  }
  
  return;
}

// b-tagging analyzer loop:
// as you need possibly many different b-taggers it is useful to pass the event so that we can iterate over all of them.
void 
JetChecker::analyzeWithBjets(const std::vector<reco::CaloJet>& jets,  bool useJES, const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::LogInfo("Debug|JetChecker") << "now looking at the jets, including b-tagging info. there are " << jets.size() << " jets" << std::endl;
  //  check that bool is set correctly
  if(!checkBtaggingSet_)
    return;
  
  // do matching between jets and tags, this is necessary in case the tags were reconstructed with a different jet algo as the jets. For more documentation, see for example:
  // https://twiki.cern.ch/twiki/bin/view/CMS/WorkBookBTagging#BtagEdAnalyzer
  edm::Handle<reco::JetTagCollection> bTagHandle;

  // keep track of the number of tagged events.
  std::vector<size_t> tagcounter;
  for(size_t iwp=0; iwp<workingpoints_.size(); iwp++){
    tagcounter.push_back(0);
  }
  // loop over algos.
  for(size_t ialgo=0; ialgo<btaggingalgonames_.size(); ++ialgo){
    edm::LogInfo("Debug|JetChecker") << "examining jets for tags of type: " << btaggingalgonames_[ialgo] << std::endl;
    iEvent.getByLabel(btaggingalgonames_[ialgo], bTagHandle);
    if(!bTagHandle.isValid()){
      edm::LogInfo("Debug|JetChecker") << "trying to get handle for b-tag algo: " << btaggingalgonames_[ialgo] << " and failed..." << std::endl;
      // do not throw error... to be discussed.
      continue;
    }
    const reco::JetTagCollection & bTags = *(bTagHandle.product());
    std::string name=makeBtagHistName(ialgo);
    // now get tags:
   
    
    for(size_t itag=0; itag<bTags.size(); ++itag){
      //
      // and loop over jets, do association:
      for(size_t ijet=0; ijet<jets.size(); ++ijet){
	if(deltaR((jets)[ijet].eta(),(jets)[ijet].phi(),(bTags)[itag].first->eta(),(bTags)[itag].first->phi())<btaggingMatchDr_){
	  edm::LogInfo("Debug|JetChecker") << "matched jet " << ijet << " to tag " << itag << " for algo " << btaggingalgonames_[ialgo] << std::endl;
	  // fill and break.
	  hists_[name]->Fill((bTags)[itag].second);
	  for(size_t iwp=0; iwp<workingpoints_.size(); iwp++){
	    edm::LogInfo("Debug|JetChecker") << "now comparing algos: " << workingpoints_[iwp].first << " " << btaggingalgonames_[ialgo] << std::endl;
	    if(workingpoints_[iwp].first==btaggingalgonames_[ialgo] && (bTags)[itag].second>workingpoints_[iwp].second) // the right algo and cut passed
	      tagcounter[iwp]++;
	  }
	}
      }
    }    
  }
  // and for each cut fill the jet counts:
  for(size_t iwc=0; iwc<workingpoints_.size(); iwc++)
    hists_[makeBtagCutHistName(iwc)]->Fill(tagcounter[iwc]);
  tagcounter.resize(0);
}
