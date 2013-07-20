// $Id: HLTMonSimpleBTag.cc,v 1.4 2011/04/15 10:15:36 fblekman Exp $
// See header file for information. 
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/HLTMonSimpleBTag.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DQMServices/Core/interface/DQMNet.h"


using namespace edm;

HLTMonSimpleBTag::HLTMonSimpleBTag(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{
  LogDebug("HLTMonSimpleBTag") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogWarning("Status") << "unable to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  
  dirname_="HLT/HLTMonSimpleBTag" ;
  
  if (dbe_ != 0 ) {
    LogDebug("Status") << "Setting current directory to " << dirname_;
    dbe_->setCurrentFolder(dirname_);
  }
  
  
  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",200.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",50);
  dRTrigObjMatch_ = iConfig.getUntrackedParameter<double>("dRMatch",0.3);
  refresheff_ = iConfig.getUntrackedParameter<unsigned int>("Nrefresh",10);
 
 

  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> filters = 
    iConfig.getParameter<std::vector<edm::ParameterSet> >("filters");
  for(std::vector<edm::ParameterSet>::iterator 
	filterconf = filters.begin() ; filterconf != filters.end(); 
      filterconf++) {
    std::string me  = filterconf->getParameter<std::string>("name");
    std::string denom = filterconf->getParameter<std::string>("refname");
    // only fill if both triggers are not there yet.
    if(hltPaths_.find(me)==hltPaths_.end())
      hltPaths_.push_back(PathInfo(me,ptMin_, ptMax_));
    if(hltPaths_.find(denom)==hltPaths_.end())
      hltPaths_.push_back(PathInfo(denom, ptMin_, ptMax_));
    
    std::string effname = makeEffName(me,denom);
    std::string numername=makeEffNumeratorName(me,denom);
    std::pair<std::string,std::string> trigpair(me,denom);
    if(find(triggerMap_.begin(),triggerMap_.end(),trigpair)==triggerMap_.end())
      triggerMap_.push_back(trigpair);
    if(hltEfficiencies_.find(effname)==hltEfficiencies_.end())
      hltEfficiencies_.push_back(PathInfo(effname,ptMin_,ptMax_));
    if(hltEfficiencies_.find(numername)==hltEfficiencies_.end())
      hltEfficiencies_.push_back(PathInfo(numername,ptMin_,ptMax_));

  }
  triggerSummaryLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
}


HLTMonSimpleBTag::~HLTMonSimpleBTag()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HLTMonSimpleBTag::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace trigger;
  ++nev_;
  LogDebug("Status")<< "analyze" ;
  
  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogInfo("Status") << "Summary HLT object (TriggerEvent) not found, "
      "skipping event"; 
    return;
  }
  
  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());

  for ( size_t ia = 0; ia < triggerObj->sizeFilters(); ++ ia) {
    std::string fullname = triggerObj->filterTag(ia).encode();
    // the name can have in it the module label as well as the process and
    // other labels - strip 'em
    std::string name;
    size_t p = fullname.find_first_of(':');
    if ( p != std::string::npos) {
      name = fullname.substr(0, p);
    }
    else {
      name = fullname;
    }
    
    LogDebug("Parameter")  << "filter " << ia << ", full name = " << fullname
			  << ", p = " << p 
			  << ", abbreviated = " << name ;
    //      std::cout << std::endl;
    // check that trigger is in 'watch list'
    PathInfoCollection::iterator pic = hltPaths_.find(name);
    if(pic==hltPaths_.end())
      continue;
    
    // find keys 
    
    const trigger::Keys & k = triggerObj->filterKeys(ia);
    for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
      LogDebug("Parameters")  << name << "(" << ki-k.begin() << "): pt, eta, phi = " 
			     << toc[*ki].pt() << ", "
			     << toc[*ki].eta() << ", "
			      << toc[*ki].phi()  <<","
			      << toc[*ki].id();
      pic->getEtHisto()->Fill(toc[*ki].pt());
      pic->getEtaHisto()->Fill(toc[*ki].eta());
      pic->getPhiHisto()->Fill(toc[*ki].phi());
      pic->getEtaVsPhiHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
    }

    // check which trigger this trigger is reference to, fill those histograms separately:
    for(std::vector<std::pair<std::string, std::string> >::iterator matchIter = triggerMap_.begin(); matchIter!=triggerMap_.end(); ++matchIter){
      // only do anything if you actually have a match
      if(matchIter->second!=name)
	continue;
      LogDebug("HLTMonSimpleBTag") << "found match! " << " " << matchIter->first << " " << matchIter->second ;
      // now go find this one in the trigger collection... this is somewhat painful :(
      for(size_t ib = 0; ib < triggerObj->sizeFilters(); ++ ib) {
	if(ib==ia)
	  continue;
	std::string fullname_b = triggerObj->filterTag(ib).encode();
	// the name can have in it the module label as well as the process and
	// other labels - strip 'em
	std::string name_b;
	size_t p_b = fullname_b.find_first_of(':');
	if ( p_b != std::string::npos) {
	  name_b = fullname_b.substr(0, p_b);
	}
	else {
	  name_b = fullname_b;
	}
	// this is where the matching to the trigger array happens
	if(name_b!=matchIter->first)
	  continue;
	
	// ok, now we have two matching triggers with indices ia and ib in the trigger index. Get the keys for ib.
	
	// find the correct monitoring element:
	std::string numeratorname = makeEffNumeratorName(matchIter->first,matchIter->second);
	PathInfoCollection::iterator pic_b = hltEfficiencies_.find(numeratorname);
	if(pic_b==hltEfficiencies_.end())
	  continue;

	const trigger::Keys & k_b = triggerObj->filterKeys(ib);

	const trigger::Keys & k = triggerObj->filterKeys(ia);
	for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	  for (trigger::Keys::const_iterator ki_b = k_b.begin(); ki_b !=k_b.end(); ++ki_b ) {
	    // do cone match...
	    if(reco::deltaR(toc[*ki].eta(),toc[*ki_b].phi(),toc[*ki].eta(),toc[*ki_b].phi())>dRTrigObjMatch_)
	      continue;
	    
	    LogDebug("Parameters") << "matched pt, eta, phi = " 
				   << toc[*ki].pt() << ", "
				   << toc[*ki].eta() << ", "
				   << toc[*ki].phi() << " to " 
				   << toc[*ki_b].pt() << ", "
				   << toc[*ki_b].eta() << ", "
				   << toc[*ki_b].phi() << " (using the first to fill histo to avoid division problems...)";
	    // as these are going to be divided it is important to fill numerator and denominator with the same pT spectrum. So this is why these are filled with ki objects instead of ki_b objects...
	    pic_b->getEtHisto()->Fill(toc[*ki].pt());
	    pic_b->getEtaHisto()->Fill(toc[*ki].eta());
	    pic_b->getPhiHisto()->Fill(toc[*ki].phi());
	    pic_b->getEtaVsPhiHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	    // for now just take the first match there are two keys within the same cone...
	    break;
	  }
	}
      }
    }
  }
  // and calculate efficiencies, only if the number of events is the refresh rate:
  if(nev_%refresheff_==0 && nev_!=1){
    calcEff();
  }
}


// -- method called once each job just before starting event loop  --------
void 
HLTMonSimpleBTag::beginJob()
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

    for(PathInfoCollection::iterator v = hltPaths_.begin();
	v!= hltPaths_.end(); ++v ) {
      MonitorElement *et, *eta, *phi, *etavsphi=0;
      std::string histoname(v->getName()+"_et");
      std::string title(v->getName()+" E_t");
      et =  dbe->book1D(histoname.c_str(),
			title.c_str(),nBins_,
			v->getPtMin(),
			v->getPtMax());
      
      histoname = v->getName()+"_eta";
      title = v->getName()+" #eta";
      eta =  dbe->book1D(histoname.c_str(),
			 title.c_str(),nBins_/2,-2.7,2.7);
      
      histoname = v->getName()+"_phi";
      title = v->getName()+" #phi";
      phi =  dbe->book1D(histoname.c_str(),
			 histoname.c_str(),nBins_/2,-3.14,3.14);
      
      
      histoname = v->getName()+"_etaphi";
      title = v->getName()+" #eta vs #phi";
      etavsphi =  dbe->book2D(histoname.c_str(),
			      title.c_str(),
			      nBins_/2,-2.7,2.7,
			      nBins_/2,-3.14, 3.14);
      
      v->setHistos( et, eta, phi, etavsphi);
    }

    for(PathInfoCollection::iterator v = hltEfficiencies_.begin();
	v!= hltEfficiencies_.end(); ++v ) {
      MonitorElement *et, *eta, *phi, *etavsphi=0;
      std::string histoname(v->getName()+"_et");
      std::string title(v->getName()+" E_t");
      et =  dbe->book1D(histoname.c_str(),
			title.c_str(),nBins_,
			v->getPtMin(),
			v->getPtMax());
      
      histoname = v->getName()+"_eta";
      title = v->getName()+" #eta";
      eta =  dbe->book1D(histoname.c_str(),
			 title.c_str(),nBins_/2,-2.7,2.7);
      
      histoname = v->getName()+"_phi";
      title = v->getName()+" #phi";
      phi =  dbe->book1D(histoname.c_str(),
			 histoname.c_str(),nBins_/2,-3.14,3.14);
      
      
      histoname = v->getName()+"_etaphi";
      title = v->getName()+" #eta vs #phi";
      etavsphi =  dbe->book2D(histoname.c_str(),
			      title.c_str(),
			      nBins_/2,-2.7,2.7,
			      nBins_/2,-3.14, 3.14);
      
      v->setHistos( et, eta, phi, etavsphi);
    }
  }
}

// - method called once each job just after ending the event loop  ------------
void 
HLTMonSimpleBTag::endJob() 
{
   LogInfo("Status") << "endJob: analyzed " << nev_ << " events";
   return;
}


// BeginRun
void HLTMonSimpleBTag::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("Status") << "beginRun, run " << run.id();
}

/// EndRun
void HLTMonSimpleBTag::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("Status") << "final re-calculation efficiencies!" ;
  calcEff();
  LogDebug("Status") << "endRun, run " << run.id();
  
}


/// calcEff: calculates efficiency using histograms booked in  std::map<std::string,std::string> triggerMap_; 

void 
HLTMonSimpleBTag::calcEff(void){
 
  for( std::vector<std::pair<std::string,std::string> >::iterator iter = triggerMap_.begin(); iter!=triggerMap_.end(); iter++){
    if(hltPaths_.find(iter->first)==hltPaths_.end())
      continue;
    if(hltPaths_.find(iter->second)==hltPaths_.end())
      continue;
    
    std::string effname = makeEffName(iter->first,iter->second);
    std::string numeratorname = makeEffNumeratorName(iter->first,iter->second);
    LogDebug("HLTMonBTagSlim::calcEff") << "calculating efficiencies for histogram with effname=" << effname << " using also histogram " << numeratorname ;
    PathInfoCollection::iterator effHists = hltEfficiencies_.find(effname);
    if(effHists==hltEfficiencies_.end())
      continue;
    PathInfoCollection::iterator numerHists = hltEfficiencies_.find(numeratorname);
    if(numerHists==hltEfficiencies_.end())
      continue;
    LogDebug("HLTMonBTagSlim::calcEff") << "found histo with name " << effname << " and " << numeratorname << "!" ;
    // do the hists all separately:

    doEffCalc(effHists->getEtHisto(),numerHists->getEtHisto(),hltPaths_.find(iter->second)->getEtHisto());
    doEffCalc(effHists->getEtaHisto(),numerHists->getEtaHisto(),hltPaths_.find(iter->second)->getEtaHisto());
    doEffCalc(effHists->getPhiHisto(),numerHists->getPhiHisto(),hltPaths_.find(iter->second)->getPhiHisto());
    doEffCalc(effHists->getEtaVsPhiHisto(),numerHists->getEtaVsPhiHisto(),hltPaths_.find(iter->second)->getEtaVsPhiHisto());


  }
  LogDebug("HLTMonBTagSlim::calcEff") << "done with efficiencies!" ;
}

// function to actually do the efficiency calculation per-bin
void 
HLTMonSimpleBTag::doEffCalc(MonitorElement *eff, MonitorElement *num, MonitorElement *denom){
  double x,y,errx,erry;

  // presuming error propagation is non-quadratic (binominal):
  // so: err!= Sqrt(errx^2/x^2+erry^2/y^2) but instead
  // 
  // err=sqrt(errx^2/x^2+erry^2/y^2)* x/y*(1-x/y)
  bool is1d=false;
  bool is2d=false;
  if(num->kind()==DQMNet::DQM_PROP_TYPE_TH1F || num->kind()==DQMNet::DQM_PROP_TYPE_TH1S || num->kind()== DQMNet::DQM_PROP_TYPE_TH1D)
    is1d=true;
  else if(num->kind()==DQMNet::DQM_PROP_TYPE_TH2F || num->kind()==DQMNet::DQM_PROP_TYPE_TH2S || num->kind()== DQMNet::DQM_PROP_TYPE_TH2D)
    is2d=true;
  

  for(int ibin=0; ibin<=eff->getNbinsX() && ibin<=num->getNbinsX(); ibin++){
    if(is1d){ // 1D histograms!
      x=num->getBinContent(ibin);
      errx=num->getBinError(ibin);
      y=denom->getBinContent(ibin);
      erry=denom->getBinError(ibin);
      if(fabs(y)<0.00001 || fabs(x)<0.0001){// very stringent non-zero!
	eff->setBinContent(ibin,0);
	eff->setBinError(ibin,0);
	continue;
      }

      LogDebug("HLTMonSimpleBTag::calcEff()") << eff->getName() << " (" << ibin << ") " <<  x << " " << y;
      eff->setBinContent(ibin,x/y);
      eff->setBinError(ibin,sqrt((errx*errx)/(x*x)+(erry*erry)/(y*y))* (x/y)*(1-(x/y)));
    }
    else if(is2d){ // 2D histograms!
      for(int jbin=0; jbin<=num->getNbinsY(); jbin++){
	x=num->getBinContent(ibin,jbin);
	errx=num->getBinError(ibin,jbin);
	y=denom->getBinContent(ibin,jbin);
	erry=denom->getBinError(ibin,jbin);
	if(fabs(y)<0.0001 || fabs(x)<0.0001){ // very stringent non-zero!
	  eff->setBinContent(ibin,jbin,0.);
	  eff->setBinError(ibin,jbin,0.);
	  continue;
	}
	LogDebug("HLTMonSimpleBTag::calcEff()") << eff->getName() << " (" << ibin<< "," << jbin << ") " <<  x << " " << y ;
	  
	eff->setBinContent(ibin,jbin,x/y);
	eff->setBinError(ibin,jbin,sqrt((errx*errx)/(x*x)+(erry*erry)/(y*y))* (x/y)*(1-(x/y)));
      }
    }
  }
}
  
