#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include <string>
using namespace std;
using namespace reco;
using namespace isodeposit;
namespace edm { class EventSetup; }

bool IsMuMatchedToHLTMu ( const reco::Candidate * dau, std::vector<reco::Particle> HLTMu , double DR, double DPtRel ) {
  size_t dim =  HLTMu.size();
  //std::cout<< "HLT muons size== " << dim << endl; 
  //std::costd::cout << "dau  pt " << dau->pt()  << std::endl;                
  size_t nPass=0;
  if (dim==0) return false;
  for (size_t k =0; k< dim; k++ ) {
    //std::cout << "HLT Mu " << k << " pt " << HLTMu[k].pt()  << std::endl;    
   if (  (deltaR(HLTMu[k], *dau) < DR)   && (fabs(HLTMu[k].pt() - dau->pt())/ HLTMu[k].pt()<DPtRel)){     nPass++ ;
   //      std::cout << "matched HLT Mu " << k << " pt " << HLTMu[k].pt()  << std::endl;   

    }
  }
  return (nPass>0);
}

bool IsMuMatchedToHLTSingleMu ( const reco::Candidate * dau, reco::Particle HLTMu , double DR, double DPtRel ) {
  size_t nPass=0;
  if (  (deltaR(HLTMu, *dau) < DR)   && (fabs(HLTMu.pt() - dau->pt())/ HLTMu.pt()<DPtRel)) {
    nPass++;
    //std::cout << "HLT Mu matched" <<   " .pt= " << HLTMu.pt()  << std::endl;   
    //std::cout << "Mu matched" <<   " .pt= " << dau->pt()  << std::endl;   
    //std::cout << "deltaR(HLTMu, *dau)= "  << deltaR(HLTMu, *dau) << std::endl;   double DPtRelMu =  fabs(HLTMu.pt() - dau->pt())/ HLTMu.pt();
    //std::cout << "fabs(HLTMu.pt() - dau->pt())/ HLTMu.pt() =  "  << DPtRelMu << std::endl;
  }
  return (nPass>0);
}

double muIsolation( IsoDeposit isdep, const reco::Candidate * cand, double dRTrk, double dRVetoTrk , double ptThreshold )  {        
    Direction dir = Direction(cand->eta(), cand->phi());
    AbsVetos vetosTrk;
    vetosTrk.push_back(new ConeVeto( dir, dRVetoTrk ));
    vetosTrk.push_back(new ThresholdVeto( ptThreshold ));
    double isovalueTrk = (isdep.sumWithin(dRTrk,vetosTrk));
    // for relative isolation 
    // isovalueTrk /= cand->pt();
    return isovalueTrk;
  }

class ZGoldenFilter {

public:
  ZGoldenFilter(const edm::ParameterSet&  );
  bool operator()(const reco::Candidate & ) const;
  void newEvent (const edm::Event&, const edm::EventSetup&);
  edm::InputTag  muIso_;
  double isoCut_;
  double deltaRTrk_,  ptThreshold_,  deltaRVetoTrk_ ;
  edm::Handle<reco::IsoDepositMap>  hMuIso_;
  edm::InputTag trigTag_;   
  edm::InputTag  trigEv_;
  std::string cond_ ;
  std::string hltPath_;
  std::string L3FilterName_;
  edm::Handle<edm::TriggerResults> triggerResults_;
  edm::TriggerNames trigNames_;
  edm::Handle< trigger::TriggerEvent > handleTriggerEvent_;
  double maxDPtRel_, maxDeltaR_ ;
}; 

   ZGoldenFilter::ZGoldenFilter(const edm::ParameterSet& cfg ) :
   muIso_(cfg.getParameter<edm::InputTag>("muonIsolations")),
   isoCut_(cfg.getParameter<double>("isoCut")),
   deltaRTrk_(cfg.getParameter<double>("deltaRTrk")),
   ptThreshold_(cfg.getParameter<double>("ptThreshold")),
   deltaRVetoTrk_(cfg.getParameter<double>("deltaRVetoTrk")),
   trigTag_(cfg.getParameter<edm::InputTag> ("TrigTag")),
   trigEv_(cfg.getParameter<edm::InputTag> ("triggerEvent")),
   cond_(cfg.getParameter<std::string >("condition")),
   hltPath_(cfg.getParameter<std::string >("hltPath")),
   L3FilterName_(cfg.getParameter<std::string >("L3FilterName")),
   maxDPtRel_(cfg.getParameter<double>("maxDPtRel")),
   maxDeltaR_(cfg.getParameter<double>("maxDeltaR")){
}

    void  ZGoldenFilter::newEvent(const edm::Event& ev, const edm::EventSetup& ){
      if (!ev.getByLabel(muIso_, hMuIso_)){
	edm::LogWarning("") << ">>> TRIGGER collection does not exist !!!";
	return ;     
    }
      ev.getByLabel(muIso_, hMuIso_);

    if (!ev.getByLabel(trigTag_, triggerResults_)) {
      edm::LogWarning("") << ">>> TRIGGER collection does not exist !!!";
      return ;
    }
    ev.getByLabel(trigTag_, triggerResults_);
    // std::cout << " before worning .... "  << std::endl; 
    //std::cout << " before trigNames.init "  << std::endl; 
    //std::cout << " triggerResult.size " << triggerResults_->size() << std::endl;
    trigNames_.init(*triggerResults_);
   
    //std::cout << " after trigNames.init "  << std::endl; 

   
    if ( ! ev.getByLabel( trigEv_, handleTriggerEvent_ ) ) {
      edm::LogError( "errorTriggerEventValid" ) << "trigger::TriggerEvent product with InputTag " << trigEv_.encode() << " not in event";
     return;
    }
  ev.getByLabel( trigEv_, handleTriggerEvent_ ); 
    } 
    


  bool ZGoldenFilter::operator()(const reco::Candidate & z) const {     
    //  int i = newEvent( edm::Event& const , edm::EventSetup& const );
    assert(z.numberOfDaughters()==2);
    bool singleTrigFlag0 = false;
    bool singleTrigFlag1 = false;
    bool exactlyOneTriggerFlag = false;
    bool bothTriggerFlag = false;
    bool atLeastOneTriggerFlag=false;
    bool FirstTriggerFlag = false;
    bool globalisTriggerFlag =false;
    if((((cond_ !="exactlyOneMatched" && cond_!="atLeastOneMatched") && cond_ !="bothMatched") && cond_ != "firstMatched") && cond_ != "globalisMatched")
      throw edm::Exception(edm::errors::Configuration) 
	<< "Invalid condition type: " << cond_ << ". Valid types are:"
	  << " exactlyOneMatched, atLeastOneMatched, bothMatched, firstMatched,globalisMatched\n";
    const reco::Candidate * dau0 = z.daughter(0);
    const reco::Candidate * dau1 = z.daughter(1);
    //std::cout << " dau0 pt " << dau0->pt() << std::endl;
    //std::cout << " dau1 pt " << dau1->pt() << std::endl;
    const trigger::TriggerObjectCollection & toc(handleTriggerEvent_->getObjects());
    
    size_t nMuHLT =0;
    std::vector<reco::Particle>  HLTMuMatched; 
    
    for ( size_t ia = 0; ia < handleTriggerEvent_->sizeFilters(); ++ ia) {
      std::string fullname = handleTriggerEvent_->filterTag(ia).encode();
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
      
      //std::cout << "filter " << ia << ", full name = " << fullname
      //                       << ", p = " << p 
      //		 << ", abbreviated = " << name <<std::endl;
      // std::cout << "name " << name << std::endl;
      if ( &toc !=0 ) {
	const trigger::Keys & k = handleTriggerEvent_->filterKeys(ia);
	for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	  if (name == L3FilterName_  ) { 
	  
	    HLTMuMatched.push_back(toc[*ki].particle());
	    nMuHLT++;     
	  }
	}    
      }
    }
    
 


 if(!(dau0->hasMasterClone()&&dau1->hasMasterClone()))
      throw edm::Exception(edm::errors::InvalidReference) 
	<< "Candidate daughters have no master clone\n"; 
  
  double iso0 =-1;
  double iso1=-1;
  const reco::CandidateBaseRef & c0 =  dau0->masterClone();
  const reco::CandidateBaseRef & c1 =  dau1->masterClone();

    
 
   reco::IsoDeposit isdep0_((*hMuIso_)[c0]);
   //isdep0_.setVeto(deltaRVetoTrk_);
   iso0 = muIsolation (isdep0_, dau0,  deltaRTrk_, deltaRVetoTrk_ ,  ptThreshold_ );   
   if (iso0 > isoCut_) return false;

   reco::IsoDeposit isdep1_((*hMuIso_)[c1]);
   iso1 = muIsolation (isdep1_, dau1,  deltaRTrk_, deltaRVetoTrk_ ,  ptThreshold_ );    

   if (iso1 > isoCut_) return false;


   
    bool trigger_fired = false;
  
    for (unsigned int i=0; i<triggerResults_->size(); i++) {
      std::string trigName = trigNames_.triggerName(i);
      //     std::cout << " trigName " << trigName << std::endl; 
      if ( trigName == hltPath_ && triggerResults_->accept(i)) trigger_fired = true;
    }
    
    
    // std::cout << ">>> Trigger bit: " << trigger_fired << " (" << hltPath_ << ")" << std::endl;
  
    bool firstdismuon = (dau0->isGlobalMuon() ? true : false); 
    bool firstdisStandAlone = (dau0->isStandAloneMuon() ? true : false); 
    std::vector<bool> IsDau0Matched_;
    std::vector<bool> IsDau1Matched_;
    if(dau0 != 0){
      // checking if dau0 is matched to any HLT muon....
         singleTrigFlag0 = IsMuMatchedToHLTMu ( dau0,  HLTMuMatched ,maxDeltaR_, maxDPtRel_ );
         
	for (size_t y=0; y< HLTMuMatched.size(); y++  ){
	  IsDau0Matched_.push_back( IsMuMatchedToHLTSingleMu ( dau0,  HLTMuMatched[y] ,maxDeltaR_, maxDPtRel_ )); 
          
         } 
    }
    
    bool secondismuon = (dau1->isGlobalMuon() ? true : false);    
    bool secondisStandAlone = (dau1->isStandAloneMuon() ? true : false); 
    if(dau1 != 0 && (secondismuon ||secondisStandAlone) ){
      // checking if dau1 is matched to any HLT muon....
      singleTrigFlag1 = IsMuMatchedToHLTMu ( dau1,  HLTMuMatched ,maxDeltaR_, maxDPtRel_ ); 
      for (size_t y=0; y< HLTMuMatched.size(); y++  ){
	IsDau1Matched_.push_back( IsMuMatchedToHLTSingleMu ( dau1,  HLTMuMatched[y] ,maxDeltaR_, maxDPtRel_ )); 
	
      } 
      
    }
    if ( (IsDau0Matched_.size() * IsDau1Matched_.size())!=0 ) {
    for (size_t y=0; y< IsDau1Matched_.size(); y++ ){
      if ( IsDau0Matched_[y]== true && IsDau1Matched_[y]== true ){
	std::cout<< "WARNING--> I'm matching the two muons to the same HLT muon....." << std::endl;}
    } 
    }
    
    
    if(!singleTrigFlag0 && !singleTrigFlag1)return false;
    if((singleTrigFlag0 && singleTrigFlag1) && secondismuon ) bothTriggerFlag = true;
    if(((singleTrigFlag0 && !singleTrigFlag1) && secondismuon) || ((!singleTrigFlag0 && singleTrigFlag1) && secondismuon)) exactlyOneTriggerFlag = true;
    if((((singleTrigFlag0  && firstdismuon) && secondisStandAlone) && !secondismuon ) || (((singleTrigFlag1 && secondismuon) && firstdisStandAlone) && !firstdismuon))globalisTriggerFlag = true;
    if((singleTrigFlag0 && !singleTrigFlag1) && !secondismuon) FirstTriggerFlag = true;
    if((singleTrigFlag0 || singleTrigFlag1) && secondismuon) atLeastOneTriggerFlag=true;
    if(cond_=="exactlyOneMatched") return exactlyOneTriggerFlag;
    if(cond_=="atLeastOneMatched") return atLeastOneTriggerFlag;
    if(cond_=="bothMatched") return bothTriggerFlag;
    if(cond_=="firstMatched") return FirstTriggerFlag; 
    if(cond_=="globalisMatched") return globalisTriggerFlag; 
    return false; 
  }
    

  
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

EVENTSETUP_STD_INIT(ZGoldenFilter);


typedef SingleObjectSelector<
  edm::View<reco::Candidate>,
  AndSelector< ZGoldenFilter, 	StringCutObjectSelector<reco::Candidate> >
  > ZGoldenSelectorAndFilter;


DEFINE_FWK_MODULE( ZGoldenSelectorAndFilter );

