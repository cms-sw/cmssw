/* \class ZHLTFilter
 *
 * \author Pasquale Noli, Universita' di Napoli & INFN Napoli
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"
#include <string>
using namespace std;
namespace modules {
  struct ZHLTMatchFilter {
    ZHLTMatchFilter(const edm::ParameterSet& cfg) :
    cond_(cfg.getParameter<std::string >("condition")),
    hltPath_(cfg.getParameter<std::string >("hltPath")){ }
    bool operator()(const reco::Candidate & z) const { 
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
      const reco::Candidate* dau0 = z.daughter(0);
      const reco::Candidate * m0 = &*dau0->masterClone();
      const pat::Muon * mu0 = dynamic_cast<const pat::Muon*>(m0);//cast in patMuon
      bool firstdismuon = (dau0->isGlobalMuon() ? true : false); 
      bool firstdisStandAlone = (dau0->isStandAloneMuon() ? true : false); 
      if(mu0 != 0){
	const std::vector<pat::TriggerPrimitive> & trig0 =mu0->triggerMatches();//vector of triggerPrimitive
	int dimTrig0 = trig0.size();
	if(dimTrig0 !=0){
	  for(int j = 0; j < dimTrig0 ; ++j){
	    const std::string  filtername = trig0[j].filterName();
	    if(filtername == hltPath_){ 
	      singleTrigFlag0 = true;
	    }
	  }
	}
      }
      const reco::Candidate* dau1 = z.daughter(1);
      const reco::Candidate * m1 = &*dau1->masterClone();
      bool secondismuon = (dau1->isGlobalMuon() ? true : false);    
      bool secondisStandAlone = (dau1->isStandAloneMuon() ? true : false); 
      const pat::Muon * mu1 = dynamic_cast<const pat::Muon*>(m1);
      if(mu1 != 0 && (secondismuon ||secondisStandAlone) ){
	const std::vector<pat::TriggerPrimitive> & trig1 =mu1->triggerMatches();
	int dimTrig1 = trig1.size();
	if(dimTrig1 !=0){
	  for(int j = 0; j < dimTrig1 ; ++j){
	    const std::string  filtername = trig1[j].filterName();
	    if(filtername == hltPath_){ 
	      singleTrigFlag1 = true;
	    }
	  }
	}
      }
      if(!singleTrigFlag0 && !singleTrigFlag1)return false;
      if((singleTrigFlag0 && singleTrigFlag1) && secondismuon ) bothTriggerFlag = true;
      if(((singleTrigFlag0 && !singleTrigFlag1) && secondismuon) || ((!singleTrigFlag0 && singleTrigFlag1) && secondismuon)) exactlyOneTriggerFlag = true;
      if(((singleTrigFlag0  && firstdismuon) && secondisStandAlone) || ((singleTrigFlag1 && secondismuon) && firstdisStandAlone))globalisTriggerFlag = true;
      if((singleTrigFlag0 && !singleTrigFlag1) && !secondismuon) FirstTriggerFlag = true;
      if((singleTrigFlag0 || singleTrigFlag1) && secondismuon) atLeastOneTriggerFlag=true;
      if(cond_=="exactlyOneMatched") return exactlyOneTriggerFlag;
      if(cond_=="atLeastOneMatched") return atLeastOneTriggerFlag;
      if(cond_=="bothMatched") return bothTriggerFlag;
      if(cond_=="firstMatched") return FirstTriggerFlag; 
      if(cond_=="globalisMatched") return globalisTriggerFlag; 
      
      return false;
      }
  
  private:
  std::string cond_ ;
  std::string hltPath_;
    
  };
}

typedef SingleObjectSelector<
  edm::View<reco::Candidate>,
  modules::ZHLTMatchFilter
  > ZHLTMatchFilter;

DEFINE_FWK_MODULE( ZHLTMatchFilter );
