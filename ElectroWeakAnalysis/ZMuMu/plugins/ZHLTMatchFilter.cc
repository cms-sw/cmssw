/* \class ZHLTFilter
 *
 * \author Pasquale Noli, Universita' di Napoli & INFN Napoli
 *
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include <string>
using namespace std;
namespace modules {
  struct ZHLTMatchFilter {
    ZHLTMatchFilter(const edm::ParameterSet& cfg, edm::ConsumesCollector & iC) :
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
      bool firstismuon = (dau0->isGlobalMuon() ? true : false);
      bool firstisStandAlone = (dau0->isStandAloneMuon() ? true : false);
      bool firstisTrackerMuon = (dau0->isTrackerMuon() ? true : false);
      if(mu0 != nullptr && (firstismuon ||firstisStandAlone||firstisTrackerMuon )){
	// get the vector of trigger objects matched to the muon corresponding to hltPath_
	const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	  mu0->triggerObjectMatchesByPath( hltPath_ );

	int dimTrig0 = mu0HLTMatches.size();
	if(dimTrig0 !=0 ){
	  singleTrigFlag0 = true;
	}
      }
      const reco::Candidate* dau1 = z.daughter(1);
      const reco::Candidate * m1 = &*dau1->masterClone();
      bool secondismuon = (dau1->isGlobalMuon() ? true : false);
      bool secondisStandAlone = (dau1->isStandAloneMuon() ? true : false);
      bool secondisTrackerMuon = (dau1->isTrackerMuon() ? true : false);
      const pat::Muon * mu1 = dynamic_cast<const pat::Muon*>(m1);
      if(mu1 != nullptr && (secondismuon ||secondisStandAlone||secondisTrackerMuon ) ){
	// get the vector of trigger objects matched to the muon corresponding to hltPath_
	const pat::TriggerObjectStandAloneCollection mu1HLTMatches =
	  mu1->triggerObjectMatchesByPath( hltPath_ );

	int dimTrig1 = mu1HLTMatches.size();
	if(dimTrig1 !=0){
	  singleTrigFlag1 = true;
	}
      }
      if(!singleTrigFlag0 && !singleTrigFlag1)return false;
      if((singleTrigFlag0 && singleTrigFlag1) && firstismuon && secondismuon ) bothTriggerFlag = true;
      if(((singleTrigFlag0 && !singleTrigFlag1) && firstismuon && secondismuon) || ((!singleTrigFlag0 && singleTrigFlag1) && firstismuon && secondismuon)) exactlyOneTriggerFlag = true;
      if((((singleTrigFlag0  && firstismuon) && (secondisStandAlone || secondisTrackerMuon ) ) && !secondismuon ) || (((singleTrigFlag1 && secondismuon) && (firstisStandAlone|| firstisTrackerMuon) ) && !firstismuon))globalisTriggerFlag = true;

      if((singleTrigFlag0 && !singleTrigFlag1) && !secondismuon) FirstTriggerFlag = true;
      if((singleTrigFlag0 || singleTrigFlag1) && firstismuon && secondismuon) atLeastOneTriggerFlag=true;
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
