#include "DQMOffline/Trigger/interface/EleHLTFilterMon.h"
#include "DQMOffline/Trigger/interface/CutCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/MonElemFuncs.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "PhysicsTools/Utilities/interface/deltaR.h"

EleHLTFilterMon::EleHLTFilterMon(const std::string& filterName,TrigCodes::TrigBitSet filterBit):
  filterName_(filterName),
  filterBit_(filterBit)
{
  eleMonElems_.push_back(new MonElemContainer<EgHLTOffEle>());
  eleMonElems_.push_back(new MonElemContainer<EgHLTOffEle>("_posCharge"," q=+1 ",new ChargeCut<EgHLTOffEle>(1)));
  eleMonElems_.push_back(new MonElemContainer<EgHLTOffEle>("_negCharge"," q=-1 ",new ChargeCut<EgHLTOffEle>(-1)));
  for(size_t i=0;i<eleMonElems_.size();i++){
    MonElemFuncs::initStdEleHists(eleMonElems_[i]->monElems(),filterName_+"_gsfEle_passFilter"+eleMonElems_[i]->name());
  }
  
  eleFailMonElems_.push_back(new MonElemContainer<EgHLTOffEle>());
  eleFailMonElems_.push_back(new MonElemContainer<EgHLTOffEle>("_posCharge"," q=+1 ",new ChargeCut<EgHLTOffEle>(1)));
  eleFailMonElems_.push_back(new MonElemContainer<EgHLTOffEle>("_negCharge"," q=-1 ",new ChargeCut<EgHLTOffEle>(-1)));
  for(size_t i=0;i<eleFailMonElems_.size();i++){
    MonElemFuncs::initStdEleHists(eleFailMonElems_[i]->monElems(),filterName_+"_gsfEle_failFilter"+eleMonElems_[i]->name());
  }

 
  int effProbeCutCode = CutCodes::getCode("et:detEta");
  int effTagCutCode = CutCodes::getCode("detEta:crack:sigmaEtaEta:hadem:dPhiIn:dEtaIn");
  int fakeRateProbeCut = CutCodes::getCode("et:detEta:hadem");
  eleEffHists_.push_back(new MonElemContainer<EgHLTOffEle>());
  eleEffHists_.push_back(new MonElemContainer<EgHLTOffEle>("_tagProbe"," Tag and Probe ",new EgTagProbeCut<EgHLTOffEle>(effProbeCutCode,&EgHLTOffEle::cutCode,effTagCutCode,&EgHLTOffEle::tagCutCode)));
  eleEffHists_.push_back(new MonElemContainer<EgHLTOffEle>("_fakeRate"," Fake Rate ",new EgJetTagProbeCut<EgHLTOffEle>(fakeRateProbeCut,&EgHLTOffEle::probeCutCode)));
  for(size_t i=0;i<eleEffHists_.size();i++){ 
    MonElemFuncs::initStdEffHists(eleEffHists_[i]->cutMonElems(),filterName_+"_gsfEle_effVsEt"+eleEffHists_[i]->name(),11,-10.,100.,&EgHLTOffEle::et);
  }

  typedef MonElemManager<ParticlePair<EgHLTOffEle>,float >  DiEleMon;
  diEleMonElems_.push_back(new DiEleMon(filterName_+"_diEle_passFilter_mass",
					filterName_+"_diEle_passFilter Mass;M_{ee} (GeV/c^{2}",
					420,-10.,2000.,&ParticlePair<EgHLTOffEle>::mass));
  
  
  
  
}

EleHLTFilterMon::~EleHLTFilterMon()
{
  for(size_t i=0;i<eleMonElems_.size();i++) delete eleMonElems_[i];
  for(size_t i=0;i<eleFailMonElems_.size();i++) delete eleFailMonElems_[i];
  for(size_t i=0;i<eleEffHists_.size();i++) delete eleEffHists_[i];
  for(size_t i=0;i<diEleMonElems_.size();i++) delete diEleMonElems_[i];
}


void EleHLTFilterMon::fill(const EgHLTOffData& evtData,float weight)
{ 
  for(size_t eleNr=0;eleNr<evtData.eles->size();eleNr++){
    const EgHLTOffEle& ele = (*evtData.eles)[eleNr];
    if((ele.trigBits()&filterBit_)!=0){ //ele passes
      for(size_t monElemNr=0;monElemNr<eleMonElems_.size();monElemNr++) eleMonElems_[monElemNr]->fill(ele,evtData,weight);
      for(size_t monElemNr=0;monElemNr<eleEffHists_.size();monElemNr++) eleEffHists_[monElemNr]->fill(ele,evtData,weight);
    }else { //ele didnt pass trigger
      for(size_t monElemNr=0;monElemNr<eleFailMonElems_.size();monElemNr++) eleFailMonElems_[monElemNr]->fill(ele,evtData,weight);
    }
  }//end loop over electrons

  if((evtData.evtTrigBits&filterBit_)!=0){
    for(size_t monElemNr=0;monElemNr<diEleMonElems_.size();monElemNr++){
      for(size_t ele1Nr=0;ele1Nr<evtData.eles->size();ele1Nr++){
	for(size_t ele2Nr=ele1Nr+1;ele2Nr<evtData.eles->size();ele2Nr++){
	  const EgHLTOffEle& ele1 = (*evtData.eles)[ele1Nr];
	  const EgHLTOffEle& ele2 = (*evtData.eles)[ele2Nr];
	  //  edm::LogInfo("EleHLTFilterMon") << "filling "<<monElemNr<<" ele1 "<<ele1Nr<<" ele2 "<<ele2Nr<<" nr ele "<<evtData.eles->size();
	  diEleMonElems_[monElemNr]->fill(ParticlePair<EgHLTOffEle>(ele1,ele2),weight);
	}//end inner ele loop
      }//end outer ele loop
    }//end di ele mon elem loop
  }//end check if filter is present
}




