#include "DQMOffline/Trigger/interface/EgHLTEleHLTFilterMon.h"
#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/EgHLTBinData.h"
#include "DQMOffline/Trigger/interface/EgHLTCutMasks.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace egHLT;

EleHLTFilterMon::EleHLTFilterMon(MonElemFuncs& monElemFuncs,const std::string& filterName,TrigCodes::TrigBitSet filterBit,const BinData& bins,const CutMasks& masks,bool doHEP):
  filterName_(filterName),
  filterBit_(filterBit),
  doHEP_(doHEP)
{
  bool doChargeSep = false;
  bool monHLTFailedEle = false;
  bool doFakeRate=false;
  bool doTagAndProbe=false;
  bool doN1andSingleEffs=false;
  std::string histname="egamma";

  eleMonElems_.push_back(new MonElemContainer<OffEle>());
  //---Morse-------
  //eleMonElems_.push_back(new MonElemContainer<OffEle>("_cut"," cut, debug hists ",new EgHLTDQMVarCut<OffEle>(~0x0,&OffEle::cutCode)));
  //-----------------
  if(doChargeSep){
    eleMonElems_.push_back(new MonElemContainer<OffEle>("_posCharge"," q=+1 ",new ChargeCut<OffEle>(1)));
    eleMonElems_.push_back(new MonElemContainer<OffEle>("_negCharge"," q=-1 ",new ChargeCut<OffEle>(-1)));
  }

  for(auto & eleMonElem : eleMonElems_){
    if(doHEP_){
                monElemFuncs.initStdEleHistsHEP(eleMonElem->monElems(),filterName,histname+"_passFilter"+eleMonElem->name(),bins);
             }else{  
                   monElemFuncs.initStdEleHists(eleMonElem->monElems(),filterName,filterName_+"_gsfEle_passFilter"+eleMonElem->name(),bins);
                  }
  }
  if(monHLTFailedEle){
    eleFailMonElems_.push_back(new MonElemContainer<OffEle>());
    if(doChargeSep) {
      eleFailMonElems_.push_back(new MonElemContainer<OffEle>("_posCharge"," q=+1 ",new ChargeCut<OffEle>(1)));
      eleFailMonElems_.push_back(new MonElemContainer<OffEle>("_negCharge"," q=-1 ",new ChargeCut<OffEle>(-1)));
    }
  }
  for(size_t i=0;i<eleFailMonElems_.size();i++){
    if(doHEP_){
               monElemFuncs.initStdEleHistsHEP(eleMonElems_[i]->monElems(),filterName,histname+"_passFilter"+eleMonElems_[i]->name(),bins);
              }else{  
                    monElemFuncs.initStdEleHists(eleFailMonElems_[i]->monElems(),filterName,filterName_+"_gsfEle_failFilter"+eleMonElems_[i]->name(),bins);
                   }
  }
 
  int effProbeCutCode = masks.probeEle;
  int effTagCutCode = masks.stdEle;
  int fakeRateProbeCut = masks.fakeEle;
  eleEffHists_.push_back(new MonElemContainer<OffEle>());
  if(doTagAndProbe) eleEffHists_.push_back(new MonElemContainer<OffEle>("_tagProbe"," Tag and Probe ",new EgTagProbeCut<OffEle>(effProbeCutCode,&OffEle::cutCode,effTagCutCode,&OffEle::cutCode)));
  if(doFakeRate) eleEffHists_.push_back(new MonElemContainer<OffEle>("_fakeRate"," Fake Rate ",new EgJetTagProbeCut<OffEle>(fakeRateProbeCut,&OffEle::looseCutCode)));
  if(doN1andSingleEffs){
    for(auto & eleEffHist : eleEffHists_){ 
      monElemFuncs.initStdEffHists(eleEffHist->cutMonElems(),filterName,
				    filterName_+"_gsfEle_effVsEt"+eleEffHist->name(),bins.et,&OffEle::et,masks);
      monElemFuncs.initStdEffHists(eleEffHist->cutMonElems(),filterName,
				    filterName_+"_gsfEle_effVsEta"+eleEffHist->name(),bins.eta,&OffEle::eta,masks); 
      /*  monElemFuncs.initStdEffHists(eleEffHists_[i]->cutMonElems(),filterName,
	  filterName_+"_gsfEle_effVsPhi"+eleEffHists_[i]->name(),bins.phi,&OffEle::phi,masks); */
      // monElemFuncs.initStdEffHists(eleEffHists_[i]->cutMonElems(),filterName,
      //			  filterName_+"_gsfEle_effVsCharge"+eleEffHists_[i]->name(),bins.charge,&OffEle::chargeF);
    }
  }
 if(!doHEP_)
 {
  typedef MonElemManager<ParticlePair<OffEle>,float >  DiEleMon;
  diEleMassBothME_ = new DiEleMon(monElemFuncs.getIB(), filterName_+"_diEle_bothPassFilter_mass",
				  filterName_+"_diEle_bothPassFilter Mass;M_{ee} (GeV/c^{2})",
				  bins.mass.nr,bins.mass.min,bins.mass.max,&ParticlePair<OffEle>::mass);
  diEleMassOnlyOneME_ = new DiEleMon(monElemFuncs.getIB(), filterName_+"_diEle_onlyOnePass Filter_mass",
				     filterName_+"_diEle_onlyOnePassFilter Mass;M_{ee} (GeV/c^{2})",
				     bins.mass.nr,bins.mass.min,bins.mass.max,&ParticlePair<OffEle>::mass);
  
  diEleMassBothHighME_ = new DiEleMon(monElemFuncs.getIB(), filterName_+"_diEle_bothPassFilter_massHigh",
				      filterName_+"_diEle_bothPassFilter Mass;M_{ee} (GeV/c^{2})",
				      bins.massHigh.nr,bins.massHigh.min,bins.massHigh.max,&ParticlePair<OffEle>::mass);
  diEleMassOnlyOneHighME_ = new DiEleMon(monElemFuncs.getIB(), filterName_+"_diEle_onlyOnePassFilter_massHigh",
					 filterName_+"_diEle_onlyOnePassFilter Mass;M_{ee} (GeV/c^{2})",
					 bins.massHigh.nr,bins.massHigh.min,bins.massHigh.max,&ParticlePair<OffEle>::mass);
 } 
}

EleHLTFilterMon::~EleHLTFilterMon()
{
  for(auto & eleMonElem : eleMonElems_) delete eleMonElem;
  for(auto & eleFailMonElem : eleFailMonElems_) delete eleFailMonElem;
  for(auto & eleEffHist : eleEffHists_) delete eleEffHist;
  if(!doHEP_)
  { 
  delete diEleMassBothME_;
  delete diEleMassOnlyOneME_;  
  delete diEleMassBothHighME_;
  delete diEleMassOnlyOneHighME_;
  }
}

void EleHLTFilterMon::fill(const OffEvt& evt,float weight)
{ 
  for(size_t eleNr=0;eleNr<evt.eles().size();eleNr++){
    const OffEle& ele = evt.eles()[eleNr];
    if((ele.trigBits()&filterBit_)!=0){ //ele passes
      for(auto & eleMonElem : eleMonElems_) eleMonElem->fill(ele,evt,weight);
      for(auto & eleEffHist : eleEffHists_) eleEffHist->fill(ele,evt,weight);
    }else { //ele didnt pass trigger
      for(auto & eleFailMonElem : eleFailMonElems_) eleFailMonElem->fill(ele,evt,weight);
    }
  }//end loop over electrons

  if((evt.evtTrigBits()&filterBit_)!=0){
    for(size_t ele1Nr=0;ele1Nr<evt.eles().size();ele1Nr++){
      for(size_t ele2Nr=ele1Nr+1;ele2Nr<evt.eles().size();ele2Nr++){
	const OffEle& ele1 = evt.eles()[ele1Nr];
	const OffEle& ele2 = evt.eles()[ele2Nr];
       if(!doHEP_)
       {
	if((ele1.trigBits()&ele2.trigBits()&filterBit_)==filterBit_) {
	  diEleMassBothME_->fill(ParticlePair<OffEle>(ele1,ele2),weight);
	  diEleMassBothHighME_->fill(ParticlePair<OffEle>(ele1,ele2),weight);
	}else if((ele1.trigBits()&filterBit_)==filterBit_ || 
		(ele2.trigBits()&filterBit_)==filterBit_){
	  diEleMassOnlyOneME_->fill(ParticlePair<OffEle>(ele1,ele2),weight);
	  diEleMassOnlyOneHighME_->fill(ParticlePair<OffEle>(ele1,ele2),weight);
	}
	
       }
      }//end inner ele loop
    }//end outer ele loop
  }//end check if filter is present
}




