#include <utility>

#include "DQMOffline/Trigger/interface/EgHLTPhoHLTFilterMon.h"

#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemFuncs.h"
#include "DQMOffline/Trigger/interface/EgHLTBinData.h"
#include "DQMOffline/Trigger/interface/EgHLTCutMasks.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace egHLT ;

PhoHLTFilterMon::PhoHLTFilterMon(MonElemFuncs& monElemFuncs, std::string  filterName,TrigCodes::TrigBitSet filterBit,const BinData& bins,const CutMasks& masks, bool doHEP):
  filterName_(std::move(filterName)),
  filterBit_(filterBit),
  doHEP_(doHEP)
{
  bool monHLTFailedPho=false;
  bool doN1andSingleEffsPho=false;
  std::string histname="egamma";
 
  phoMonElems_.push_back(new MonElemContainer<OffPho>());
  //phoMonElems_.push_back(new MonElemContainer<OffPho>("_cut"," cut, debug hists ",new EgHLTDQMVarCut<OffPho>(~0x0,&OffPho::cutCode)));
  for(auto & phoMonElem : phoMonElems_){
   if(doHEP_){
               monElemFuncs.initStdPhoHistsHEP(phoMonElem->monElems(),filterName_,histname+"_passFilter"+phoMonElem->name(),bins);
              }else{ 
                     monElemFuncs.initStdPhoHists(phoMonElem->monElems(),filterName_,filterName_+"_pho_passFilter"+phoMonElem->name(),bins);
                    }
  }

  if(monHLTFailedPho) phoFailMonElems_.push_back(new MonElemContainer<OffPho>());

  for(size_t i=0;i<phoFailMonElems_.size();i++){
   if(doHEP_){
               monElemFuncs.initStdPhoHistsHEP(phoFailMonElems_[i]->monElems(),filterName_,histname+"_failFilter"+phoMonElems_[i]->name(),bins);
             }else{ 
                   monElemFuncs.initStdPhoHists(phoFailMonElems_[i]->monElems(),filterName_,filterName_+"_pho_failFilter"+phoMonElems_[i]->name(),bins);
                  }
  } 
  phoEffHists_.push_back(new MonElemContainer<OffPho>()); 
  // phoEffHists_.push_back(new MonElemContainer<OffPho>("_jetTag"," Tag and Probe ",new EgJetB2BCut<OffPho>(-M_PI/12,M_PI/12,0.3)));
  if(doN1andSingleEffsPho){
    for(auto & phoEffHist : phoEffHists_){ 
      monElemFuncs.initStdEffHists(phoEffHist->cutMonElems(),filterName_,
				    filterName_+"_pho_effVsEt"+phoEffHist->name(),bins.et,&OffPho::et,masks);
      monElemFuncs.initStdEffHists(phoEffHist->cutMonElems(),filterName_,
				    filterName_+"_pho_effVsEta"+phoEffHist->name(),bins.eta,&OffPho::eta,masks); 
      /* monElemFuncs.initStdEffHists(phoEffHists_[i]->cutMonElems(),filterName_,
	 filterName_+"_pho_effVsPhi"+phoEffHists_[i]->name(),bins.phi,&OffPho::phi,masks);*/
    }
  }
 if(!doHEP_)
 {
  typedef MonElemManager<ParticlePair<OffPho>,float >  DiPhoMon;
  diPhoMassBothME_ = new DiPhoMon(monElemFuncs.getIB(), filterName_+"_diPho_bothPassFilter_mass",
				  filterName_+"_diPho_bothPassFilter Mass;M_{#gamma#gamma} (GeV/c^{2})",
				  bins.mass.nr,bins.mass.min,bins.mass.max,&ParticlePair<OffPho>::mass);
  diPhoMassOnlyOneME_ = new DiPhoMon(monElemFuncs.getIB(), filterName_+"_diPho_onlyOnePassFilter_mass",
				     filterName_+"_diPho_onlyOnePassFilter Mass;M_{#gamma#gamma} (GeV/c^{2})",
				     bins.mass.nr,bins.mass.min,bins.mass.max,&ParticlePair<OffPho>::mass);
  diPhoMassBothHighME_ = new DiPhoMon(monElemFuncs.getIB(), filterName_+"_diPho_bothPassFilter_massHigh",
				  filterName_+"_diPho_bothPassFilter Mass;M_{#gamma#gamma} (GeV/c^{2})",
				  bins.massHigh.nr,bins.massHigh.min,bins.massHigh.max,&ParticlePair<OffPho>::mass);
  diPhoMassOnlyOneHighME_ = new DiPhoMon(monElemFuncs.getIB(), filterName_+"_diPho_onlyOnePassFilter_massHigh",
				     filterName_+"_diPho_onlyOnePassFilter Mass;M_{#gamma#gamma} (GeV/c^{2})",
				     bins.massHigh.nr,bins.massHigh.min,bins.massHigh.max,&ParticlePair<OffPho>::mass);
  
 }
}
  
PhoHLTFilterMon::~PhoHLTFilterMon()
{
  for(auto & phoMonElem : phoMonElems_) delete phoMonElem;
  for(auto & phoFailMonElem : phoFailMonElems_) delete phoFailMonElem;
  for(auto & phoEffHist : phoEffHists_) delete phoEffHist;
 if(!doHEP_)
 { 
  delete diPhoMassBothME_;
  delete diPhoMassOnlyOneME_; 
  delete diPhoMassBothHighME_;
  delete diPhoMassOnlyOneHighME_;
 }
}

void PhoHLTFilterMon::fill(const OffEvt& evt,float weight)
{ 
  for(size_t phoNr=0;phoNr<evt.phos().size();phoNr++){
    const OffPho& pho = evt.phos()[phoNr];
    if((pho.trigBits()&filterBit_)!=0){ //pho passes
      for(auto & phoMonElem : phoMonElems_) phoMonElem->fill(pho,evt,weight);
      for(auto & phoEffHist : phoEffHists_) phoEffHist->fill(pho,evt,weight);
    }else { //pho didnt pass trigger
      for(auto & phoFailMonElem : phoFailMonElems_) phoFailMonElem->fill(pho,evt,weight);
    }
  }//end loop over photons



  if((evt.evtTrigBits()&filterBit_)!=0){
    for(size_t pho1Nr=0;pho1Nr<evt.phos().size();pho1Nr++){
      for(size_t pho2Nr=pho1Nr+1;pho2Nr<evt.phos().size();pho2Nr++){
	const OffPho& pho1 = evt.phos()[pho1Nr];
	const OffPho& pho2 = evt.phos()[pho2Nr];
        if(!doHEP_)
       {    
	if((pho1.trigBits()&pho2.trigBits()&filterBit_)==filterBit_) diPhoMassBothME_->fill(ParticlePair<OffPho>(pho1,pho2),weight);
	else if((pho1.trigBits()&filterBit_)==filterBit_ || 
		(pho2.trigBits()&filterBit_)==filterBit_){
	  diPhoMassOnlyOneME_->fill(ParticlePair<OffPho>(pho1,pho2),weight);
	}
       }
      }//end inner pho loop
    }//end outer pho loop
  }//end check if filter is present
}
