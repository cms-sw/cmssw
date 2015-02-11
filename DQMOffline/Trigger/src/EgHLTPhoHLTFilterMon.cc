#include "DQMOffline/Trigger/interface/EgHLTPhoHLTFilterMon.h"

#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemFuncs.h"
#include "DQMOffline/Trigger/interface/EgHLTBinData.h"
#include "DQMOffline/Trigger/interface/EgHLTCutMasks.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace egHLT ;

PhoHLTFilterMon::PhoHLTFilterMon(MonElemFuncs& monElemFuncs, const std::string& filterName,TrigCodes::TrigBitSet filterBit,const BinData& bins,const CutMasks& masks):
  filterName_(filterName),
  filterBit_(filterBit)
{
  bool monHLTFailedPho=false;
  bool doN1andSingleEffsPho=false;

  phoMonElems_.push_back(new MonElemContainer<OffPho>());
  //phoMonElems_.push_back(new MonElemContainer<OffPho>("_cut"," cut, debug hists ",new EgHLTDQMVarCut<OffPho>(~0x0,&OffPho::cutCode)));
  for(size_t i=0;i<phoMonElems_.size();i++){
    monElemFuncs.initStdPhoHists(phoMonElems_[i]->monElems(),filterName_,filterName_+"_pho_passFilter"+phoMonElems_[i]->name(),bins);
  }
  
  if(monHLTFailedPho) phoFailMonElems_.push_back(new MonElemContainer<OffPho>());
  for(size_t i=0;i<phoFailMonElems_.size();i++){
    monElemFuncs.initStdPhoHists(phoFailMonElems_[i]->monElems(),filterName_,filterName_+"_pho_failFilter"+phoMonElems_[i]->name(),bins);
  }
  
  phoEffHists_.push_back(new MonElemContainer<OffPho>()); 
  // phoEffHists_.push_back(new MonElemContainer<OffPho>("_jetTag"," Tag and Probe ",new EgJetB2BCut<OffPho>(-M_PI/12,M_PI/12,0.3)));
  if(doN1andSingleEffsPho){
    for(size_t i=0;i<phoEffHists_.size();i++){ 
      monElemFuncs.initStdEffHists(phoEffHists_[i]->cutMonElems(),filterName_,
				    filterName_+"_pho_effVsEt"+phoEffHists_[i]->name(),bins.et,&OffPho::et,masks);
      monElemFuncs.initStdEffHists(phoEffHists_[i]->cutMonElems(),filterName_,
				    filterName_+"_pho_effVsEta"+phoEffHists_[i]->name(),bins.eta,&OffPho::eta,masks); 
      /* monElemFuncs.initStdEffHists(phoEffHists_[i]->cutMonElems(),filterName_,
	 filterName_+"_pho_effVsPhi"+phoEffHists_[i]->name(),bins.phi,&OffPho::phi,masks);*/
    }
  }
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
  
PhoHLTFilterMon::~PhoHLTFilterMon()
{
  for(size_t i=0;i<phoMonElems_.size();i++) delete phoMonElems_[i];
  for(size_t i=0;i<phoFailMonElems_.size();i++) delete phoFailMonElems_[i];
  for(size_t i=0;i<phoEffHists_.size();i++) delete phoEffHists_[i];
  delete diPhoMassBothME_;
  delete diPhoMassOnlyOneME_; 
  delete diPhoMassBothHighME_;
  delete diPhoMassOnlyOneHighME_;
}


void PhoHLTFilterMon::fill(const OffEvt& evt,float weight)
{ 
  for(size_t phoNr=0;phoNr<evt.phos().size();phoNr++){
    const OffPho& pho = evt.phos()[phoNr];
    if((pho.trigBits()&filterBit_)!=0){ //pho passes
      for(size_t monElemNr=0;monElemNr<phoMonElems_.size();monElemNr++) phoMonElems_[monElemNr]->fill(pho,evt,weight);
      for(size_t monElemNr=0;monElemNr<phoEffHists_.size();monElemNr++) phoEffHists_[monElemNr]->fill(pho,evt,weight);
    }else { //pho didnt pass trigger
      for(size_t monElemNr=0;monElemNr<phoFailMonElems_.size();monElemNr++) phoFailMonElems_[monElemNr]->fill(pho,evt,weight);
    }
  }//end loop over photons



  if((evt.evtTrigBits()&filterBit_)!=0){
    for(size_t pho1Nr=0;pho1Nr<evt.phos().size();pho1Nr++){
      for(size_t pho2Nr=pho1Nr+1;pho2Nr<evt.phos().size();pho2Nr++){
	const OffPho& pho1 = evt.phos()[pho1Nr];
	const OffPho& pho2 = evt.phos()[pho2Nr];

	if((pho1.trigBits()&pho2.trigBits()&filterBit_)==filterBit_) diPhoMassBothME_->fill(ParticlePair<OffPho>(pho1,pho2),weight);
	else if((pho1.trigBits()&filterBit_)==filterBit_ || 
		(pho2.trigBits()&filterBit_)==filterBit_){
	  diPhoMassOnlyOneME_->fill(ParticlePair<OffPho>(pho1,pho2),weight);
	}
	
      }//end inner pho loop
    }//end outer pho loop
  }//end check if filter is present
}
