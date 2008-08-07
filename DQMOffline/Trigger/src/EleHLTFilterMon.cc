#include "DQMOffline/Trigger/interface/EleHLTFilterMon.h"
#include "DQMOffline/Trigger/interface/CutCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "PhysicsTools/Utilities/interface/deltaR.h"

EleHLTFilterMon::EleHLTFilterMon(std::string filterName):
  filterName_(filterName)
{
  initStdEleHists(eleMonElems_,filterName_+"_gsfEle_passFilter");
  initStdEleHists(eleFailMonElems_,filterName_+"_gsfEle_failFilter");
  initStdEffHists(eleEffHists_,filterName_+"_gsfEle_effVsEt",110,-10.,100.,&EgHLTOffEle::et);
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


void EleHLTFilterMon::initStdEleHists(std::vector<MonElemManagerBase<EgHLTOffEle>*>& histVec,std::string baseName)
{
  histVec.push_back(new MonElemManager<EgHLTOffEle,float>(baseName+"_et",baseName+" E_{T}:E_{T} (GeV)",110,-10.,100.,&EgHLTOffEle::et));
  histVec.push_back(new MonElemManager<EgHLTOffEle,float>(baseName+"_eta",baseName+" #eta:#eta",54,-2.7,2.7,&EgHLTOffEle::eta));		
  histVec.push_back(new MonElemManager<EgHLTOffEle,float>(baseName+"_phi",baseName+" #phi:#phi (rad)",50,-3.14,3.14,&EgHLTOffEle::phi));
  histVec.push_back(new MonElemManager<EgHLTOffEle,int>(baseName+"_charge",baseName+" Charge: charge",2,-1.5,1.5,&EgHLTOffEle::charge));
  
  histVec.push_back(new MonElemManager<EgHLTOffEle,float>(baseName+"_hOverE",baseName+" H/E: H/E",60,-0.05,0.25,&EgHLTOffEle::hOverE));
  histVec.push_back(new MonElemManager<EgHLTOffEle,float>(baseName+"_dPhiIn",baseName+" #Delta #phi_{in}: #Delta #phi_{in}",50,-0.15,0.15,&EgHLTOffEle::dPhiIn));
  histVec.push_back(new MonElemManager<EgHLTOffEle,float>(baseName+"_dEtaIn",baseName+" #Delta #eta_{in}: #Delta #eta_{in}",50,-0.02,0.02,&EgHLTOffEle::dEtaIn));
  histVec.push_back(new MonElemManager<EgHLTOffEle,float>(baseName+"_sigmaEtaEta",baseName+" #sigma_{#eta#eta} : #sigma_{#eta#eta}",60,-0.01,0.05,&EgHLTOffEle::sigmaEtaEta));

}

void EleHLTFilterMon::initStdEffHists(std::vector<EgammaHLTEffSrcBase<EgHLTOffEle>*>& histVec,std::string baseName,int nrBins,double xMin,double xMax,float (EgHLTOffEle::*vsVarFunc)()const)
{
  //some convience typedefs, I hate typedefs but atleast here where they are defined is obvious
  typedef EgammaHLTEffSrc<EgHLTOffEle,float> EffFloat;
  typedef EgammaHLTEffSrc<EgHLTOffEle,int> EffInt;
  typedef EgHLTDQMVarCut<EgHLTOffEle> VarCut;
  int stdCutCode = CutCodes::getCode("et:detEta:crack:sigmaEtaEta:hadem:dPhiIn:dEtaIn"); //will have it non hardcoded at a latter date
  histVec.push_back(new EffFloat(baseName+"_single_hOverE",baseName+" Single H/E",nrBins,xMin,xMax,vsVarFunc,
				 new VarCut(CutCodes::HADEM,&EgHLTOffEle::cutCode)));
  histVec.push_back(new EffFloat(baseName+"_n1_hOverE",baseName+" N1 H/E",nrBins,xMin,xMax,vsVarFunc,
				 new VarCut(stdCutCode,&EgHLTOffEle::cutCode),
				 new VarCut(~CutCodes::HADEM &stdCutCode,&EgHLTOffEle::cutCode)));
  histVec.push_back(new EffFloat(baseName+"_single_dEtaIn",baseName+" Single #Delta#eta_{in}",nrBins,xMin,xMax,vsVarFunc,
				 new VarCut(CutCodes::DETAIN,&EgHLTOffEle::cutCode)));
  histVec.push_back(new EffFloat(baseName+"_n1_dEtaIn",baseName+" N1 #Delta#eta_{in}",nrBins,xMin,xMax,vsVarFunc,
				 new VarCut(stdCutCode,&EgHLTOffEle::cutCode),
				 new VarCut(~CutCodes::DETAIN&stdCutCode,&EgHLTOffEle::cutCode)));
  histVec.push_back(new EffFloat(baseName+"_single_dPhiIn",baseName+" Single #Delta#phi_{in}",nrBins,xMin,xMax,vsVarFunc,
				 new VarCut(CutCodes::DPHIIN,&EgHLTOffEle::cutCode)));
  histVec.push_back(new EffFloat(baseName+"_n1_dPhiIn",baseName+" N1 #Delta#phi_{in}",nrBins,xMin,xMax,vsVarFunc,
				 new VarCut(stdCutCode,&EgHLTOffEle::cutCode),
				 new VarCut(~CutCodes::DPHIIN&stdCutCode,&EgHLTOffEle::cutCode)));
  histVec.push_back(new EffFloat(baseName+"_single_sigmaEtaEta",baseName+" Single #sigma_{#eta#eta}",nrBins,xMin,xMax,vsVarFunc,
				 new VarCut(CutCodes::SIGMAETAETA,&EgHLTOffEle::cutCode)));
  histVec.push_back(new EffFloat(baseName+"_n1_sigmaEtaEta",baseName+" N1 #sigma_{#eta#eta}",nrBins,xMin,xMax,vsVarFunc,
				 new VarCut(stdCutCode,&EgHLTOffEle::cutCode),
				 new VarCut(~CutCodes::SIGMAETAETA&stdCutCode,&EgHLTOffEle::cutCode)));
}

void EleHLTFilterMon::fill(const EgHLTOffData& evtData,float weight)
{
  //there are two cases: 1) no events passed filter so it doesnt exist or 2) alteast one electron passed
  //first case, filter not found, all electrons failed it, just fill the fail hists
  //second case, have to test each electron to see if it was the one that passed
   
  

  //edm::LogInfo("EleHLTFilterMon") << "begin filling ";

  size_t filterNrInEvt=evtData.trigEvt->filterIndex(edm::InputTag(filterName_,"","HLT").encode());

  //fill the histograms
  //first check if any electron passed the filter, if not, no need to check each one
  if(filterNrInEvt==evtData.trigEvt->sizeFilters()){ //filter not found
    //edm::LogInfo("EleHLTFilterMon") << "filter "<<filterName_<<" not found ";
    for(size_t eleNr=0;eleNr<evtData.eles->size();eleNr++){
      for(size_t monElemNr=0;monElemNr<eleFailMonElems_.size();monElemNr++){
	eleFailMonElems_[monElemNr]->fill((*evtData.eles)[eleNr],weight);
      }
    }//end electron loop
  }else{ //filter found 
    //  edm::LogInfo("EleHLTFilterMon") << "filter "<<filterName_<<" found ";
    for(size_t eleNr=0;eleNr<evtData.eles->size();eleNr++){
      std::vector<int>& filtersPassed = (*evtData.filtersElePasses)[eleNr];
      if(std::binary_search(filtersPassed.begin(),filtersPassed.end(),static_cast<int>(filterNrInEvt))) { //passed filter
	for(size_t monElemNr=0;monElemNr<eleMonElems_.size();monElemNr++){
	  eleMonElems_[monElemNr]->fill((*evtData.eles)[eleNr],weight);
	}	
	for(size_t effHistNr=0;effHistNr<eleEffHists_.size();effHistNr++){
	  eleEffHists_[effHistNr]->fill((*evtData.eles)[eleNr],evtData,weight);
	}
	
	
      }else{ //failed filter
	for(size_t monElemNr=0;monElemNr<eleFailMonElems_.size();monElemNr++){
	  eleFailMonElems_[monElemNr]->fill((*evtData.eles)[eleNr],weight);
	}
      }//end pass filter check
    }//end electron loop
    
    for(size_t monElemNr=0;monElemNr<diEleMonElems_.size();monElemNr++){
      for(size_t ele1Nr=0;ele1Nr<evtData.eles->size();ele1Nr++){
	for(size_t ele2Nr=ele1Nr+1;ele2Nr<evtData.eles->size();ele2Nr++){
	  const EgHLTOffEle& ele1 = (*evtData.eles)[ele1Nr];
	  const EgHLTOffEle& ele2 = (*evtData.eles)[ele2Nr];
	  //  edm::LogInfo("EleHLTFilterMon") << "filling "<<monElemNr<<" ele1 "<<ele1Nr<<" ele2 "<<ele2Nr<<" nr ele "<<evtData.eles->size();
	  diEleMonElems_[monElemNr]->fill(ParticlePair<EgHLTOffEle>(ele1,ele2),weight);
	}
      }
    }//end di ele mon elem loop
  }//end check if filter is present
  // edm::LogInfo("EleHLTFilterMon") <<"filter "<<filterName_<< "end fill ";
}




