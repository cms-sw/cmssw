#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"

using namespace egHLT;

bool EgTrigTagProbeCut::pass(const OffEle& theEle,const OffEvt& evt)const
{
  //first we check if our electron passes our id
  if( ((theEle.*cutCodeFunc_)() & cutCode_)!=0x0) return false;

  //new we check that there is another tag in the event (this electron may be a tag, we are not going to test this, all we care about is that another electron in the event is a tag)
  int nrTags=0;
  const OffEle* tagEle=NULL;
  const std::vector<OffEle>& eles = evt.eles();
  //we are looking for an *additional* tag
  for(size_t eleNr=0;eleNr<eles.size();eleNr++){
    if( ((eles[eleNr].*cutCodeFunc_)() & cutCode_)==0x0 && (bitsToPass_&eles[eleNr].trigBits())==bitsToPass_){
      //now a check that the tag is not the same as the probe
      if(reco::deltaR2(theEle.eta(),theEle.phi(),eles[eleNr].eta(),eles[eleNr].phi())>0.1*0.1){//not in a cone of 0.1 of probe electron
	nrTags++;
	tagEle = &eles[eleNr];
      }
    }
  }
  if(nrTags==1){ //we are requiring one and only one additional tag (the theEle is automatically excluded from the tag list) 
    float mass = (theEle.p4()+tagEle->p4()).mag();
    if(mass>minMass_ && mass<maxMass_) return true; //mass requirements
  }
  return false; 
}

bool EgDiEleCut::pass(const OffEle& obj,const OffEvt& evt)const
{
  const std::vector<OffEle>& eles = evt.eles();
  for(size_t eleNr=0;eleNr<eles.size();eleNr++){
    if(&eles[eleNr]!=&obj){ //different electrons
     
      int diEleCutCode = (obj.*cutCodeFunc_)() | (eles[eleNr].*cutCodeFunc_)();
      if( (diEleCutCode & cutCode_)==0x0) return true;
    }
  }
  return false;
}



bool EgDiPhoCut::pass(const OffPho& obj,const OffEvt& evt)const
{
  const std::vector<OffPho>& phos = evt.phos();
  for(size_t phoNr=0;phoNr<phos.size();phoNr++){
    if(&phos[phoNr]!=&obj){ //different phos
     
      int diPhoCutCode = (obj.*cutCodeFunc_)() | (phos[phoNr].*cutCodeFunc_)();
      if( (diPhoCutCode & cutCode_)==0x0) return true;
    }
  }
  return false;
}
