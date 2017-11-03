#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"

using namespace egHLT;

bool EgTrigTagProbeCut::pass(const OffEle& theEle,const OffEvt& evt)const
{
  //first we check if our electron passes our id
  if( ((theEle.*cutCodeFunc_)() & cutCode_)!=0x0) return false;

  //new we check that there is another tag in the event (this electron may be a tag, we are not going to test this, all we care about is that another electron in the event is a tag)
  int nrTags=0;
  const OffEle* tagEle=nullptr;
  const std::vector<OffEle>& eles = evt.eles();
  //we are looking for an *additional* tag
  for(auto const & ele : eles){
    if( ((ele.*cutCodeFunc_)() & cutCode_)==0x0 && (bitsToPass_&ele.trigBits())==bitsToPass_){
      //now a check that the tag is not the same as the probe
      if(reco::deltaR2(theEle.eta(),theEle.phi(),ele.eta(),ele.phi())>0.1*0.1){//not in a cone of 0.1 of probe electron
	nrTags++;
	tagEle = &ele;
      }
    }
  }
  if(nrTags==1){ //we are requiring one and only one additional tag (the theEle is automatically excluded from the tag list) 
    float mass = (theEle.p4()+tagEle->p4()).mag();
    if(mass>minMass_ && mass<maxMass_) return true; //mass requirements
  }
  return false; 
}

bool EgTrigTagProbeCut_New::pass(const OffEle& theEle,const OffEvt& evt)const
{
  //looking at only Et>20, since probe is required to pass SC17
  //if(theEle.et()>20/* && theEle.et()<20*/){
  //first we check if our probe electron passes WP80 and the second leg of our T&P trigger
  if( ((theEle.*cutCodeFunc_)() & cutCode_)!=0x0 || (bit2ToPass_&theEle.trigBits())!=bit2ToPass_) return false;

  //now we check that there is a WP80 tag electron that passes the first leg of the trigger(this electron may be a tag, we are not going to test this, all we care about is that another electron in the event is a tag)
  int nrTags=0;
  const OffEle* tagEle=nullptr;
  const std::vector<OffEle>& eles = evt.eles();
  //we are looking for an *additional* tag
  for(auto const & ele : eles){
    if( ((ele.*cutCodeFunc_)() & cutCode_)==0x0 && (bit1ToPass_&ele.trigBits())==bit1ToPass_){
      //now a check that the tag is not the same as the probe
      if(reco::deltaR2(theEle.eta(),theEle.phi(),ele.eta(),ele.phi())>0.1*0.1){//not in a cone of 0.1 of probe electron
	nrTags++;
	tagEle = &ele;
      }
    }
  }
  if(nrTags==1){ //we are requiring one and only one additional tag (the theEle is automatically excluded from the tag list) 
    float mass = (theEle.p4()+tagEle->p4()).mag();
    if(mass>minMass_ && mass<maxMass_) return true; //mass requirements
  }
  //}//if 10<pt<20
  return false; 
}
//same for photons
bool EgTrigTagProbeCut_NewPho::pass(const OffPho& thePho,const OffEvt& evt)const
{
  //looking at only Et>20, since probe is required to pass SC17
  //if(theEle.et()>20/* && theEle.et()<20*/){
  //first we check if our probe electron passes WP80 and the second leg of our T&P trigger
  if( ((thePho.*cutCodeFunc_)() & cutCode_)!=0x0 || (bit2ToPass_&thePho.trigBits())!=bit2ToPass_) return false;

  //now we check that there is a WP80 tag electron that passes the first leg of the trigger(this electron may be a tag, we are not going to test this, all we care about is that another electron in the event is a tag)
  int nrTags=0;
  const OffPho* tagPho=nullptr;
  const std::vector<OffPho>& phos = evt.phos();
  //we are looking for an *additional* tag
  for(auto const & pho : phos){
    if( ((pho.*cutCodeFunc_)() & cutCode_)==0x0 && (bit1ToPass_&pho.trigBits())==bit1ToPass_){
      //now a check that the tag is not the same as the probe
      if(reco::deltaR2(thePho.eta(),thePho.phi(),pho.eta(),pho.phi())>0.1*0.1){//not in a cone of 0.1 of probe "photon"
	nrTags++;
	tagPho = &pho;
      }
    }
  }
  if(nrTags==1){ //we are requiring one and only one additional tag (the thePho is automatically excluded from the tag list) 
    float mass = (thePho.p4()+tagPho->p4()).mag();
    if(mass>minMass_ && mass<maxMass_) return true; //mass requirements
  }
  //}//if 10<pt<20
  return false; 
}

bool EgDiEleCut::pass(const OffEle& obj,const OffEvt& evt)const
{
  const std::vector<OffEle>& eles = evt.eles();
  for(auto const & ele : eles){
    if(&ele!=&obj){ //different electrons
     
      int diEleCutCode = (obj.*cutCodeFunc_)() | (ele.*cutCodeFunc_)();
      if( (diEleCutCode & cutCode_)==0x0) return true;
    }
  }
  return false;
}



bool EgDiPhoCut::pass(const OffPho& obj,const OffEvt& evt)const
{
  const std::vector<OffPho>& phos = evt.phos();
  for(auto const & pho : phos){
    if(&pho!=&obj){ //different phos
     
      int diPhoCutCode = (obj.*cutCodeFunc_)() | (pho.*cutCodeFunc_)();
      if( (diPhoCutCode & cutCode_)==0x0) return true;
    }
  }
  return false;
}
