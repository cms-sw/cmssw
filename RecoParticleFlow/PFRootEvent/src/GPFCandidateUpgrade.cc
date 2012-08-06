#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFCandidateUpgrade.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManagerUpgrade.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//________________________________________________________________________
GPFCandidateUpgrade::GPFCandidateUpgrade(DisplayManagerUpgrade * display,int view,int ident,
                     reco::PFCandidate *candidate,int size,
                     double *x, double *y, int color, std::string option)
  : GPFBaseUpgrade(display,view,ident, color),
    TGraph(size,x,y),
    candidate_(candidate), option_(option)
{
  ResetBit(kCanDelete);
  
  en_=candidate_->energy();  
    
  SetLineColor(color_);
  SetFillColor(color_);
  
}                    
//__________________________________________________________________________
void GPFCandidateUpgrade::Print()
{
  std::cout<<*candidate_<<std::endl;
}
//___________________________________________________________________________
void GPFCandidateUpgrade::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
  // Execute action corresponding to a mouse click 
  //on a GPFCandidate object in the window

  gPad->SetCursor(kHand);
  switch (event) {
  case kButton1Down:
    Print();
    display_->findAndDraw(origId_);
    break;
  default:break;
  }    
}
//__________________________________________________________________________
void GPFCandidateUpgrade::draw()
{
  TGraph::Draw(option_.data());
}
//__________________________________________________________________________
void GPFCandidateUpgrade::setColor(int color)
{
  if (option_=="f") SetFillColor(color);
  else              SetLineColor(color);
}
//_________________________________________________________________________
void GPFCandidateUpgrade::setInitialColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
}
//_________________________________________________________________________
void GPFCandidateUpgrade::setColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
}

//_________________________________________________________________________
void GPFCandidateUpgrade::setNewSize() 
{
  //not implemented
}
//_________________________________________________________________________
void GPFCandidateUpgrade::setNewStyle()
{
  //not implemented
}
