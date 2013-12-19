#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFRecHit.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//________________________________________________________________________
GPFRecHit::GPFRecHit(DisplayManager * display,int view,int ident,
                     reco::PFRecHit *rechit,int size,
                     double *x, double *y, int color, std::string option)
  : GPFBase(display,view,ident, color),
    TGraph(size,x,y),
    recHit_(rechit), option_(option)
{
  ResetBit(kCanDelete);
  
  en_=recHit_->energy();  
    
  SetLineColor(color_);
  SetFillColor(color_);
  
}                    
//__________________________________________________________________________
void GPFRecHit::Print()
{
  std::cout<<*recHit_<<std::endl;
}
//___________________________________________________________________________
void GPFRecHit::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
  // Execute action corresponding to a mouse click 
  //on a GPFRecHit object in the window

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
void GPFRecHit::draw()
{
  TGraph::Draw(option_.data());
}
//__________________________________________________________________________
void GPFRecHit::setColor(int color)
{
  if (option_=="f") SetFillColor(color);
  else              SetLineColor(color);
}
//_________________________________________________________________________
void GPFRecHit::setInitialColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
}
//_________________________________________________________________________
void GPFRecHit::setColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
}

//_________________________________________________________________________
void GPFRecHit::setNewSize() 
{
  //not implemented
}
//_________________________________________________________________________
void GPFRecHit::setNewStyle()
{
  //not implemented
}
