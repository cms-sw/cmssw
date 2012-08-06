#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFRecHitUpgrade.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManagerUpgrade.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//________________________________________________________________________
GPFRecHitUpgrade::GPFRecHitUpgrade(DisplayManagerUpgrade * display,int view,int ident,
                     reco::PFRecHit *rechit,int size,
                     double *x, double *y, int color, std::string option)
  : GPFBaseUpgrade(display,view,ident, color),
    TGraph(size,x,y),
    recHit_(rechit), option_(option)
{
  ResetBit(kCanDelete);
  
  en_=recHit_->energy();  
    
  SetLineColor(color_);
  SetFillColor(color_);
  
}                    
//__________________________________________________________________________
void GPFRecHitUpgrade::Print()
{
  std::cout<<*recHit_<<std::endl;
}
//___________________________________________________________________________
void GPFRecHitUpgrade::ExecuteEvent(Int_t event, Int_t px, Int_t py)
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
void GPFRecHitUpgrade::draw()
{
  TGraph::Draw(option_.data());
}
//__________________________________________________________________________
void GPFRecHitUpgrade::setColor(int color)
{
  if (option_=="f") SetFillColor(color);
  else              SetLineColor(color);
}
//_________________________________________________________________________
void GPFRecHitUpgrade::setInitialColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
}
//_________________________________________________________________________
void GPFRecHitUpgrade::setColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
}

//_________________________________________________________________________
void GPFRecHitUpgrade::setNewSize() 
{
  //not implemented
}
//_________________________________________________________________________
void GPFRecHitUpgrade::setNewStyle()
{
  //not implemented
}
