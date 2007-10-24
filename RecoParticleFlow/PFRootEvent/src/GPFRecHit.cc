#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFRecHit.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>

//_______________________________________________________________________
GPFRecHit::GPFRecHit() : GPFBase(0, 0, 0),recHit_(new reco::PFRecHit)
{}
//________________________________________________________________________
GPFRecHit::GPFRecHit(DisplayManager * display,int view,int ident,reco::PFRecHit *rechit,int size,
                     double *x, double *y, int color, std::string option)
		     : GPFBase(display,view,ident),
		       TGraph(size,x,y),
		       recHit_(rechit), option_(option), color_(color)
{
  ResetBit(kCanDelete);
  
  en_=recHit_->energy();  
    
  SetLineColor(color_);
  SetFillColor(color_);
  
}		     
//____________________________________________________________________________________________________________
void GPFRecHit::Print()
{
  std::cout<<*recHit_<<std::endl;
}
//_______________________________________________________________________________    
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
//______________________________________________________________________________
void GPFRecHit::draw()
{
 TGraph::Draw(option_.data());
}
//______________________________________________________________________________
void GPFRecHit::setColor(int color)
{
  if (option_=="f") SetFillColor(color);
  else              SetLineColor(color);
}
//_____________________________________________________________________________
void GPFRecHit::setInitialColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
}
   
