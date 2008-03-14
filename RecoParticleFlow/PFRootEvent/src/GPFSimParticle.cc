#include "RecoParticleFlow/PFRootEvent/interface/GPFSimParticle.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>



//________________________________________________________________________
GPFSimParticle::GPFSimParticle(DisplayManager *display,int view, int ident, 
			       const reco::PFSimParticle *ptc, 
			       int size, double *x, double *y,
                               double pt,int markerstyle, int color,
			       std::string option)
  : GPFBase(display,view,ident, color),
    TGraph(size,x,y), part_(ptc), pt_(pt), option_(option) 
{
    
  ResetBit(kCanDelete);
  
  int    linestyle = 2;
  double markersize = 0.8;
  
  SetLineColor(color_);
  SetLineStyle(linestyle);
  SetMarkerStyle(markerstyle);
  SetMarkerSize(markersize);
  SetMarkerColor(color_);
  
}                    
//____________________________________________________________________________________________________________
void GPFSimParticle::Print()
{
  std::cout<<*part_<<std::endl;
}
//_______________________________________________________________________________    
void GPFSimParticle::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
  // Execute action corresponding to a left mouse button click 
  //on a GPFSimParticle object in the window

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
void GPFSimParticle::draw()
{
  TGraph::Draw(option_.data());
}
//_______________________________________________________________________________
void GPFSimParticle::setColor(int color)
{
  if (option_=="f") SetFillColor(color);
  else              SetLineColor(color);
  SetMarkerColor(color);
}
//_____________________________________________________________________________
void GPFSimParticle::setInitialColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
  SetMarkerColor(color_);
}


