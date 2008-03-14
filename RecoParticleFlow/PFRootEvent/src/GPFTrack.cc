#include "RecoParticleFlow/PFRootEvent/interface/GPFTrack.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//________________________________________________________________________
GPFTrack::GPFTrack(DisplayManager * display,int view,int ident,
                   reco::PFRecTrack *tra, int size, double *x, double *y,
                   double pt,int linestyle, int linecolor, std::string option)
  : GPFBase(display,view,ident, linecolor),
    TGraph(size,x,y), track_(tra),pt_(pt), option_(option)
{
  ResetBit(kCanDelete);
  
  int    markerstyle = 8;
  double markersize = 0.8;

  SetLineColor(color_);
  SetLineStyle(linestyle);
  SetMarkerStyle(markerstyle);
  SetMarkerSize(markersize);
  SetMarkerColor(color_);
  
}                    
//____________________________________________________________________________________________________________
void GPFTrack::Print()
{
  std::cout<<*track_<<std::endl;
}
//_______________________________________________________________________________    
void GPFTrack::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
  // Execute action corresponding to a left mouse button click 
  //on a GPFTrack object in the window

  gPad->SetCursor(kHand);
  switch (event) {
  case kButton1Down:
    Print();
    display_->findBlock(origId_);
    display_->findAndDraw(origId_);
    break;
  default:break;
  }    
     
}
//______________________________________________________________________________
void GPFTrack::draw()
{
  TGraph::Draw(option_.data());
}
//______________________________________________________________________________
void GPFTrack::setColor(int color)
{
  if (option_=="f") SetFillColor(color);
  else              SetLineColor(color);
  SetMarkerColor(color);
}
//_____________________________________________________________________________
void GPFTrack::setInitialColor()
{
  if (option_=="f") SetFillColor(color_);
  else              SetLineColor(color_);
  SetMarkerColor(color_);
}
