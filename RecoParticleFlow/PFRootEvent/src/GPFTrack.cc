#include "RecoParticleFlow/PFRootEvent/interface/GPFTrack.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayCommon.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//________________________________________________________________________
GPFTrack::GPFTrack(DisplayManager * display,int view,int ident,
                   reco::PFRecTrack *tra, int size, double *x, double *y,
                   double pt,TAttMarker *attm,TAttLine * attl, std::string option)
  : GPFBase(display,view,ident,attm,attl ),
    TGraph(size,x,y), track_(tra),pt_(pt), option_(option)
{
  ResetBit(kCanDelete);
  
  SetLineColor(lineAttr_->GetLineColor());
  SetLineStyle(lineAttr_->GetLineStyle());
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
  SetMarkerSize(markerAttr_->GetMarkerSize());
  SetMarkerColor(markerAttr_->GetMarkerColor());
  
}
//_________________________________________________________________________                    
void GPFTrack::Print()
{
   if ((origId_>>SHIFTID) != BREMID) 
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
void GPFTrack::setColor()
{
  if (option_=="f") SetFillColor(lineAttr_->GetLineColor());
  else              SetLineColor(lineAttr_->GetLineColor());
  SetMarkerColor(markerAttr_->GetMarkerColor());
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
  //redondant avec setColor ???? to check
{
  if (option_=="f") SetFillColor(lineAttr_->GetLineColor());
  else              SetLineColor(lineAttr_->GetLineColor());
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//_____________________________________________________________________________
void GPFTrack::setNewSize()
{
  SetMarkerSize(markerAttr_->GetMarkerSize());
}
//____________________________________________________________________________
void GPFTrack::setNewStyle()
{
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
}
 
