#include "RecoParticleFlow/PFRootEvent/interface/GPFTrackUpgrade.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayCommon.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManagerUpgrade.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//________________________________________________________________________
GPFTrackUpgrade::GPFTrackUpgrade(DisplayManagerUpgrade * display,int view,int ident,
                   reco::PFRecTrack *tra, int size, double *x, double *y,
                   double pt,TAttMarker *attm,TAttLine * attl, std::string option)
  : GPFBaseUpgrade(display,view,ident,attm,attl ),
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
void GPFTrackUpgrade::Print()
{
   if ((origId_>>SHIFTID) != BREMID) 
     std::cout<<*track_<<std::endl;
}
//_______________________________________________________________________________    
void GPFTrackUpgrade::ExecuteEvent(Int_t event, Int_t px, Int_t py)
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
void GPFTrackUpgrade::draw()
{
  TGraph::Draw(option_.data());
}
//______________________________________________________________________________
void GPFTrackUpgrade::setColor()
{
  if (option_=="f") SetFillColor(lineAttr_->GetLineColor());
  else              SetLineColor(lineAttr_->GetLineColor());
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//______________________________________________________________________________
void GPFTrackUpgrade::setColor(int color)
{
  if (option_=="f") SetFillColor(color);
  else              SetLineColor(color);
  SetMarkerColor(color);
}

//_____________________________________________________________________________
void GPFTrackUpgrade::setInitialColor()
  //redondant avec setColor ???? to check
{
  if (option_=="f") SetFillColor(lineAttr_->GetLineColor());
  else              SetLineColor(lineAttr_->GetLineColor());
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//_____________________________________________________________________________
void GPFTrackUpgrade::setNewSize()
{
  SetMarkerSize(markerAttr_->GetMarkerSize());
}
//____________________________________________________________________________
void GPFTrackUpgrade::setNewStyle()
{
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
}
 
