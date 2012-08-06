#include "RecoParticleFlow/PFRootEvent/interface/GPFSimParticleUpgrade.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManagerUpgrade.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>



//________________________________________________________________________
GPFSimParticleUpgrade::GPFSimParticleUpgrade(DisplayManagerUpgrade *display,int view, int ident, 
                               const reco::PFSimParticle *ptc, 
                               int size, double *x, double *y,
                               double pt,TAttMarker *attm, TAttLine *attl,
                               std::string option)
  : GPFBaseUpgrade(display,view,ident,attm,attl),
    TGraph(size,x,y), part_(ptc), pt_(pt), option_(option) 
{
    
  ResetBit(kCanDelete);
  SetLineColor(lineAttr_->GetLineColor());
  SetLineStyle(lineAttr_->GetLineStyle());
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
  SetMarkerSize(markerAttr_->GetMarkerSize());
  SetMarkerColor(markerAttr_->GetMarkerColor());
}                    
//____________________________________________________________________________________________________________
void GPFSimParticleUpgrade::Print()
{
  std::cout<<*part_<<std::endl;
}
//_______________________________________________________________________________    
void GPFSimParticleUpgrade::ExecuteEvent(Int_t event, Int_t px, Int_t py)
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
void GPFSimParticleUpgrade::draw()
{
  TGraph::Draw(option_.data());
}
//_______________________________________________________________________________
void GPFSimParticleUpgrade::setColor()
{
  if (option_=="f") SetFillColor(lineAttr_->GetLineColor());
  else              SetLineColor(lineAttr_->GetLineColor());
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//_______________________________________________________________________________
void GPFSimParticleUpgrade::setColor(int newcol)
{
  if (option_=="f") SetFillColor(newcol);
  else              SetLineColor(newcol);
  SetMarkerColor(newcol);
}
//_____________________________________________________________________________
void GPFSimParticleUpgrade::setInitialColor()
{
  if (option_=="f") SetFillColor(lineAttr_->GetLineColor());
  else              SetLineColor(lineAttr_->GetLineColor());
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//_____________________________________________________________________________
void GPFSimParticleUpgrade::setNewSize()
{
  SetMarkerSize(markerAttr_->GetMarkerSize());
}
//____________________________________________________________________________
void GPFSimParticleUpgrade::setNewStyle()
{
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
}

