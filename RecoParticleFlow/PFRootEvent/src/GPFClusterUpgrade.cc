#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFClusterUpgrade.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManagerUpgrade.h"
#include "TObject.h"
#include "TPad.h"
#include "TMarker.h"


//_________________________________________________________________
GPFClusterUpgrade::GPFClusterUpgrade(DisplayManagerUpgrade * display,
                       int view,int ident,
                       const reco::PFCluster* clus,
                       double x,double y,TAttMarker *attm)
  : GPFBaseUpgrade(display,view,ident, attm ),
    TMarker(x,y,1),clus_(clus) {

  ResetBit(kCanDelete);
  en_=clus_->energy();
  SetMarkerColor(markerAttr_->GetMarkerColor());
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
  SetMarkerSize(markerAttr_->GetMarkerSize());
}   
                    
//_________________________________________________________________
void GPFClusterUpgrade::Print()
{
  std::cout<<*clus_<<std::endl;
}
//_________________________________________________________________    
void GPFClusterUpgrade::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
  // Execute action corresponding to a mouse click 
  //on a GPFRecHit object in the window

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
//_________________________________________________________________
void GPFClusterUpgrade::draw()
{
  TMarker::Draw();
}
//_________________________________________________________________
void GPFClusterUpgrade::setColor()
{
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//_________________________________________________________________
void GPFClusterUpgrade::setColor(int newcol)
{
  SetMarkerColor(newcol);
}
//_________________________________________________________________
void GPFClusterUpgrade::setInitialColor()
{
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//_________________________________________________________________
void GPFClusterUpgrade::setNewStyle()
{
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
}
//_________________________________________________________________
void GPFClusterUpgrade::setNewSize()
{
  SetMarkerSize(markerAttr_->GetMarkerSize());
}
