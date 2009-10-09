#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "TObject.h"
#include "TPad.h"
#include "TMarker.h"


//_________________________________________________________________
GPFCluster::GPFCluster(DisplayManager * display,
                       int view,int ident,
                       const reco::PFCluster* clus,
                       double x,double y,TAttMarker *attm)
  : GPFBase(display,view,ident, attm ),
    TMarker(x,y,1),clus_(clus) {

  ResetBit(kCanDelete);
  en_=clus_->energy();
  SetMarkerColor(markerAttr_->GetMarkerColor());
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
  SetMarkerSize(markerAttr_->GetMarkerSize());
}   
                    
//_________________________________________________________________
void GPFCluster::Print()
{
  std::cout<<*clus_<<std::endl;
}
//_________________________________________________________________    
void GPFCluster::ExecuteEvent(Int_t event, Int_t px, Int_t py)
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
void GPFCluster::draw()
{
  TMarker::Draw();
}
//_________________________________________________________________
void GPFCluster::setColor()
{
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//_________________________________________________________________
void GPFCluster::setColor(int newcol)
{
  SetMarkerColor(newcol);
}
//_________________________________________________________________
void GPFCluster::setInitialColor()
{
  SetMarkerColor(markerAttr_->GetMarkerColor());
}
//_________________________________________________________________
void GPFCluster::setNewStyle()
{
  SetMarkerStyle(markerAttr_->GetMarkerStyle());
}
//_________________________________________________________________
void GPFCluster::setNewSize()
{
  SetMarkerSize(markerAttr_->GetMarkerSize());
}
