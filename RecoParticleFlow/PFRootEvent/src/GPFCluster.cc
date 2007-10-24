#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFCluster.h"
#include "TObject.h"
#include "TPad.h"
#include "TMarker.h"


//_________________________________________________________________
GPFCluster::GPFCluster() : GPFBase(0, 0, 0),clus_(0) {}
//_________________________________________________________________
GPFCluster::GPFCluster(DisplayManager * display,int view,int ident,const reco::PFCluster* clus,double x,double y,int color)
		       : GPFBase(display,view,ident),
		         TMarker(x,y,20),clus_(clus),color_(color)
{
 ResetBit(kCanDelete);
 en_=clus_->energy();
 SetMarkerColor(color_);
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
void GPFCluster::setColor(int color)
{
 SetMarkerColor(color);
}
//_________________________________________________________________
void GPFCluster::setInitialColor()
{
 SetMarkerColor(color_);
}
