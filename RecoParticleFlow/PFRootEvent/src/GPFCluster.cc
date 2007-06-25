#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFCluster.h"
#include "TObject.h"
#include "TPad.h"
#include "TMarker.h"


//_________________________________________________________________
GPFCluster::GPFCluster() : clus_(0), clusNr_(0) {}
//_________________________________________________________________
GPFCluster::GPFCluster(const reco::PFCluster* clus,double x,double y,
                       int color,unsigned clusNr)
		       :TMarker(x,y,20),clus_(clus),clusNr_(clusNr)
{
 SetMarkerColor(color);
}                       
//_________________________________________________________________
void GPFCluster::Print()
{
  std::cout<< "clusNr:"<<clusNr_<<std::endl;
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
     break;
   default:break;
 }    
}
