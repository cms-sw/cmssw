#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFRecHit.h"
//#include "TPolyLine.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>

//_______________________________________________________________________
GPFRecHit::GPFRecHit() : recHit_(new reco::PFRecHit)
{}
//________________________________________________________________________
GPFRecHit::GPFRecHit(reco::PFRecHit *rechit,int size,
                     double *x, double *y, int color, std::string option)
		     : TGraph(size,x,y), recHit_(rechit), option_(option)
{
    
  ResetBit(kCanDelete);
    
  SetLineColor(color);
  SetFillColor(color);
  
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
     break;
   default:break;
 }    
     
}
//______________________________________________________________________________
void GPFRecHit::Draw()
{
 TGraph::Draw(option_.data());
}
