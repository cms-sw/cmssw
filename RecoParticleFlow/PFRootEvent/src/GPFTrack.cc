#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFTrack.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//_______________________________________________________________________
GPFTrack::GPFTrack() : track_(new reco::PFRecTrack())  {}
//________________________________________________________________________
GPFTrack::GPFTrack(reco::PFRecTrack *tra, int size, double *x, double *y,
                     int linestyle, int markerstyle, double markersize, int color,
                     std::string option)
		     : TGraph(size,x,y), track_(tra), option_(option) 
{
    
  ResetBit(kCanDelete);
  
  SetLineColor(color);
  SetLineStyle(linestyle);
  SetMarkerStyle(markerstyle);
  SetMarkerSize(markersize);
  SetMarkerColor(color);
  
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
     break;
   default:break;
 }    
     
}
//______________________________________________________________________________
void GPFTrack::Draw()
{
 TGraph::Draw(option_.data());
}
