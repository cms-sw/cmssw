#include "RecoParticleFlow/PFRootEvent/interface/GPFPart.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//_______________________________________________________________________
GPFPart::GPFPart() : part_(new reco::PFSimParticle())
{}
//________________________________________________________________________
GPFPart::GPFPart(const reco::PFSimParticle *ptc, int size, double *x, double *y,
                 int linestyle, int markerstyle, double markersize, int color,
                 std::string option)
		: TGraph(size,x,y), part_(ptc), option_(option) 
{
    
  ResetBit(kCanDelete);
  
  SetLineColor(color);
  SetLineStyle(linestyle);
  SetMarkerStyle(markerstyle);
  SetMarkerSize(markersize);
  SetMarkerColor(color);
  
}		     
//____________________________________________________________________________________________________________
void GPFPart::Print()
{
  std::cout<<*part_<<std::endl;
}
//_______________________________________________________________________________    
void GPFPart::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
 // Execute action corresponding to a left mouse button click 
 //on a GPFPart object in the window

 gPad->SetCursor(kHand);
 switch (event) {
   case kButton1Down:
     Print();
     break;
   default:break;
 }    
     
}
//______________________________________________________________________________
void GPFPart::Draw()
{
 TGraph::Draw(option_.data());
}
