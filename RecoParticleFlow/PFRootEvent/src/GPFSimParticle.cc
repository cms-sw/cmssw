#include "RecoParticleFlow/PFRootEvent/interface/GPFSimParticle.h"
#include "TPad.h"
#include "TObject.h"
#include "TGraph.h"
#include <string>


//_______________________________________________________________________
GPFSimParticle::GPFSimParticle() : part_(new reco::PFSimParticle())
{}
//________________________________________________________________________
GPFSimParticle::GPFSimParticle(const reco::PFSimParticle *ptc, int size, double *x, double *y,
                 double pt,int markerstyle, std::string option)
		: TGraph(size,x,y), part_(ptc), pt_(pt), option_(option) 
{
    
  ResetBit(kCanDelete);
  
  int    color = 4;
  int    linestyle = 2;
  double markersize = 0.8;
  
  SetLineColor(color);
  SetLineStyle(linestyle);
  SetMarkerStyle(markerstyle);
  SetMarkerSize(markersize);
  SetMarkerColor(color);
  
}		     
//____________________________________________________________________________________________________________
void GPFSimParticle::Print()
{
  std::cout<<*part_<<std::endl;
}
//_______________________________________________________________________________    
void GPFSimParticle::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
 // Execute action corresponding to a left mouse button click 
 //on a GPFSimParticle object in the window

 gPad->SetCursor(kHand);
 switch (event) {
   case kButton1Down:
     Print();
     break;
   default:break;
 }    
     
}
//______________________________________________________________________________
void GPFSimParticle::Draw()
{
 TGraph::Draw(option_.data());
}
