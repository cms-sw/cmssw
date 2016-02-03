#include "L1Trigger/L1TNtuples/interface/L1AnalysisRCT.h"

// need of maxRCTREG ??


L1Analysis::L1AnalysisRCT::L1AnalysisRCT()
{
}

L1Analysis::L1AnalysisRCT::L1AnalysisRCT(int maxRCTREG)
{
  rct_.maxRCTREG_=maxRCTREG;
  rct_.Reset();
}

L1Analysis::L1AnalysisRCT::~L1AnalysisRCT()
{

}

void L1Analysis::L1AnalysisRCT::SetHdRCT(const edm::Handle < L1CaloRegionCollection > rgn)
{ 

  // Regions
    rct_.RegSize=rgn->size();
    for (L1CaloRegionCollection::const_iterator ireg = rgn->begin();
	 ireg != rgn->end(); ireg++) {
	 
   // local eta phi
      rct_.RegEta.push_back( ireg->rctEta() );
      rct_.RegPhi.push_back( ireg->rctPhi() );
   // global eta phi   
      rct_.RegGEta.push_back( ireg->gctEta() );      
      rct_.RegGPhi.push_back( ireg->gctPhi() );
            
      rct_.RegRnk.push_back( ireg->et() );
      rct_.RegVeto.push_back( ireg->tauVeto() );
      rct_.RegBx.push_back( ireg->bx() );
      rct_.RegOverFlow.push_back( ireg->overFlow() );
      rct_.RegMip.push_back( ireg->mip() );
      rct_.RegFGrain.push_back( ireg->fineGrain() );
     }
   
}

void L1Analysis::L1AnalysisRCT::SetEmRCT(const edm::Handle < L1CaloEmCollection > em)
{ 
   
  // Isolated and non-isolated EM
  rct_.EmSize = em->size();
  for (L1CaloEmCollection::const_iterator emit = em->begin(); emit != em->end(); emit++) {
      rct_.IsIsoEm.push_back(  emit->isolated() );
      rct_.EmEta.push_back( emit->regionId().ieta() );
      rct_.EmPhi.push_back( emit->regionId().iphi() );
      rct_.EmRnk.push_back( emit->rank() );
      rct_.EmBx.push_back( emit->bx() );
  } 
  
}

