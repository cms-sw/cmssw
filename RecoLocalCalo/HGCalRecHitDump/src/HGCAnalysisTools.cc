#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCAnalysisTools.h"
#include <vector>

using namespace std;

//
float getLambdaForHGCLayer(int hit_layer)
{
  if (hit_layer==1)                       return 0.01; 
  else if(hit_layer>=2 && hit_layer<=11)  return 0.036;
  else if(hit_layer>=12 && hit_layer<=21) return 0.043;
  else if(hit_layer>=22 && hit_layer<=30) return 0.056;
  else if(hit_layer==31)                  return 0.338;
  else if(hit_layer>=32 && hit_layer<=42) return 0.273;
  else if(hit_layer>42)                   return 0.475;
  return 0;
}

//
G4InteractionPositionInfo getInteractionPosition(const std::vector<SimTrack> *SimTk, 
						 const std::vector<SimVertex> *SimVtx, 
						 int barcode,
						 bool debug)
{
  G4InteractionPositionInfo toRet;
  toRet.pos=math::XYZVectorD(0,0,0);
  toRet.info=0;

  //loop over vertices
  for (const SimVertex &simVtx : *SimVtx) 
    {
      //require the parent to be the given barcode
      bool noParent( simVtx.noParent() );
      if(noParent) continue;
      int pIdx( simVtx.parentIndex() );
      if( pIdx!=barcode) continue;

      int vtxIdx(simVtx.vertexId());
      int rawTkMult(0),eTkMult(0),gTkMult(0),nTkMult(0),nucleiTkMult(0),pTkMult(0);
      for (const SimTrack &vtxTk : *SimTk)
	{
	  int tkVtxIdx( vtxTk.vertIndex() ); 
	  if(tkVtxIdx!=vtxIdx) continue;
	  
	  int tkType=vtxTk.type();
	  rawTkMult++;
	  eTkMult      += (abs(tkType)==11);
	  gTkMult      += (abs(tkType)==22);
	  nTkMult      += (abs(tkType)==2112 || abs(tkType)==2212);
	  nucleiTkMult += (abs(tkType)>1000000000);
	  pTkMult      += (abs(tkType)==211 || abs(tkType)==111);
	}
      
      if(rawTkMult<2) continue;

      toRet.pos=math::XYZVectorD(simVtx.position());
      if(rawTkMult==3 && nTkMult==1 && nucleiTkMult==1 && pTkMult==1) toRet.info=1;
      return toRet;
    }

  return toRet;
}
