#ifndef __L1Analysis_L1AnalysisRecoVertex_H__
#define __L1Analysis_L1AnalysisRecoVertex_H__

//-------------------------------------------------------------------------------
// Created 15/04/2010 - A.C. Le Bihan
// 
//
// Original code : L1Trigger/L1TNtuples/L1TrackVertexRecoTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "L1AnalysisRecoVertexDataFormat.h"


namespace L1Analysis
{
  class L1AnalysisRecoVertex 
  {
  public:
    L1AnalysisRecoVertex(){Reset();};
    ~L1AnalysisRecoVertex(){};
    
    void SetVertices(const edm::Handle<reco::VertexCollection> vertices, unsigned maxVtx); 
    L1AnalysisRecoVertexDataFormat * getData() {return &recoVertex_;}
    void Reset() {recoVertex_.Reset();}

  private :
    L1AnalysisRecoVertexDataFormat recoVertex_;
  }; 
}
#endif


