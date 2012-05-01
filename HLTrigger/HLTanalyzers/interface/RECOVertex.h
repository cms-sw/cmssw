#ifndef RECOVTX_H
#define RECOVTX_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <vector>
#include "TChain.h"

#include "DataFormats/VertexReco/interface/Vertex.h" 
#include "DataFormats/VertexReco/interface/VertexFwd.h" 


/** \class RECOVertex
  *  
  * $Date: Dec 2009
  * $Revision: 
  */
class RECOVertex {
public:
  RECOVertex(); 
  ~RECOVertex();

  void setup(const edm::ParameterSet& pSet, TTree* tree, std::string vertexType);
  void clear(void);

  /** Analyze the Data */
  void analyze(edm::Handle<reco::VertexCollection> recoVertexs, TTree* tree);

private:

  // Tree variables
  int   NVrtx;
  float *VertexCand_x, *VertexCand_y, *VertexCand_z;
  int   *VertexCand_tracks;
  float *VertexCand_chi2;
  float *VertexCand_ndof;

  // input variables
  bool _Debug;

};

#endif
