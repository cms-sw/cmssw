#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/RECOVertex.h"
#include "HLTMessages.h"

static const size_t kMaxVrt = 50; 

RECOVertex::RECOVertex() {

  //set parameter defaults 
  _Debug=false;

  NVrtx                 = 0;
  VertexCand_x          = new float[kMaxVrt];
  VertexCand_y          = new float[kMaxVrt];
  VertexCand_z          = new float[kMaxVrt];
  VertexCand_tracks     = new int[kMaxVrt];
  VertexCand_chi2       = new float[kMaxVrt];
  VertexCand_ndof       = new float[kMaxVrt];
}

RECOVertex::~RECOVertex() {
  delete[] VertexCand_x;
  delete[] VertexCand_y;
  delete[] VertexCand_z;
  delete[] VertexCand_tracks;
  delete[] VertexCand_chi2;
  delete[] VertexCand_ndof;
}

void RECOVertex::clear() 
{
  NVrtx          = 0;
  std::memset(VertexCand_x,            '\0', kMaxVrt * sizeof(float));
  std::memset(VertexCand_y,            '\0', kMaxVrt * sizeof(float));
  std::memset(VertexCand_z,            '\0', kMaxVrt * sizeof(float));
  std::memset(VertexCand_tracks,       '\0', kMaxVrt * sizeof(int));
  std::memset(VertexCand_chi2,         '\0', kMaxVrt * sizeof(float));
  std::memset(VertexCand_ndof,         '\0', kMaxVrt * sizeof(float));
}

/*  Setup the analysis to put the branch-variables into the tree. */
void RECOVertex::setup(const edm::ParameterSet& pSet, TTree* HltTree, std::string vertexType) {

  edm::ParameterSet myHltParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  std::vector<std::string> parameterNames = myHltParams.getParameterNames() ;
  
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
        iParam != parameterNames.end(); iParam++ ){
    if ( (*iParam) == "Debug" ) _Debug =  myHltParams.getParameter<bool>( *iParam );
  }

  TString br_recoNVrt = "recoNVrt";
  br_recoNVrt.Append(vertexType);
  HltTree->Branch(br_recoNVrt,        & NVrtx,            "NVrtx/I");

  TString br_recoVrtX = "recoVrtX";
  br_recoVrtX.Append(vertexType);
  HltTree->Branch(br_recoVrtX,     VertexCand_x,          "recoVrtX[NVrtx]/F");

  TString br_recoVrtY = "recoVrtY";
  br_recoVrtY.Append(vertexType);
  HltTree->Branch(br_recoVrtY,     VertexCand_y,          "recoVrtY[NVrtx]/F");

  TString br_recoVrtZ = "recoVrtZ";
  br_recoVrtZ.Append(vertexType);
  HltTree->Branch(br_recoVrtZ,     VertexCand_z,          "recoVrtZ[NVrtx]/F");

  TString br_recoVrtNtrk = "recoVrtNtrk";
  br_recoVrtNtrk.Append(vertexType);
  HltTree->Branch(br_recoVrtNtrk,  VertexCand_tracks,     "recoVrtNtrk[NVrtx]/I");

  TString br_recoVrtChi2 = "recoVrtChi2";
  br_recoVrtChi2.Append(vertexType);
  HltTree->Branch(br_recoVrtChi2,  VertexCand_chi2,       "recoVrtChi2[NVrtx]/F");

  TString br_recoVrtNdof = "recoVrtNdof";
  br_recoVrtNdof.Append(vertexType);
  HltTree->Branch(br_recoVrtNdof,  VertexCand_ndof,       "recoVrtNdof[NVrtx]/F");



}

/* **Analyze the event** */
void RECOVertex::analyze(edm::Handle<reco::VertexCollection> recoVertexs, TTree* HltTree) {

  // reset the tree variables
  clear();

  if ( recoVertexs.isValid() ) {
    const reco::VertexCollection* vertexs = recoVertexs.product();
    reco::VertexCollection::const_iterator vertex_i;

    size_t size = std::min(kMaxVrt, size_t(vertexs->size()) ); 
    NVrtx= size;

    int nVertexCand=0;
    if (_Debug)  std::cout << "Found " << vertexs->size() << " vertices" << std::endl;  
    for (vertex_i = vertexs->begin(); vertex_i != vertexs->end(); vertex_i++){
      if (nVertexCand>=NVrtx) break;
      VertexCand_x[nVertexCand] = vertex_i->x();
      VertexCand_y[nVertexCand] = vertex_i->y();
      VertexCand_z[nVertexCand] = vertex_i->z();
      VertexCand_tracks[nVertexCand] = vertex_i->tracksSize();
      VertexCand_chi2[nVertexCand] = vertex_i->chi2();
      VertexCand_ndof[nVertexCand] = vertex_i->ndof();
      if (_Debug) { 
	std::cout << "RECOVertex -- VX, VY VZ   = " 
		  << VertexCand_x[nVertexCand] << " "
		  << VertexCand_y[nVertexCand] << " "
		  << VertexCand_z[nVertexCand]
		  << std::endl;
	std::cout << "RECOVertex -- Ntracks, Chi2/Dof   = " 
		  << VertexCand_tracks[nVertexCand] << " "
		  << VertexCand_chi2[nVertexCand] << " / " << VertexCand_ndof[nVertexCand]
		  << std::endl;
      }
      nVertexCand++;
      
    }
  }
}
  
