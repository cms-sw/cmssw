#ifndef _hgc_jet_tools_h_
#define _hgc_jet_tools_h_

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"


//
std::pair<float,float> betaVariables(const reco::PFJet * jet,
				     const reco::Vertex * vtx,
				     const reco::VertexCollection & allvtx);


#endif
