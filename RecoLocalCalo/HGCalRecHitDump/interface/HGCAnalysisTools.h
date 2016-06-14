#ifndef _hgc_analysis_tools_h_
#define _hgc_analysis_tools_h_

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>

#include <vector>


/**
   @short gets lambda for a given layer
 */
enum HGCWeightType { LAMBDA, X0, DEDX };
float getLambdaForHGCLayer(int hit_layer);

/**
   @short tries to find the interaction position based on G4 information
   @return interaction position and a flag for the interaction type
 */
struct G4InteractionPositionInfo
{
  math::XYZVectorD pos;
  int info;
};
G4InteractionPositionInfo getInteractionPosition(const std::vector<SimTrack> *SimTk, 
						 const std::vector<SimVertex> *SimVtx, 
						 int barcode,
						 bool debug=false);

template<class T>
std::vector<fastjet::PseudoJet> runFastJetOn(const std::vector<const T*> &input,float cone=0.4,fastjet::JetAlgorithm algo=fastjet::kt_algorithm,float ptmin=1.0)
{
  std::vector<fastjet::PseudoJet> input_particles;
  for(typename std::vector<const T*>::const_iterator c_it=input.begin(); c_it!=input.end(); c_it++)
    {
      // create a fastjet::PseudoJet with these components and put it onto
      // back of the input_particles vector
      float energy = (*c_it)->energy();
      float eta    = (*c_it)->eta();
      float phi    = (*c_it)->phi();
      float et     = energy/TMath::CosH(eta);
      input_particles.push_back(fastjet::PseudoJet(et*TMath::Cos(phi),et*TMath::Sin(phi),et*TMath::SinH(eta),energy));
    }
  
  // create a jet definition: a jet algorithm with a given radius parameter
  fastjet::JetDefinition jet_def(algo, cone);
  
  // run the jet clustering with the above jet definition
  fastjet::ClusterSequence clust_seq(input_particles, jet_def);
  
  // get the resulting jets ordered in pt
  return sorted_by_pt(clust_seq.inclusive_jets(ptmin));
}


#endif
