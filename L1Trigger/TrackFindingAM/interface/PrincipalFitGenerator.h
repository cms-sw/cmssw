#ifndef _PRINCIPALFITGENERATOR_H_
#define _PRINCIPALFITGENERATOR_H_

#include <string>
#include <TChain.h>
#include <TSystemDirectory.h>
#include "SectorTree.h"
#include "PrincipalTrackFitter.h"


/**
   \brief Class used to generate the parameters of a Principal Components Analysis Fitter
**/
class PrincipalFitGenerator{

 private:
  string inputDirectory;
  SectorTree* st;

  // Containers to load the TTree branches
  int m_stub;
  vector<int>                   m_stub_tp;  // tp of the particule
  vector<int>                   m_stub_layer;  // Layer du stub (5 a 10 pour les 6 layers qui nous interessent)
  vector<int>                   m_stub_module; // Position en Z du module contenant le stub
  vector<int>                   m_stub_ladder; // Position en PHI du module contenant le stub
  vector<float>                 m_stub_pxGEN;  // Px de la particule initiale (en GeV/c)
  vector<float>                 m_stub_pyGEN;  // Py de la particule initiale (en GeV/c)
  vector<float>                 m_stub_etaGEN;  // Eta de la particule initiale
  vector<float>                 m_stub_x;      // x coordinate of the hit
  vector<float>                 m_stub_y;      // y coordinate of the hit
  vector<float>                 m_stub_z;      // z coordinate of the hit
  vector<float>                 m_clus_zmc;
  vector<int>                   m_stub_clust1;
  vector<int>                   m_stub_pdg;
  vector<float>                 m_stub_x0;
  vector<float>                 m_stub_y0;
  vector<float>                 m_stub_z0;

  

  vector<int>                   *p_m_stub_tp;
  vector<int>                   *p_m_stub_layer;
  vector<int>                   *p_m_stub_module;
  vector<int>                   *p_m_stub_ladder;
  vector<float>                 *p_m_stub_pxGEN;
  vector<float>                 *p_m_stub_pyGEN;
  vector<float>                 *p_m_stub_etaGEN;
  vector<float>                 *p_m_stub_x;
  vector<float>                 *p_m_stub_y;
  vector<float>                 *p_m_stub_z;
  vector<float>                 *p_m_clus_zmc;
  vector<int>                   *p_m_stub_clust1;
  vector<int>                   *p_m_stub_pdg;
  vector<float>                 *p_m_stub_x0;
  vector<float>                 *p_m_stub_y0;
  vector<float>                 *p_m_stub_z0;

  TChain* createTChain();
  void generatePrincipal(map<int,pair<float,float> > eta_limits, float min_pt, float max_pt, float min_eta, float max_eta);
  void generateMultiDim(map<int,pair<float,float> > eta_limits, float min_pt, float max_pt, float min_eta, float max_eta);
  bool selectTrack(vector<int> &tracker_layers, int* layers, vector<int> &ladder_per_layer, vector<int> &module_per_layer,
		   map<int,pair<float,float> > &eta_limits, float min_pt, float max_pt, float min_eta, float max_eta);

 public:
  PrincipalFitGenerator(string f, SectorTree *s);
  void generate(map<int,pair<float,float> > eta_limits, float min_pt, float max_pt, float min_eta, float max_eta);
};

#endif
