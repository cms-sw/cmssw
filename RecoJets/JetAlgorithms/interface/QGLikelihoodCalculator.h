// ------------------------------------------------------------
//  
//    QGLikelihoodCalculator - Class
//    for the computation of the QG likelihood.
//
// ------------------------------------------------------------

#ifndef JetAlgorithms_QGLikelihoodCalculator_h
#define JetAlgorithms_QGLikelihoodCalculator_h

#include <string>

#include "TFile.h"
#include "TH1F.h"


class QGLikelihoodCalculator{

 public:
  QGLikelihoodCalculator( TString dataDir, bool chs = false);
   ~QGLikelihoodCalculator(){};

  float computeQGLikelihood2012( float pt, float eta, float rho, float nPFCandidates_QC, float ptD_QC, float axis2_QC );
  float likelihoodProduct( float nCharged, float nNeutral, float ptD, float rmsCand, TH1F* h1_nCharged, TH1F* h1_nNeutral, TH1F* h1_ptD, TH1F* h1_rmsCand);

 private:
  void loadTH1F(int etaIndex, int qgIndex, int varIndex, int ptIndex, int rhoIndex);
  int indexTH1F(int etaIndex, int qgIndex, int varIndex, int ptIndex, int rhoIndex);

  TFile* histoFile;
  std::vector<TH1F*> plots;

  std::vector<int> RhoBins;
  std::vector<int> PtBins;
};


#endif
