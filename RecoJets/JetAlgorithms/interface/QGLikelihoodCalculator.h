// ------------------------------------------------------------
//  
//    QGLikelihoodCalculator - Class
//    for the computation of the QG likelihood.
//    Needs files provided by having run the
//    Ntp1Finalizer_QG on QCD samples.
//
// ------------------------------------------------------------

#ifndef JetAlgorithms_QGLikelihoodCalculator_h
#define JetAlgorithms_QGLikelihoodCalculator_h

#include <string>

#include "TFile.h"
#include "TH1F.h"
#include <map>


class QGLikelihoodCalculator{

 public:
  QGLikelihoodCalculator( TString dataDir, bool chs = false);
   ~QGLikelihoodCalculator(){};

  float computeQGLikelihood2012( float pt, float eta, float rho, int nPFCandidates_QC, float ptD_QC, float axis2_QC );
  float likelihoodProduct( float nCharged, float nNeutral, float ptD, float rmsCand, TH1F* h1_nCharged, TH1F* h1_nNeutral, TH1F* h1_ptD, TH1F* h1_rmsCand);

 private:
  TFile* histoFile_;
  std::map<std::string,TH1F*> plots_;
  unsigned int nPtBins_;
  unsigned int nRhoBins_;

};


#endif
