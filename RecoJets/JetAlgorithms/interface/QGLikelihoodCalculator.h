// ------------------------------------------------------------
//  
//    QGLikelihoodCalculator - Class
//    for the computation of the QG likelihood.
//    Needs files provided by having run the
//    Ntp1Finalizer_QG on QCD samples.
//
// ------------------------------------------------------------

#ifndef QGLikelihoodCalculator_h
#define QGLikelihoodCalculator_h

#include <string>

#include "TFile.h"
#include "TH1F.h"
#include <map>



class QGLikelihoodCalculator {

 public:
//  QGLikelihoodCalculator( const std::string& fileName="QG_QCD_Pt-15to3000_TuneZ2_Flat_7TeV_pythia6_Summer11-PU_S3_START42_V11-v2.root", unsigned nPtBins=21, unsigned int nRhoBins=25 );
  QGLikelihoodCalculator( TString dataDir, Bool_t chs = false);
   ~QGLikelihoodCalculator();

  float computeQGLikelihoodPU( float pt, float rhoPF, int nCharged, int nNeutral, float ptD, float rmsCand=-1. );
  float computeQGLikelihood2012( float pt, float eta, float rho, int nPFCandidates_QC, float ptD_QC, float axis2_QC ); //new
  Float_t QGvalue(std::map<TString, Float_t>);

  float likelihoodProduct( float nCharged, float nNeutral, float ptD, float rmsCand, TH1F* h1_nCharged, TH1F* h1_nNeutral, TH1F* h1_ptD, TH1F* h1_rmsCand);



 private:

  TFile* histoFile_;
  std::map<std::string,TH1F*> plots_;
  unsigned int nPtBins_;
  unsigned int nRhoBins_;

};


#endif
