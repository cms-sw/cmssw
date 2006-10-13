#ifndef RecoTracker_TrackProducer_FamosRecHitAnalysis_h
#define RecoTracker_TrackProducer_FamosRecHitAnalysis_h 1

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>

class FamosRecHitAnalysis : public edm::EDAnalyzer
{
public:
  
  explicit FamosRecHitAnalysis(const edm::ParameterSet& pset);
  
  virtual ~FamosRecHitAnalysis();
  virtual void beginJob(const edm::EventSetup& setup);
  virtual void endJob(); 
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  
private:
  edm::ParameterSet _pset;
  edm::InputTag theRecHits_;
  
  TFile* theRootFile;
  
  void book();
  void bookValues( std::vector<TH1F*>& histos_x , std::vector<TH1F*>& histos_y , std::vector<TH1F*>& histos_z , int nBin , float range , char* det , unsigned int nHist );
  void bookErrors( std::vector<TH1F*>& histos_x , std::vector<TH1F*>& histos_y , std::vector<TH1F*>& histos_z , int nBin , float range , char* det , unsigned int nHist );
  void bookNominals( std::vector<TH1F*>& histos_x , int nBin , float range , char* det , unsigned int nHist );
  void write(std::vector<TH1F*> histos);
  
  void chooseHist(unsigned int rawid, TH1F*& hist_x , TH1F*& hist_y , TH1F*& hist_z, TH1F*& hist_err_x , TH1F*& hist_err_y , TH1F*& hist_err_z);
  
  // ROOT
  void rootStyle();
  void rootMacroStrip( std::vector<TH1F*>& histos_x      , std::vector<TH1F*>& histos_y     , std::vector<TH1F*>& histos_z     ,
		       std::vector<TH1F*>& histos_err_x  , std::vector<TH1F*>& histos_err_y , std::vector<TH1F*>& histos_err_z ,
		       std::vector<TH1F*>& histos_nom_x   );
  void rootComparison( std::vector<TH1F*> histos_value , std::vector<TH1F*> histos_nominal , int binFactor );
  //
  
  
  // Histograms
  // Detectors
  // TIB - 4 different detectors
  static const unsigned int nHist_TIB = 4;
  std::vector<TH1F*> histos_TIB_x;
  std::vector<TH1F*> histos_TIB_y;
  std::vector<TH1F*> histos_TIB_z;
  std::vector<TH1F*> histos_TIB_err_x;
  std::vector<TH1F*> histos_TIB_err_y;
  std::vector<TH1F*> histos_TIB_err_z;
  std::vector<TH1F*> histos_TIB_nom_x;
  // TID - 3 different detectors
  static const unsigned int nHist_TID = 3;
  std::vector<TH1F*> histos_TID_x;
  std::vector<TH1F*> histos_TID_y;
  std::vector<TH1F*> histos_TID_z;
  std::vector<TH1F*> histos_TID_err_x;
  std::vector<TH1F*> histos_TID_err_y;
  std::vector<TH1F*> histos_TID_err_z;
  std::vector<TH1F*> histos_TID_nom_x;
  // TOB - 6 different detectors
  static const unsigned int nHist_TOB = 6;
  std::vector<TH1F*> histos_TOB_x;
  std::vector<TH1F*> histos_TOB_y;
  std::vector<TH1F*> histos_TOB_z;
  std::vector<TH1F*> histos_TOB_err_x;
  std::vector<TH1F*> histos_TOB_err_y;
  std::vector<TH1F*> histos_TOB_err_z;
  std::vector<TH1F*> histos_TOB_nom_x;
  // TEC - 7 different detectors
  static const unsigned int nHist_TEC = 7;
  std::vector<TH1F*> histos_TEC_x;
  std::vector<TH1F*> histos_TEC_y;
  std::vector<TH1F*> histos_TEC_z;
  std::vector<TH1F*> histos_TEC_err_x;
  std::vector<TH1F*> histos_TEC_err_y;
  std::vector<TH1F*> histos_TEC_err_z;
  std::vector<TH1F*> histos_TEC_nom_x;
  //
  
};

#endif
