// Why these variables? The names below make more sense to me - MG
//#ifndef RecoTracker_TrackProducer_FamosRecHitAnalysis_h
//#define RecoTracker_TrackProducer_FamosRecHitAnalysis_h 
#ifndef FastSimulation_TrackingRecHitProducer_FamosRecHitAnalysis_h
#define FastSimulation_TrackingRecHitProducer_FamosRecHitAnalysis_h
 
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>

class TrackerGeometry;
class TrackerTopology;

class FamosRecHitAnalysis : public edm::stream::EDAnalyzer <>
{
public:
  
  explicit FamosRecHitAnalysis(const edm::ParameterSet& pset);
  
  virtual ~FamosRecHitAnalysis();
  virtual void beginRun(edm::Run const&, const edm::EventSetup & ) override;
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  
private:
  edm::ParameterSet _pset;
  edm::InputTag theRecHits_;
  
  TFile* theRootFile;
  
  const TrackerGeometry* geometry;
  bool isFlipped(const PixelGeomDetUnit* theDet) const;
  
  void book();
  void bookValues( std::vector<TH1F*>& histos_x , std::vector<TH1F*>& histos_y , std::vector<TH1F*>& histos_z , int nBin , float range , char* det , unsigned int nHist );
  void bookErrors( std::vector<TH1F*>& histos_x , std::vector<TH1F*>& histos_y , std::vector<TH1F*>& histos_z , int nBin , float range , char* det , unsigned int nHist );
  void bookNominals( std::vector<TH1F*>& histos_x , int nBin , float range , char* det , unsigned int nHist );
  void bookEnergyLosses( std::vector<TH1F*>& histos_x, int nBin, float range, char* det, unsigned int nHist );
  void loadPixelData(TFile* pixelMultiplicityFile, TFile* pixelBarrelResolutionFile, TFile* pixelForwardResolutionFile);
  void loadPixelData(TFile* pixelDataFile, unsigned int nMultiplicity, std::string histName,
		     std::vector<TH1F*>& theMultiplicityProbabilities, bool isBig = false);
  void loadPixelData( TFile* pixelDataFile, unsigned int nMultiplicity, int nBins, double binWidth,
		      std::vector<TH1F*>& theResolutionHistograms, bool isAlpha, bool isBig = false);
  void bookPixel( std::vector<TH1F*>& histos_alpha , std::vector<TH1F*>& histos_beta , std::vector<TH1F*>& histos_nom_alpha  , 
                  std::vector<TH1F*>& histos_nom_beta,
                  std::vector<TH1F*>& histos_dedx_alpha, std::vector<TH1F*>& histos_dedx_beta,
                  char* det );
  void bookPixel( std::vector<TH1F*>& histos_alpha , std::vector<TH1F*>& histos_beta , std::vector<TH1F*>& histos_nom_alpha  , 
                  std::vector<TH1F*>& histos_nom_beta, 
                  char* det, unsigned int nAlphaMultiplicity ,
                  double resAlpha_binMin , double resAlpha_binWidth , int resAlpha_binN , 
                  unsigned int nBetaMultiplicity ,
                  double resBeta_binMin  , double resBeta_binWidth  , int resBeta_binN  );
  void write(std::vector<TH1F*> histos);
  
  void chooseHist( unsigned int rawid ,
		   TH1F*& hist_x , TH1F*& hist_y , TH1F*& hist_z, TH1F*& hist_err_x , TH1F*& hist_err_y , TH1F*& hist_err_z ,
		   TH1F*& hist_alpha , TH1F*& hist_beta , TH1F*& hist_res_alpha , TH1F*& hist_res_beta , TH1F*& hist_dedx, 
                   TH1F*& hist_dedx_alpha, TH1F*& hist_dedx_beta,
		   unsigned int mult_alpha , unsigned int mult_beta ,
		   double       alpha      , double       beta      , 
                   const bool hasBigPixelInX, const bool hasBigPixelInY, const TrackerTopology *tTopo );
  
  // ROOT
  void rootStyle();
  void rootMacroStrip( std::vector<TH1F*>& histos_x      , std::vector<TH1F*>& histos_y     , std::vector<TH1F*>& histos_z     ,
		       std::vector<TH1F*>& histos_err_x  , std::vector<TH1F*>& histos_err_y , std::vector<TH1F*>& histos_err_z ,
		       std::vector<TH1F*>& histos_nom_x   );
  void rootMacroPixel( std::vector<TH1F*>& histos_angle );
  void rootComparison( std::vector<TH1F*> histos_value , std::vector<TH1F*> histos_nominal ,
		       int binFactor , int yLogScale = 1 , float yMax = -1.);
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
  std::vector<TH1F*> histos_TIB_dedx;
// TID - 3 different detectors
  static const unsigned int nHist_TID = 3;
  std::vector<TH1F*> histos_TID_x;
  std::vector<TH1F*> histos_TID_y;
  std::vector<TH1F*> histos_TID_z;
  std::vector<TH1F*> histos_TID_err_x;
  std::vector<TH1F*> histos_TID_err_y;
  std::vector<TH1F*> histos_TID_err_z;
  std::vector<TH1F*> histos_TID_nom_x;
  std::vector<TH1F*> histos_TID_dedx;
// TOB - 6 different detectors
  static const unsigned int nHist_TOB = 6;
  std::vector<TH1F*> histos_TOB_x;
  std::vector<TH1F*> histos_TOB_y;
  std::vector<TH1F*> histos_TOB_z;
  std::vector<TH1F*> histos_TOB_err_x;
  std::vector<TH1F*> histos_TOB_err_y;
  std::vector<TH1F*> histos_TOB_err_z;
  std::vector<TH1F*> histos_TOB_nom_x;
  std::vector<TH1F*> histos_TOB_dedx;
// TEC - 7 different detectors
  static const unsigned int nHist_TEC = 7;
  std::vector<TH1F*> histos_TEC_x;
  std::vector<TH1F*> histos_TEC_y;
  std::vector<TH1F*> histos_TEC_z;
  std::vector<TH1F*> histos_TEC_err_x;
  std::vector<TH1F*> histos_TEC_err_y;
  std::vector<TH1F*> histos_TEC_err_z;
  std::vector<TH1F*> histos_TEC_nom_x;
  std::vector<TH1F*> histos_TEC_dedx;
  //
  
  // PSimHits
  std::vector<edm::InputTag> trackerContainers;
  //
  
  // Pixel more detailed analysis
  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  bool useCMSSWPixelParameterization;
  // multiplicity bins
  unsigned int nAlphaBarrel, nBetaBarrel, nAlphaForward, nBetaForward;
  // resolution bins
  double resAlphaBarrel_binMin , resAlphaBarrel_binWidth;
  unsigned int resAlphaBarrel_binN;
  double resBetaBarrel_binMin  , resBetaBarrel_binWidth;
  unsigned int resBetaBarrel_binN;
  double resAlphaForward_binMin , resAlphaForward_binWidth;
  unsigned int resAlphaForward_binN;
  double resBetaForward_binMin  , resBetaForward_binWidth;
  unsigned int resBetaForward_binN;
  //
  // ROOT files with nominal distributions
  std::string thePixelMultiplicityFileName;
  std::string thePixelBarrelResolutionFileName;
  std::string thePixelForwardResolutionFileName;
  TFile* thePixelMultiplicityFile;
  TFile* thePixelBarrelResolutionFile;
  TFile* thePixelForwardResolutionFile;
  //
  // internal vector: bins ; external vector: multiplicity
  std::vector<TH1F*> histos_PXB_alpha;
  std::vector<TH1F*> histos_PXB_beta;
  std::vector<TH1F*> histos_PXF_alpha;
  std::vector<TH1F*> histos_PXF_beta;
  std::vector<TH1F*> histos_PXB_nom_alpha;
  std::vector<TH1F*> histos_PXB_nom_beta;
  std::vector<TH1F*> histos_PXF_nom_alpha;
  std::vector<TH1F*> histos_PXF_nom_beta;
   // energy losses
  std::vector<TH1F*> histos_PXB_dedx_alpha;
  std::vector<TH1F*> histos_PXB_dedx_beta;
  std::vector<TH1F*> histos_PXF_dedx_alpha;
  std::vector<TH1F*> histos_PXF_dedx_beta;
  // resolutions
  std::vector<TH1F*> histos_PXB_res_alpha;
  std::vector<TH1F*> histos_PXB_res_beta;
  std::vector<TH1F*> histos_PXF_res_alpha;
  std::vector<TH1F*> histos_PXF_res_beta;
  std::vector<TH1F*> histos_PXB_nom_res_alpha;
  std::vector<TH1F*> histos_PXB_nom_res_beta;
  std::vector<TH1F*> histos_PXF_nom_res_alpha;
  std::vector<TH1F*> histos_PXF_nom_res_beta;
  //  
};

#endif
