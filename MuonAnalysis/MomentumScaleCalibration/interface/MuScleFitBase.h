#ifndef MUSCLEFITBASE_H
#define MUSCLEFITBASE_H
/**
 * This class is used as a base for MuSlceFit. The reason for putting some of the methods
 * inside this base class is that they are used also by the TestCorrection analyzer.
 */

#include <map>
#include <string>
#include "TFile.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Histograms.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitUtils.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;

class MuScleFitBase
{
public:
  MuScleFitBase(const edm::ParameterSet& iConfig) :
    theMuonType_( iConfig.getParameter<int>( "MuonType" ) ),
    theMuonLabel_( iConfig.getParameter<edm::InputTag>( "MuonLabel" ) ),
    theRootFileName_( iConfig.getUntrackedParameter<string>("OutputFileName") ),
    debug_( iConfig.getUntrackedParameter<int>("debug",0) )
  {}
  virtual ~MuScleFitBase() {}
protected:
  /// Create the histograms map
  void fillHistoMap(TFile* outputFile, unsigned int iLoop);
  /// Clean the histograms map
  void clearHistoMap();
  /// Save the histograms map to file
  void writeHistoMap();

  /**
   * Read probability distributions from the database.
   * These are 2-D PDFs containing a grid of 1000x1000 values of the
   * integral of Lorentz * Gaussian as a function
   * of mass and resolution of a given measurement,
   * for each of the six considered diLmuon resonances.
   */
  // void readProbabilityDistributions( const edm::EventSetup & eventSetup );
  /// Raed probability distributions from a local root file.
  void readProbabilityDistributionsFromFile();

  int theMuonType_;
  edm::InputTag theMuonLabel_;
  string theRootFileName_;

  int debug_;

  /// The map of histograms
  map<string, Histograms*> mapHisto;
  TProfile * Mass_P;
  TProfile * Mass_fine_P;
  TH2D * PtminvsY;
  TH2D * PtmaxvsY;
  TH2D * EtamuvsY;
  TH1D * Y;
  TH2D * MY;
  TProfile * MYP;
  TProfile * YL;
  TProfile * PL;
  TProfile * PTL;
  TH1D * GM;
  TH1D * SM;
  TH1D *GSM;
  HCovarianceVSxy * massResolutionVsPtEta_;
};

#endif
