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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;

class MuScleFitBase
{
public:
  MuScleFitBase(const edm::ParameterSet& iConfig) :
    probabilitiesFileInPath_( iConfig.getUntrackedParameter<string>( "ProbabilitiesFileInPath" , "MuonAnalysis/MomentumScaleCalibration/test/Probs_new_Horace_CTEQ_1000.root" ) ),
    probabilitiesFile_( iConfig.getUntrackedParameter<string>( "ProbabilitiesFile" , "" ) ),
    theMuonType_( iConfig.getParameter<int>( "MuonType" ) ),
    theMuonLabel_( iConfig.getParameter<edm::InputTag>( "MuonLabel" ) ),
    theRootFileName_( iConfig.getUntrackedParameter<string>("OutputFileName") ),
    theGenInfoRootFileName_( iConfig.getUntrackedParameter<string>("OutputGenInfoFileName", "genSimRecoPlots.root") ),
    debug_( iConfig.getUntrackedParameter<int>("debug",0) ),
    useType_( iConfig.getUntrackedParameter<unsigned int>("UseType",0) )
  {}
  virtual ~MuScleFitBase() {}
protected:
  /// Create the histograms map
  void fillHistoMap(TFile* outputFile, unsigned int iLoop);
  /// Clean the histograms map
  void clearHistoMap();
  /// Save the histograms map to file
  void writeHistoMap( const unsigned int iLoop );

  /**
   * Read probability distributions from the database.
   * These are 2-D PDFs containing a grid of 1000x1000 values of the
   * integral of Lorentz * Gaussian as a function
   * of mass and resolution of a given measurement,
   * for each of the six considered di-muon resonances.
   */
  // void readProbabilityDistributions( const edm::EventSetup & eventSetup );
  /// Raed probability distributions from a local root file.
  void readProbabilityDistributionsFromFile();

  string probabilitiesFileInPath_;
  string probabilitiesFile_;

  int theMuonType_;
  edm::InputTag theMuonLabel_;
  string theRootFileName_;
  string theGenInfoRootFileName_;

  int debug_;

  unsigned int useType_;


  /// The files were the histograms are saved
  std::vector<TFile*> theFiles_;

  /// The map of histograms
  map<string, Histograms*> mapHisto_;
};

#endif
