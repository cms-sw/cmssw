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

class MuScleFitBase
{
public:
  MuScleFitBase(const edm::ParameterSet& iConfig) :
    probabilitiesFileInPath_( iConfig.getUntrackedParameter<std::string>( "ProbabilitiesFileInPath" , "MuonAnalysis/MomentumScaleCalibration/test/Probs_new_Horace_CTEQ_1000.root" ) ),
    probabilitiesFile_( iConfig.getUntrackedParameter<std::string>( "ProbabilitiesFile" , "" ) ),
    theMuonType_( iConfig.getParameter<int>( "MuonType" ) ),
    theMuonLabel_( iConfig.getParameter<edm::InputTag>( "MuonLabel" ) ),
    theRootFileName_( iConfig.getUntrackedParameter<std::string>("OutputFileName") ),
    theGenInfoRootFileName_( iConfig.getUntrackedParameter<std::string>("OutputGenInfoFileName", "genSimRecoPlots.root") ),
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

  std::string probabilitiesFileInPath_;
  std::string probabilitiesFile_;

  int theMuonType_;
  edm::InputTag theMuonLabel_;
  std::string theRootFileName_;
  std::string theGenInfoRootFileName_;

  int debug_;

  unsigned int useType_;


  /// The files were the histograms are saved
  std::vector<TFile*> theFiles_;

  /// The map of histograms
  std::map<std::string, Histograms*> mapHisto_;
};

#endif
