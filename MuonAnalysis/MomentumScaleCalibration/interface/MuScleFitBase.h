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

  /// Read probability distributions from a local root file.
  void readProbabilityDistributionsFromFile();

  string probabilitiesFileInPath_;
  string probabilitiesFile_;

  int theMuonType_;
  edm::InputTag theMuonLabel_;
  string theRootFileName_;
  string theGenInfoRootFileName_;

  int debug_;

  unsigned int useType_;

  /// Functor used to compute the normalization integral of probability functions
  class ProbForIntegral
  {
  public:
    ProbForIntegral( const double & massResol, const int iRes, const int iY, const bool isZ ) :
      massResol_(massResol),
      iRes_(iRes), iY_(iY), isZ_(isZ)
    {}
    double operator()(const double * mass, const double *)
    {
      if( isZ_ ) {
        return( MuScleFitUtils::probability(*mass, massResol_, MuScleFitUtils::GLZValue, MuScleFitUtils::GLZNorm, iRes_, iY_) );
      }
      return( MuScleFitUtils::probability(*mass, massResol_, MuScleFitUtils::GLValue, MuScleFitUtils::GLNorm, iRes_, iY_) );
    }
  protected:
    double massResol_;
    int iRes_, iY_;
    bool isZ_;
  };

  /// The files were the histograms are saved
  std::vector<TFile*> theFiles_;

  /// The map of histograms
  map<string, Histograms*> mapHisto_;
};

#endif
