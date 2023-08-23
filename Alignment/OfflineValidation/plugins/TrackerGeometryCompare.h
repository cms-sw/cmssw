#ifndef Alignment_OfflineValidation_TrackerGeometryCompare_h
#define Alignment_OfflineValidation_TrackerGeometryCompare_h

/** \class TrackerGeometryCompare
 *
 * Module that reads survey info from DB and prints them out.
 *
 *  $Date: 2012/12/02 22:13:12 $
 *  $Revision: 1.14 $
 *  \author Nhan Tran
 *
 * ********
 * ******** Including surface deformations in the geometry comparison ******** 
 * ********
 *
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"

#include "Alignment/CommonAlignment/interface/AlignTools.h"

//******** Single include for the TkMap *************
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
//***************************************************

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <algorithm>
#include <string>
#include "TTree.h"
#include "TH1D.h"

class AlignTransform;
class TrackerTopology;

class TrackerGeometryCompare : public edm::one::EDAnalyzer<> {
public:
  typedef AlignTransform SurveyValue;
  typedef Alignments SurveyValues;

  /// Do nothing. Required by framework.
  TrackerGeometryCompare(const edm::ParameterSet&);

  /// Read from DB and print survey info.
  void beginJob() override;

  void endJob() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  //parameters
  std::vector<align::StructureType> m_theLevels;
  //std::vector<int> theSubDets;

  //compare surface deformations
  void compareSurfaceDeformations(TTree* _inputTree11, TTree* _inputTree12);
  //compares two geometries
  void compareGeometries(Alignable* refAli,
                         Alignable* curAli,
                         const TrackerTopology* tTopo,
                         const edm::EventSetup& iSetup);
  //filling the ROOT file
  void fillTree(Alignable* refAli,
                const AlgebraicVector& diff,  // typedef CLHEP::HepVector      AlgebraicVector;
                const TrackerTopology* tTopo,
                const edm::EventSetup& iSetup);
  //for filling identifiers
  void fillIdentifiers(int subdetlevel, int rawid, const TrackerTopology* tTopo);
  //converts surveyRcd into alignmentRcd
  void surveyToTracker(AlignableTracker* ali, Alignments* alignVals, AlignmentErrorsExtended* alignErrors);
  //need for conversion for surveyToTracker
  void addSurveyInfo(Alignable* ali);
  //void createDBGeometry(const edm::EventSetup& iSetup);
  void createROOTGeometry(const edm::EventSetup& iSetup);

  // for common tracker system
  void setCommonTrackerSystem();
  void diffCommonTrackerSystem(Alignable* refAli, Alignable* curAli);
  bool passIdCut(uint32_t);

  const edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  const edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4hep_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;
  const edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> pixQualityToken_;
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;

  AlignableTracker* referenceTracker;
  AlignableTracker* dummyTracker;
  AlignableTracker* currentTracker;

  unsigned int theSurveyIndex;
  const Alignments* theSurveyValues;
  const SurveyErrors* theSurveyErrors;

  // configurables
  const std::vector<std::string> levelStrings_;
  std::string moduleListName_;
  std::string inputFilename1_;
  std::string inputFilename2_;
  std::string inputTreenameAlign_;
  std::string inputTreenameDeform_;
  bool fromDD4hep_;
  bool writeToDB_;
  std::string weightBy_;
  std::string setCommonTrackerSystem_;
  bool detIdFlag_;
  std::string detIdFlagFile_;
  bool weightById_;
  std::string weightByIdFile_;
  std::vector<unsigned int> weightByIdVector_;

  std::vector<uint32_t> detIdFlagVector_;
  align::StructureType commonTrackerLevel_;
  align::GlobalVector TrackerCommonT_;
  align::GlobalVector TrackerCommonR_;
  align::PositionType TrackerCommonCM_;

  std::ifstream moduleListFile_;
  std::vector<int> moduleList_;
  int moduleInList_;

  //root configuration
  std::string filename_;
  std::string surfdir_;
  TFile* theFile_;
  TTree* alignTree_;
  TFile* inputRootFile1_;
  TFile* inputRootFile2_;
  TTree* inputTree01_;
  TTree* inputTree02_;
  TTree* inputTree11_;
  TTree* inputTree12_;

  /**\ Tree variables */
  int id_, badModuleQuality_, inModuleList_, level_, mid_, mlevel_, sublevel_, useDetId_, detDim_;
  float xVal_, yVal_, zVal_, rVal_, etaVal_, phiVal_, alphaVal_, betaVal_, gammaVal_;
  // changes in global variables
  float dxVal_, dyVal_, dzVal_, drVal_, dphiVal_, dalphaVal_, dbetaVal_, dgammaVal_;
  // changes local variables: u, v, w, alpha, beta, gamma
  float duVal_, dvVal_, dwVal_, daVal_, dbVal_, dgVal_;
  float surWidth_, surLength_;
  uint32_t identifiers_[6];
  double surRot_[9];
  int type_;
  double surfDeform_[13];

  int m_nBins_;
  double m_rangeLow_;
  double m_rangeHigh_;

  bool firstEvent_;

  std::vector<TrackerMap> m_vtkmap_;

  std::map<std::string, TH1D*> m_h1_;
};

#endif
