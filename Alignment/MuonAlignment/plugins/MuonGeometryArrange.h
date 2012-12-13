#ifndef Alignment_OfflineValidation_MuonGeometryArrange_h
#define Alignment_OfflineValidation_MuonGeometryArrange_h

/** \class MuonGeometryArrange
 *
 * Module that reads survey info from DB and prints them out.
 *
 * Usage:
 *   module comparator = MuonGeometryArrange {
 *
 *   lots of stuff  
 *
 *   }
 *   path p = { comparator }
 *
 *
 *  $Date: 2010/01/04 17:04:08 $
 *  $Revision: 1.3 $
 *  \author Nhan Tran
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"

#include "Alignment/CommonAlignment/interface/AlignTools.h"

#include <algorithm>
#include "TTree.h"

class AlignTransform;
class MuonAlignment;
class TGraph;
class TH2F;

class MuonGeometryArrange:
public edm::EDAnalyzer
{
public:

  typedef AlignTransform SurveyValue;
  typedef Alignments SurveyValues;
  typedef std::vector<Alignable*> Alignables;

  /// Do nothing. Required by framework.
  MuonGeometryArrange(const edm::ParameterSet&);

  /// Read from DB and print survey info.
  virtual void beginJob();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:

  //parameters
  edm::ParameterSet m_params;
  std::vector<align::StructureType> theLevels;
  //std::vector<int> theSubDets;

  //compares two geometries, driver routine
  void compare(Alignable* refAli, Alignable* curAli, Alignable* curAliCopy2);
  void endHist();
  // Map one onto other and compare details
  void compareGeometries(Alignable* refAli, Alignable* curAli, Alignable* curAliCopy2);
  //filling the ROOT file
  void fillTree(Alignable *refAli, AlgebraicVector diff);
  //void createDBGeometry(const edm::EventSetup& iSetup);
  void createROOTGeometry(const edm::EventSetup& iSetup);
  void makeGraph(int sizeI, float smi, float sma, float minV,
    float maxV, TH2F* dxh, TGraph* grx, const char* name, const char* title,
    const char* titleg, const char* axis, float* xp, float* yp, int numEntries);

  bool passIdCut( uint32_t );
  bool checkChosen( Alignable* ali );  // Is ali one of wanted CSC?
  bool passChosen( Alignable* ali );  // Is ali either one of wanted
                                      // CSC or does it contain them?
  bool isMother( Alignable* ali );  // Is ali the container (ring)?

  AlignableMuon* referenceMuon;
  AlignableMuon* dummyMuon;
  AlignableMuon* currentMuon;
  Alignable* inputGeometry1;
  Alignable* inputGeometry2;

  unsigned int theSurveyIndex;
  const Alignments* theSurveyValues;
  const SurveyErrors* theSurveyErrors;

  // configurables
  std::string _inputFilename1;
  std::string _inputFilename2;
  std::string _inputTreename;
  bool _writeToDB;
  std::string _weightBy;
  std::string _setCommonMuonSystem;
  bool _detIdFlag;
  std::string _detIdFlagFile;
  bool _weightById;
  std::string _weightByIdFile;
  std::vector< unsigned int > _weightByIdVector;
  int _endcap;
  int _station;
  int _ring;

  std::vector< uint32_t > _detIdFlagVector;
  align::StructureType _commonMuonLevel;
  align::GlobalVector _MuonCommonT;
  align::EulerAngles _MuonCommonR;
  align::PositionType _MuonCommonCM;

  //root configuration
  std::string _filename;

  struct MGACollection {
    int id;
    int level;
    int mid;
    int mlevel;
    int sublevel;
    float x,y,z;
    float r, phi, eta;
    float alpha, beta, gamma;
    float dx, dy, dz;
    float dr, dphi;       // no deta?
    float dalpha, dbeta, dgamma;
    float ldx, ldy, ldz;
    float ldr, ldphi;       // no deta?
    int useDetId, detDim;
    float rotx, roty, rotz;
    float drotx, droty, drotz;
    float surW, surL;     // surWidth and length
    double surRot[9];
    int phipos;
  };

  std::vector<MGACollection> _mgacollection;
  // Two sets of alignment inputs
  std::string _inputXMLCurrent;
  std::string _inputXMLReference;
  MuonAlignment* inputAlign1;
  MuonAlignment* inputAlign2;
  MuonAlignment* inputAlign2a;

  TFile* _theFile;
  TTree* _alignTree;
  TFile* _inputRootFile1;
  TFile* _inputRootFile2;
  TTree* _inputTree1;
  TTree* _inputTree2;

  int _id, _level, _mid, _mlevel, _sublevel, _useDetId, _detDim;
  float _xVal, _yVal, _zVal, _rVal, _phiVal, _alphaVal, _betaVal, _gammaVal, _etaVal;
  float _dxVal, _dyVal, _dzVal, _drVal, _dphiVal, _dalphaVal;
  float _dbetaVal, _dgammaVal, _ldxVal, _ldyVal, _ldzVal;
  float _ldrVal, _ldphiVal;
  float _rotxVal, _rotyVal, _rotzVal;
  float _drotxVal, _drotyVal, _drotzVal;
  float _surWidth, _surLength;
  double _surRot[9];

  bool firstEvent_;
};

#endif
