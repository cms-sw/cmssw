#ifndef Alignment_OfflineValidation_TrackerGeometryCompare_h
#define Alignment_OfflineValidation_TrackerGeometryCompare_h

/** \class TrackerGeometryCompare
 *
 * Module that reads survey info from DB and prints them out.
 *
 * Usage:
 *   module comparator = TrackerGeometryCompare {
 *
 *   lots of stuff  
 *
 *   }
 *   path p = { comparator }
 *
 *
 *  $Date: 2010/01/04 18:24:37 $
 *  $Revision: 1.10 $
 *  \author Nhan Tran
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"

#include "Alignment/CommonAlignment/interface/AlignTools.h"

#include <algorithm>
#include "TTree.h"

class AlignTransform;

class TrackerGeometryCompare:
public edm::EDAnalyzer
{
public:
	typedef AlignTransform SurveyValue;
	typedef Alignments SurveyValues;
	typedef std::vector<Alignable*> Alignables;
		
  /// Do nothing. Required by framework.
  TrackerGeometryCompare(
		const edm::ParameterSet&
		);
	
  /// Read from DB and print survey info.
	virtual void beginJob();

	virtual void analyze(
		const edm::Event&,
		const edm::EventSetup&
		);
	
private:


	//parameters
	edm::ParameterSet m_params;
	std::vector<align::StructureType> theLevels;
	//std::vector<int> theSubDets;
	
	//compares two geometries
	void compareGeometries(Alignable* refAli, Alignable* curAli);
	//filling the ROOT file
	void fillTree(Alignable *refAli, AlgebraicVector diff);
	//for filling identifiers
	void fillIdentifiers( int subdetlevel, int rawid );
	//converts surveyRcd into alignmentRcd
	void surveyToTracker(AlignableTracker* ali, Alignments* alignVals, AlignmentErrors* alignErrors);
	//need for conversion for surveyToTracker
	void addSurveyInfo(Alignable* ali);
	//void createDBGeometry(const edm::EventSetup& iSetup);
	void createROOTGeometry(const edm::EventSetup& iSetup);
	
	// for common tracker system
	void setCommonTrackerSystem();
	void diffCommonTrackerSystem(Alignable* refAli, Alignable* curAli);
	bool passIdCut( uint32_t );
	
	AlignableTracker* referenceTracker;
	AlignableTracker* dummyTracker;
	AlignableTracker* currentTracker;

	unsigned int theSurveyIndex;
	const Alignments* theSurveyValues;
	const SurveyErrors* theSurveyErrors;
	
	// configurables
	std::string _inputFilename1;
	std::string _inputFilename2;
	std::string _inputTreename;
	bool _writeToDB; 
	std::string _weightBy;
	std::string _setCommonTrackerSystem;
	bool _detIdFlag;
	std::string _detIdFlagFile;
	bool _weightById;
	std::string _weightByIdFile;
	std::vector< unsigned int > _weightByIdVector;
	
	std::vector< uint32_t > _detIdFlagVector;
	align::StructureType _commonTrackerLevel;
	align::GlobalVector _TrackerCommonT;
	align::GlobalVector _TrackerCommonR;
	align::PositionType _TrackerCommonCM;
	
	//root configuration
	std::string _filename;
	TFile* _theFile;
	TTree* _alignTree;
	TFile* _inputRootFile1;
	TFile* _inputRootFile2;
	TTree* _inputTree1;
	TTree* _inputTree2;
	
	int _id, _level, _mid, _mlevel, _sublevel, _useDetId, _detDim;
	float _xVal, _yVal, _zVal, _rVal, _phiVal, _alphaVal, _betaVal, _gammaVal, _etaVal;
	// changes in global variables
	float _dxVal, _dyVal, _dzVal, _drVal, _dphiVal, _dalphaVal, _dbetaVal, _dgammaVal;
	// changes local variables: u, v, w, alpha, beta, gamma
	float _duVal, _dvVal, _dwVal, _daVal, _dbVal, _dgVal;
	float _surWidth, _surLength;
	uint32_t _identifiers[6];
	double _surRot[9];

  int m_ROWS_PER_ROC;
  int m_COLS_PER_ROC;
  int m_BIG_PIX_PER_ROC_X;
  int m_BIG_PIX_PER_ROC_Y;
  int m_ROCS_X;
  int m_ROCS_Y;
  bool m_upgradeGeometry;

	bool firstEvent_;
	
};




#endif
