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
 *  $Date: 2012/12/02 22:13:12 $
 *  $Revision: 1.14 $
 *  \author Nhan Tran
 *
 * ********
 * ******** Including surface deformations in the geometry comparison ******** 
 * ********
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"

#include "Alignment/CommonAlignment/interface/AlignTools.h"


//******** Single include for the TkMap *************
#include "CommonTools/TrackerMap/interface/TrackerMap.h" 
#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
//***************************************************

#include <algorithm>
#include <string>
#include "TTree.h"
#include "TH1D.h"

class AlignTransform;
class TrackerTopology;

class TrackerGeometryCompare: public edm::EDAnalyzer { 
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

	virtual void endJob();

	virtual void analyze(
		const edm::Event&,
		const edm::EventSetup&
		);
	
private:


	//parameters
	std::vector<align::StructureType> m_theLevels;
	//std::vector<int> theSubDets;
	
	//compare surface deformations
	void compareSurfaceDeformations(TTree* _inputTree11, TTree* _inputTree12); 
	//compares two geometries
	void compareGeometries(Alignable* refAli, Alignable* curAli, const TrackerTopology* tTopo, const edm::EventSetup& iSetup);
	//filling the ROOT file
	void fillTree(Alignable *refAli, const AlgebraicVector& diff, // typedef CLHEP::HepVector      AlgebraicVector; 
                      const TrackerTopology* tTopo, const edm::EventSetup& iSetup); 
	//for filling identifiers
	void fillIdentifiers( int subdetlevel, int rawid, const TrackerTopology* tTopo);
	//converts surveyRcd into alignmentRcd
	void surveyToTracker(AlignableTracker* ali, Alignments* alignVals, AlignmentErrorsExtended* alignErrors);
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
	std::string _moduleListName;
	std::string _inputFilename1;
	std::string _inputFilename2;
	std::string _inputTreenameAlign;
	std::string _inputTreenameDeform;
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
	
	std::ifstream _moduleListFile;
	std::vector< int > _moduleList;
	int _moduleInList;
	
	//root configuration
	std::string _filename;
	TFile* _theFile;
	TTree* _alignTree;
	TFile* _inputRootFile1;
	TFile* _inputRootFile2;
	TTree* _inputTree01;
	TTree* _inputTree02;
	TTree* _inputTree11;
	TTree* _inputTree12;
	
	/**\ Tree variables */
	int _id, _badModuleQuality, _inModuleList, _level, _mid, _mlevel, _sublevel, _useDetId, _detDim;
	float _xVal, _yVal, _zVal, _rVal, _etaVal, _phiVal, _alphaVal, _betaVal, _gammaVal;
	// changes in global variables
	float _dxVal, _dyVal, _dzVal, _drVal, _dphiVal, _dalphaVal, _dbetaVal, _dgammaVal;
	// changes local variables: u, v, w, alpha, beta, gamma
	float _duVal, _dvVal, _dwVal, _daVal, _dbVal, _dgVal;
	float _surWidth, _surLength;
	uint32_t _identifiers[6];
	double _surRot[9];
	int _type;
	double _surfDeform[13]; 

	int m_nBins ; 
	double m_rangeLow ;
	double m_rangeHigh ; 
	
	bool firstEvent_;

	std::vector<TrackerMap> m_vtkmap; 

	std::map<std::string,TH1D*> m_h1 ; 

	
};




#endif
