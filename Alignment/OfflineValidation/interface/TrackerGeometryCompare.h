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
 *  $Date: 2007/05/03 19:20:00 $
 *  $Revision: 1.1 $
 *  \author Nhan Tran
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

class TrackerGeometryCompare:
public edm::EDAnalyzer
{
public:
	
  /// Do nothing. Required by framework.
  TrackerGeometryCompare(
		const edm::ParameterSet&
		);
	
  /// Read from DB and print survey info.
	virtual void beginJob(
		const edm::EventSetup&
		);

	virtual void analyze(
		const edm::Event&,
		const edm::EventSetup&
		);
	
private:


	//parameters
	edm::ParameterSet m_params;
	std::vector<AlignableObjectId::AlignableObjectIdType> theLevels;
	std::vector<int> theSubDets;
	
	
	void compareGeometries(Alignable* refAli, Alignable* curAli);
	void fillTree(Alignable *refAli, AlgebraicVector diff);
	
	AlignableTracker* referenceTracker;
	AlignableTracker* currentTracker;

	std::string _inputType;
	
	//root configuration
	std::string _filename;
	TFile* _theFile;
	TTree* _alignTree;
	int _id, _level, _mid, _mlevel, _sublevel;
	float _xVal, _yVal, _zVal, _rVal, _phiVal, _alphaVal, _betaVal, _gammaVal;
	float _dxVal, _dyVal, _dzVal, _drVal, _dphiVal, _dalphaVal, _dbetaVal, _dgammaVal;
	

	
};




#endif
