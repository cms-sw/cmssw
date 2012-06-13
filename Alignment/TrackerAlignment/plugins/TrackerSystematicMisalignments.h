#ifndef Alignment_TrackerAlignment_TrackerSystematicMisalignments_h
#define Alignment_TrackerAlignment_TrackerSystematicMisalignments_h

/** \class TrackerSystematicMisalignments
 *
 *  Class to misaligned tracker from DB.
 *
 *  $Date: 2010/06/14 14:45:15 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */
// user include files

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


class AlignableSurface;
class Alignments;


class TrackerSystematicMisalignments:
public edm::EDAnalyzer
{
public:
	
	TrackerSystematicMisalignments(
								   const edm::ParameterSet&
								   );
	
	/// Read ideal tracker geometry from DB
	virtual void beginJob();
	
	virtual void analyze(const edm::Event&, const edm::EventSetup&);
	
private:
	
	void applySystematicMisalignment( Alignable* ); 
	//align::GlobalVector findSystematicMis( align::PositionType );
	align::GlobalVector findSystematicMis( align::PositionType, const bool blindToZ, const bool blindToR );
	
	AlignableTracker* theAlignableTracker;
	
	
	
	// configurables needed for the systematic misalignment
	bool m_fromDBGeom;
	
	double m_radialEpsilon;
	double m_telescopeEpsilon;
	double m_layerRotEpsilon;
	double m_bowingEpsilon;
	double m_zExpEpsilon;
	double m_twistEpsilon;
	double m_ellipticalEpsilon;
	double m_skewEpsilon;
	double m_saggitaEpsilon;

	// flag to steer suppression of blind movements
	bool suppressBlindMvmts;

	// flag for old z behaviour, version <= 1.5
	bool oldMinusZconvention;

  int m_ROWS_PER_ROC;
  int m_COLS_PER_ROC;
  int m_BIG_PIX_PER_ROC_X;
  int m_BIG_PIX_PER_ROC_Y;
  int m_ROCS_X;
  int m_ROCS_Y;
  bool m_upgradeGeometry;
};

#endif
