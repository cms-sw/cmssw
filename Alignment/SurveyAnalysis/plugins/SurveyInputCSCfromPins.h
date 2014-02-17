#ifndef Alignment_SurveyAnalysis_SurveyInputCSCfromPins_h
#define Alignment_SurveyAnalysis_SurveyInputCSCfromPins_h

/** \class SurveyInputCSCfromPins
 *
 *  Class to read ideal tracker from DB.
 *
 *  $Date: Fri Jun 29 09:20:52 CEST 2007
 *  $Revision: 1.3 $
 *  \author Dmitry Yakorev
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"

class SurveyInputCSCfromPins:
  public SurveyInputBase
{
public:
	
  SurveyInputCSCfromPins(const edm::ParameterSet&);
	
  /// Read ideal tracker geometry from DB
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:

  void orient(LocalVector LC1, LocalVector LC2, double a, double b, double &T, double &dx, double &dy, double &dz, double &PhX, double &PhZ);
  void errors(double a, double b, bool missing1, bool missing2, double &dx_dx, double &dy_dy, double &dz_dz, double &phix_phix, double &phiz_phiz, double &dy_phix);

  void fillAllRecords(Alignable *ali);

  std::string m_pinPositions;
  std::string m_rootFile;
  bool m_verbose;
  double m_errorX, m_errorY, m_errorZ;
  double m_missingErrorTranslation, m_missingErrorAngle;
  double m_stationErrorX, m_stationErrorY, m_stationErrorZ, m_stationErrorPhiX, m_stationErrorPhiY, m_stationErrorPhiZ;
};

#endif
