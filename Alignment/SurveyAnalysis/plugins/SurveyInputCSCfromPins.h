#ifndef Alignment_SurveyAnalysis_SurveyInputCSCfromPins_h
#define Alignment_SurveyAnalysis_SurveyInputCSCfromPins_h

/** \class SurveyInputCSCfromPins
 *
 *  Class to read ideal tracker from DB.
 *
 *  $Date: Fri Jun 29 09:20:52 CEST 2007
 *  $Revision: 1.1 $
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
  virtual void beginJob(const edm::EventSetup& iSetup);
	
private:

void fillAllRecords(Alignable *ali);

  std::string m_pinPositions;
  std::string m_rootFile;
  bool m_verbose;
};

#endif
