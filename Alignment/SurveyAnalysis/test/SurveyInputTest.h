#ifndef Alignment_SurveyAnalysis_SurveyInputTest_h
#define Alignment_SurveyAnalysis_SurveyInputTest_h

/** \class SurveyInputTest
 *
 *  Class to read survey raw measurements from a config file.
 *
 *  $Date: 2010/01/07 14:36:23 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SurveyInputTest:
  public SurveyInputBase
{
  public:

  SurveyInputTest(
		  const edm::ParameterSet&
		  );

  /// Read data from cfg file
  virtual void beginJob();
  
  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       ) {}

  private:

  Alignable* create(
		    const std::string& parName // name of alignable
		    );

  edm::ParameterSet theConfig;
};

#endif
