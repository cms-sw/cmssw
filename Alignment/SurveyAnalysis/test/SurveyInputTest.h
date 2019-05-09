#ifndef Alignment_SurveyAnalysis_SurveyInputTest_h
#define Alignment_SurveyAnalysis_SurveyInputTest_h

/** \class SurveyInputTest
 *
 *  Class to read survey raw measurements from a config file.
 *
 *  $Date: 2007/04/07 01:58:49 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SurveyInputTest : public SurveyInputBase {
public:
  SurveyInputTest(const edm::ParameterSet&);

  /// Read data from cfg file
  virtual void beginJob();

  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}

private:
  Alignable* create(const std::string& parName  // name of alignable
  );

  edm::ParameterSet theConfig;
};

#endif
