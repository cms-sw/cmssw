#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/DataRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/DataRecord/interface/TrackerSurveyErrorRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/SurveyAnalysis/test/SurveyDBReader.h"

void SurveyDBReader::analyze(const edm::Event&,
			     const edm::EventSetup& setup)
{
  typedef AlignTransform SurveyValue;
  typedef Alignments     SurveyValues;

  edm::ESHandle<SurveyValues> valuesHandle;
  edm::ESHandle<SurveyErrors> errorsHandle;

  setup.get<TrackerSurveyRcd>().get(valuesHandle);
  setup.get<TrackerSurveyErrorRcd>().get(errorsHandle);

  const std::vector<SurveyValue>& values = valuesHandle->m_align;
  const std::vector<SurveyError>& errors = errorsHandle->m_surveyErrors;

  unsigned int size = values.size();

  if ( errors.size() != size )
  {
    edm::LogError("SurveyDBReader")
      << "Value and error records have different sizes. Abort!";

    return;
  }

  for (unsigned int i = 0; i < size; ++i)
  {
    const SurveyValue& value = values[i];
    const SurveyError& error = errors[i];

    edm::LogInfo("SurveyDBReader")
      << "Type " << static_cast<unsigned int>( error.structureType() )
      << " raw id " << error.rawId()
      << " pos " << value.translation()
      << " rot " << value.rotation()
      << " errors\n" << error.matrix();
  }

  edm::LogInfo("SurveyDBReader")
    << "Number of alignables read " << size << std::endl;
}
