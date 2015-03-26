process Alignment =
{
  include "../home<PATH>/common.cff"

  source = EmptySource {}

  untracked PSet maxEvents = { untracked int32 input = 1 }

  replace HIPAlignmentAlgorithm.outpath = "<PATH>/main/" # must put backslash
  replace HIPAlignmentAlgorithm.collectorActive = true
  replace HIPAlignmentAlgorithm.collectorNJobs  = <JOBS>
  replace HIPAlignmentAlgorithm.collectorPath   = "<PATH>"
  replace HIPAlignmentAlgorithm.minimumNumberOfHits = 0
  replace HIPAlignmentAlgorithm.maxRelParameterError = 1e99

  replace HIPAlignmentAlgorithm.surveyResiduals = {"Det"}
  replace AlignmentProducer.useSurvey = true

  es_source survey = PoolDBESSource
  {
    using CondDBSetup

    string connect  = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/surveyObjects/measurementSurvey_207.db"
    string timetype = "runnumber"

    VPSet toGet =
    {
      { string record = "TrackerSurveyRcd"      string tag = "valueTag" },
      { string record = "TrackerSurveyErrorExtendedRcd" string tag = "errorTag" }
    }
  }
}
