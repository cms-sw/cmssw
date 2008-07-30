process Alignment =
{
  include "Alignment/HIPAlignmentAlgorithm/test/common.cff"
  include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"
  include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"

  source = EmptySource {}

  untracked PSet maxEvents = { untracked int32 input = 0 }

  replace HIPAlignmentAlgorithm.outpath = '<PATH>/main/' # must put backslash
  replace HIPAlignmentAlgorithm.collectorActive = true
  replace HIPAlignmentAlgorithm.collectorNJobs  = <JOBS>
  replace HIPAlignmentAlgorithm.collectorPath   = '<PATH>'
  replace HIPAlignmentAlgorithm.minimumNumberOfHits = 0
  replace HIPAlignmentAlgorithm.maxRelParameterError = 1e99
/*
#only if want to use survey

  replace HIPAlignmentAlgorithm.surveyResiduals = {'Det', 'Pixel',
    'TPBLadder', 'TPBLayer', 'TPBHalfBarrel', 'TPBBarrel',
    'TPEPanel', 'TPEBlade', 'TPEHalfDisk', 'TPEHalfCylinder', 'TPEEndcap'}

  es_source survey = PoolDBESSource
  {
    using CondDBSetup

    string connect  = "sqlite_file:/afs/cern.ch/user/n/ntran/public/HIPAlignment/measurementSurvey_StripsIdeal.db"
    string timetype = "runnumber"

    VPSet toGet =
    {
      { string record = "TrackerSurveyRcd"      string tag = "valueTag" },
      { string record = "TrackerSurveyErrorRcd" string tag = "errorTag" }
    }
  }
  replace survey.catalog = "file:/afs/cern.ch/user/n/ntran/public/HIPAlignment/measurementSurvey_StripsIdeal.xml"	
  
  replace AlignmentProducer.useSurvey = true

  replace AlignmentProducer.monitorConfig =
  {
    untracked vstring monitors = {'AlignmentMonitorSurvey'}

    untracked PSet AlignmentMonitorSurvey =
    {
      string outpath = "<PATH>/main/"
      string outfile = "histograms.root"

      bool collectorActive = false
      int32 collectorNJobs = 0
      string collectorPath = "./"
    }
  }
*/
}
