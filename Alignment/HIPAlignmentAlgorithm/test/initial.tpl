process Alignment =
{
  include "Alignment/HIPAlignmentAlgorithm/test/common.cff"
  include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"
  include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"

  source = EmptySource {}

  untracked PSet maxEvents = { untracked int32 input = 1 }

  replace HIPAlignmentAlgorithm.outpath = '<PATH>/main/' # must put backslash
  replace HIPAlignmentAlgorithm.collectorActive = true
  replace HIPAlignmentAlgorithm.collectorNJobs  = 0
  replace HIPAlignmentAlgorithm.collectorPath   = '<PATH>'
}
