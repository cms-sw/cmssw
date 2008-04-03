process Alignment =
{
  include "Alignment/HIPAlignmentAlgorithm/test/common.cff"

  source = EmptySource {}

  untracked PSet maxEvents = { untracked int32 input = 0 }

  replace HIPAlignmentAlgorithm.outpath = '<PATH>/main/' # must put backslash
  replace HIPAlignmentAlgorithm.collectorActive = true
  replace HIPAlignmentAlgorithm.collectorNJobs  = 0
  replace HIPAlignmentAlgorithm.collectorPath   = '<PATH>'
}
