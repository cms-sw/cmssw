process Alignment =
{
  include "../home<PATH>/common.cff"

  source = EmptySource {}

  untracked PSet maxEvents = { untracked int32 input = 1 }

  replace HIPAlignmentAlgorithm.outpath = "<PATH>/main/" # must put backslash
  replace HIPAlignmentAlgorithm.collectorActive = true
  replace HIPAlignmentAlgorithm.collectorNJobs  = 0
  replace HIPAlignmentAlgorithm.collectorPath   = "<PATH>"
}
