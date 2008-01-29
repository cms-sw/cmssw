process Alignment =
{
  include "Alignment/CommonAlignmentProducer/data/AlignmentProducerCSA07.cff"

  service = MessageLogger
  {
    untracked vstring destinations = {'cout'}

    untracked PSet cout = { untracked string threshold = 'WARNING' }
  }

  source = EmptySource {}

  untracked PSet maxEvents = { untracked int32 input = 0 }

  replace HIPAlignmentAlgorithm.outpath = '<PATH>/main/' # must put backslash
  replace HIPAlignmentAlgorithm.collectorActive = true
  replace HIPAlignmentAlgorithm.collectorNJobs  = <JOBS>
  replace HIPAlignmentAlgorithm.collectorPath   = '<PATH>'

  replace AlignmentProducer.maxLoops = 1
  replace AlignmentProducer.algoConfig = { using HIPAlignmentAlgorithm }
  replace AlignmentProducer.monitorConfig.monitors = {}

/* Not working in collector mode
  replace AlignmentProducer.monitorConfig.AlignmentMonitorGeneric =
  {
    string outpath = '<PATH>/main/'
    string outfile = 'histograms.root'
    bool collectorActive = true
    int32 collectorNJobs = <JOBS>
    string collectorPath = '<PATH>'
  }
*/
}
