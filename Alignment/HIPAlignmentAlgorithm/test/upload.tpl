process Alignment =
{
  include "../home<PATH>/common.cff"

  source = EmptySource {}

  untracked PSet maxEvents = { untracked int32 input = 1 }

  replace HIPAlignmentAlgorithm.outpath = "<PATH>/"

  replace AlignmentProducer.saveToDB = true

  service = PoolDBOutputService
  {
    using CondDBSetup

    string connect  = "sqlite_file:<PATH>/alignments.db"
    untracked string timetype = "runnumber"

    VPSet toPut =
    {
      { string record = "TrackerAlignmentRcd"      string tag = "Alignments" },
      { string record = "TrackerAlignmentErrorExtendedRcd" string tag = "AlignmentErrorsExtended" }
    }
  }
}
