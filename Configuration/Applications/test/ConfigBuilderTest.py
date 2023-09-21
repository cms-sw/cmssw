import FWCore.ParameterSet.Config as cms
from Configuration.Applications.ConfigBuilder import ConfigBuilder, defaultOptions

import copy

def prepareDQMSequenceOrder():
    options = copy.deepcopy(defaultOptions)
    options.scenario = "Test"

    process = cms.Process("Test")
    process.a1 = cms.EDAnalyzer("A1")
    process.a2 = cms.EDAnalyzer("A2")
    process.a3 = cms.EDAnalyzer("A3")
    process.a4 = cms.EDAnalyzer("A4")
    process.seq1 = cms.Sequence(process.a1)
    process.seq2 = cms.Sequence(process.a2)
    process.seq3 = cms.Sequence(process.a3)
    process.seq4 = cms.Sequence(process.a4)
    process.ps1 = cms.Sequence()
    process.ps2 = cms.Sequence()
    process.ps3 = cms.Sequence()
    process.ps3 = cms.Sequence()
    process.ps4 = cms.Sequence()
    return (options, process)

if __name__=="__main__":
    import unittest

    class TestModuleCommand(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            None

        def testDQMSequenceOrder(self):
            def extract(process, count):
                if count == 0:
                    return []
                ret = list(process.dqmoffline_step.moduleNames())
                for i in range(1, count):
                    ret.extend(list(getattr(process, f"dqmoffline_{i}_step").moduleNames()))
                return ret

            # DQM sequences are unique
            (options, process) = prepareDQMSequenceOrder()
            order = [3, 1, 2]
            cb = ConfigBuilder(options, process)
            cb.prepare_DQM("+".join(f"seq{o}" for o in order))
            self.assertEqual([f"a{o}" for o in order], extract(process, len(order)))

            # Code in prepare_DQM() call assumes the 'sequenceList`
            # has at least as many elements as `postSequenceList`. We
            # can't fake 'postSequenceList' as it is also derived from
            # the prepare_DQM() argument, but duplicates are not
            # removed. The only handle we have (besides code changes,
            # that are beyond this bug fix), is to modify the autoDQM.
            from DQMOffline.Configuration.autoDQM import autoDQM
            autoDQM_orig = copy.deepcopy(autoDQM)
            autoDQM.clear()
            autoDQM["alias1"] = ["seq1", "ps1", "not needed"]
            autoDQM["alias2"] = ["seq2", "ps2", "not needed"]
            autoDQM["alias3"] = ["seq3", "ps3", "not needed"]
            # seq4 is used only to have the expanded and uniquified
            # 'sequenceList' to have at least as many elements as
            # 'postSequenceList'
            autoDQM["alias4"] = ["seq2+seq4", "ps4", "not needed"]

            order = [2, 1, 3]
            cb = ConfigBuilder(options, process)
            cb.prepare_DQM("+".join(f"@alias{o}" for o in order))
            self.assertEqual([f"a{o}" for o in order], extract(process, len(order)))

            cb = ConfigBuilder(options, process)
            order = [2, 1, 4, 3]
            cb.prepare_DQM("+".join(f"@alias{o}" for o in order))
            self.assertEqual([f"a{o}" for o in order], extract(process, len(order)))

            autoDQM.clear()
            autoDQM.update(autoDQM_orig)

    unittest.main()
