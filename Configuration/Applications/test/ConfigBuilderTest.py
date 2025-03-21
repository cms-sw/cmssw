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

class OutStats(object):
    def __init__(self, name, type_, tier, fileName, commands):
        self.name = name
        self.type_ = type_
        self.tier = tier
        self.fileName = fileName
        self.commands = commands
def _test_addOutput(tester, options, process, outStats, modifier = None):
    cb = ConfigBuilder(options, process)
    if modifier:
        modifier(cb)
    cb.addOutput()
    tester.assertEqual(len(process.outputModules_()), len(outStats))
    outs = []
#    print(process.outputModules_())
    for stat in outStats:
        tester.assertTrue( stat.name in process.outputModules_() )
        r = process.outputModules_()[stat.name]
        tester.assertEqual(r.type_(), stat.type_)
        tester.assertEqual(r.dataset.dataTier.value(), stat.tier)
        tester.assertEqual(r.fileName.value(), stat.fileName)
        tester.assertEqual(r.outputCommands, stat.commands)
        outs.append(r)
    return outs

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

        def testOutputFormatWithEventContent(self):
            #RECO
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.RECOEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "RECO"
            options.eventcontent="RECO"
            _test_addOutput(self, options, process, [OutStats('RECOoutput','PoolOutputModule','RECO','output.root',outputCommands_)])
            #AOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.AODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "AOD"
            options.eventcontent="AOD"
            _test_addOutput(self, options, process, [OutStats('AODoutput','PoolOutputModule','AOD','output.root',outputCommands_)])
            #MINIAOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "MINIAOD"
            options.eventcontent="MINIAOD"
            _test_addOutput(self, options, process, [OutStats('MINIAODoutput','PoolOutputModule','MINIAOD','output.root',outputCommands_)])
            #MINIAOD w/ RNTuple
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "MINIAOD"
            options.eventcontent="MINIAOD"
            options.rntuple_out = True
            _test_addOutput(self, options, process, [OutStats('MINIAODoutput','RNTupleOutputModule','MINIAOD','output.rntpl',outputCommands_)])
            #MINIAOD1 [NOTE notiation not restricted to MINIAOD]
            #NOT SUPPORTED BY outputDefinition
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "MINIAOD"
            options.eventcontent="MINIAOD1"
            _test_addOutput(self, options, process, [OutStats('MINIAOD1output','PoolOutputModule','MINIAOD','output.root',outputCommands_)])
            #DQMIO
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.DQMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "DQMIO"
            options.eventcontent="DQM"
            _test_addOutput(self, options, process, [OutStats('DQMoutput','DQMRootOutputModule','DQMIO','output.root',outputCommands_)])
            #DQMIO & rntuple (will not change)
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.DQMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "DQMIO"
            options.eventcontent="DQM"
            options.rntuple_out = True
            _test_addOutput(self, options, process, [OutStats('DQMoutput','DQMRootOutputModule','DQMIO','output.root',outputCommands_)])
            #DQMIO&DQMIO
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.DQMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "DQMIO"
            options.eventcontent="DQMIO"
            _test_addOutput(self, options, process, [OutStats('DQMoutput','DQMRootOutputModule','DQMIO','output.root',outputCommands_)])
            #DQMIO&DQMIO & rntuple (will not change)
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.DQMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "DQMIO"
            options.eventcontent="DQMIO"
            options.rntuple_out = True
            _test_addOutput(self, options, process, [OutStats('DQMoutput','DQMRootOutputModule','DQMIO','output.root',outputCommands_)])
            #DQM, not DQMIO (decided by datatier)
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.DQMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "DQM"
            options.eventcontent="DQM"
            _test_addOutput(self, options, process, [OutStats('DQMoutput','PoolOutputModule','DQM','output.root',outputCommands_)])
            #NANOAOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.NANOAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "NANOAOD"
            options.eventcontent="NANOAOD"
            _test_addOutput(self, options, process, [OutStats('NANOAODoutput','NanoAODOutputModule','NANOAOD','output.root',outputCommands_)])
            #NANOEDMAOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.NANOAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "NANOAOD"
            options.eventcontent="NANOEDMAOD"
            _test_addOutput(self, options, process, [OutStats('NANOEDMAODoutput', 'PoolOutputModule', 'NANOAOD', 'output.root', outputCommands_)])
            #NANOAOD & rntuple (no change)
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.NANOAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "NANOAOD"
            options.eventcontent="NANOAOD"
            options.rntuple_out = True
            _test_addOutput(self, options, process, [OutStats('NANOAODoutput','NanoAODOutputModule','NANOAOD','output.root',outputCommands_)])
            #NANOEDMAOD & rntuple
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.NANOAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "NANOAOD"
            options.eventcontent="NANOEDMAOD"
            options.rntuple_out = True
            _test_addOutput(self, options, process, [OutStats('NANOEDMAODoutput', 'RNTupleOutputModule', 'NANOAOD', 'output.rntpl', outputCommands_)])
            #ALCARECO empty
            process = cms.Process("TEST")
            options.scenario = "TEST"
            options.datatier= "ALCARECO"
            options.eventcontent="ALCARECO"
            _test_addOutput(self, options, process, [])
            #ALCARECO present
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.ALCARECOEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.datatier= "ALCARECO"
            options.eventcontent="ALCARECO"
            options.step = 'ALCAPRODUCER'
            outs = _test_addOutput(self, options, process, [OutStats('ALCARECOoutput', 'PoolOutputModule', 'ALCARECO', 'output.root', outputCommands_)])
            self.assertEqual(outs[0].dataset.filterName, cms.untracked.string('StreamALCACombined'))
            #AOD+MINIAOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.AODEventContent = cms.PSet( outputCommands = outputCommands_)
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.fileout = 'stepN.root'
            options.scenario = "TEST"
            options.datatier= "AOD,MINIAOD"
            options.eventcontent="AOD,MINIAOD"
            _test_addOutput(self, options, process, [OutStats('AODoutput','PoolOutputModule','AOD','stepN.root',outputCommands_), OutStats('MINIAODoutput','PoolOutputModule','MINIAOD','stepN_inMINIAOD.root',outputCommands_)])
            #MINIAOD & generation_step
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            process.generation_step = cms.Path()
            options = copy.deepcopy(defaultOptions)
            options.fileout = 'stepN.root'
            options.scenario = "TEST"
            options.datatier= "MINIAOD"
            options.eventcontent="MINIAOD"
            out = _test_addOutput(self, options, process, [OutStats('MINIAODoutput','PoolOutputModule','MINIAOD','stepN.root',outputCommands_)])
            self.assertEqual(out[0].SelectEvents.SelectEvents, cms.vstring('generation_step'))
            #MINIAOD & filtering_step
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            process.filtering_step = cms.Path()
            options = copy.deepcopy(defaultOptions)
            options.fileout = 'stepN.root'
            options.scenario = "TEST"
            options.datatier= "MINIAOD"
            options.eventcontent="MINIAOD"
            out = _test_addOutput(self, options, process, [OutStats('MINIAODoutput','PoolOutputModule','MINIAOD','stepN.root',outputCommands_)])
            self.assertEqual(out[0].SelectEvents.SelectEvents, cms.vstring('filtering_step'))
            #LHE & generation_step
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.LHEEventContent = cms.PSet( outputCommands = outputCommands_)
            process.generation_step = cms.Path()
            options = copy.deepcopy(defaultOptions)
            options.fileout = 'stepN.root'
            options.scenario = "TEST"
            options.datatier= "LHE"
            options.eventcontent="LHE"
            out = _test_addOutput(self, options, process, [OutStats('LHEoutput','PoolOutputModule','LHE','stepN.root',outputCommands_)])
            self.assertFalse(hasattr(out[0],"SelectEvents"))

        def testOutputFormatWith_outputDefinition(self):
            #RECO
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.RECOEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"            
            options.outputDefinition = str([dict(tier='RECO'),])
            _test_addOutput(self, options, process, [OutStats('RECOoutput','PoolOutputModule','RECO','output.root',outputCommands_)])
            #RECO & moduleLabel
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.RECOEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"            
            options.outputDefinition = str([dict(tier='RECO',moduleLabel='RECOoutputOther'),])
            _test_addOutput(self, options, process, [OutStats('RECOoutputOther','PoolOutputModule','RECO','output.root',outputCommands_)])
            #RECO & fileName
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.RECOEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"            
            options.outputDefinition = str([dict(tier='RECO',fileName='other.root'),])
            _test_addOutput(self, options, process, [OutStats('RECOoutput','PoolOutputModule','RECO','other.root',outputCommands_)])
            #RECO & eventContent
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.RECOSIMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"            
            options.outputDefinition = str([dict(tier='RECO',eventContent='RECOSIM'),])
            _test_addOutput(self, options, process, [OutStats('SIMRECOoutput','PoolOutputModule','RECO','output.root',outputCommands_)])
            #RECO & selectEvents
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.RECOEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"            
            options.outputDefinition = str([dict(tier='RECO',selectEvents='select'),])
            out = _test_addOutput(self, options, process, [OutStats('RECOoutput','PoolOutputModule','RECO','output.root',outputCommands_)])
            self.assertEqual(out[0].SelectEvents.SelectEvents, cms.vstring('select'))
            #RECO & outputCommands
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.RECOEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"            
            options.outputDefinition = str([dict(tier='RECO',outputCommands=cms.untracked.vstring('keep bar')),])
            fullOutputCommands = cms.untracked.vstring('drop *', 'keep foo', 'keep bar')
            _test_addOutput(self, options, process, [OutStats('RECOoutput','PoolOutputModule','RECO','output.root',fullOutputCommands)])
            #AOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.AODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='AOD'),])
            _test_addOutput(self, options, process, [OutStats('AODoutput','PoolOutputModule','AOD','output.root',outputCommands_)])
            #MINIAOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='MINIAOD'),])
            _test_addOutput(self, options, process, [OutStats('MINIAODoutput','PoolOutputModule','MINIAOD','output.root',outputCommands_)])
            #DQMIO&DQM
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.DQMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='DQMIO', eventContent='DQM'),])
            #NOTE: THE MODULE LABEL IS DIFFERENT FROM THE OTHER CASE
            #_test_addOutput(self, options, process, [OutStats('DQMIOoutput','DQMRootOutputModule','DQMIO','output.root',outputCommands_)])
            _test_addOutput(self, options, process, [OutStats('DQMDQMIOoutput','DQMRootOutputModule','DQMIO','output.root',outputCommands_)])
            #DQMIO&DQMIO
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.DQMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='DQMIO', eventContent='DQMIO'),])
            _test_addOutput(self, options, process, [OutStats('DQMIOoutput','DQMRootOutputModule','DQMIO','output.root',outputCommands_)])
            #DQM, not DQMIO (decided by datatier)
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.DQMEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='DQM', eventContent='DQM'),])
            _test_addOutput(self, options, process, [OutStats('DQMoutput','PoolOutputModule','DQM','output.root',outputCommands_)])
            #NANOAOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.NANOAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='NANOAOD', eventContent='NANOAOD'),])
            #NOTE: Does not use NanoAODOuputModule (this code path is not capable of making that type)
            #_test_addOutput(self, options, process, [OutStats('NANOAODoutput','NanoAODOutputModule','NANOAOD','output.root',outputCommands_)])
            _test_addOutput(self, options, process, [OutStats('NANOAODoutput','PoolOutputModule','NANOAOD','output.root',outputCommands_)])
            #NANOEDMAOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            #USES A DIFFERENT EVENT CONTENT (which we do not define)
            #process.NANOAODEventContent = cms.PSet( outputCommands = outputCommands_)
            process.NANOEDMAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='NANOAOD', eventContent='NANOEDMAOD'),])
            #GENERATES A DIFFERENT NAME
            #_test_addOutput(self, options, process, [OutStats('NANOEDMAODoutput', 'PoolOutputModule', 'NANOAOD', 'output.root', outputCommands_)])
            _test_addOutput(self, options, process, [OutStats('NANOEDMAODNANOAODoutput', 'PoolOutputModule', 'NANOAOD', 'output.root', outputCommands_)])
            #ALCARECO empty
            #THIS DOES NOT STOP THE ADDITION LIKE THE OTHER BRANCH
            process = cms.Process("TEST")
            options.scenario = "TEST"
            options.datatier= "ALCARECO"
            options.eventcontent="ALCARECO"
            options.outputDefinition = str([dict(tier='ALCARECO', eventContent='ALCARECO'),])
            #OTHER BRANCH DOES NOT NEED THE FOLLOWING
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.ALCARECOEventContent = cms.PSet( outputCommands = outputCommands_)
            #_test_addOutput(self, options, process, [])
            outs = _test_addOutput(self, options, process, [OutStats('ALCARECOoutput', 'PoolOutputModule', 'ALCARECO', 'output.root', outputCommands_)], lambda cb: setattr(cb, 'AlCaPaths',[]))
            #ALCARECO present
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.ALCARECOEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='ALCARECO', eventContent='ALCARECO'),])
            options.step = 'ALCAPRODUCER'
            outs = _test_addOutput(self, options, process, [OutStats('ALCARECOoutput', 'PoolOutputModule', 'ALCARECO', 'output.root', outputCommands_)], lambda cb: setattr(cb, 'AlCaPaths', []))
            self.assertEqual(outs[0].dataset.filterName, cms.untracked.string('StreamALCACombined'))
            #AOD+MINIAOD
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.AODEventContent = cms.PSet( outputCommands = outputCommands_)
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            options = copy.deepcopy(defaultOptions)
            options.fileout = 'stepN.root'
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='AOD'),dict(tier='MINIAOD')])
            _test_addOutput(self, options, process, [OutStats('AODoutput','PoolOutputModule','AOD','stepN.root',outputCommands_), OutStats('MINIAODoutput','PoolOutputModule','MINIAOD','stepN_inMINIAOD.root',outputCommands_)])
            #MINIAOD & generation_step
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            process.generation_step = cms.Path()
            options = copy.deepcopy(defaultOptions)
            options.fileout = 'stepN.root'
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='MINIAOD'),])
            out = _test_addOutput(self, options, process, [OutStats('MINIAODoutput','PoolOutputModule','MINIAOD','stepN.root',outputCommands_)])
            self.assertEqual(out[0].SelectEvents.SelectEvents, cms.vstring('generation_step'))
            #MINIAOD & filtering_step
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.MINIAODEventContent = cms.PSet( outputCommands = outputCommands_)
            process.filtering_step = cms.Path()
            options = copy.deepcopy(defaultOptions)
            options.fileout = 'stepN.root'
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='MINIAOD'),])
            out = _test_addOutput(self, options, process, [OutStats('MINIAODoutput','PoolOutputModule','MINIAOD','stepN.root',outputCommands_)])
            self.assertEqual(out[0].SelectEvents.SelectEvents, cms.vstring('filtering_step'))
            #LHE & generation_step
            process = cms.Process("TEST")
            outputCommands_ = cms.untracked.vstring('drop *', 'keep foo')
            process.LHEEventContent = cms.PSet( outputCommands = outputCommands_)
            process.generation_step = cms.Path()
            options = copy.deepcopy(defaultOptions)
            options.fileout = 'stepN.root'
            options.scenario = "TEST"
            options.outputDefinition = str([dict(tier='LHE'),])
            out = _test_addOutput(self, options, process, [OutStats('LHEoutput','PoolOutputModule','LHE','stepN.root',outputCommands_)])
            self.assertFalse(hasattr(out[0],"SelectEvents"))

    unittest.main()

