import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool( True ),
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source( "EmptySource" )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10 )
)

# producer
process.two = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(2), iterations=cms.uint32(5*1000))

# producer
process.four = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(4), iterations=cms.uint32(10*1000))

# producer
process.ten = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(10), iterations=cms.uint32(2*1000))

process.adder = cms.EDProducer("AddIntsProducer", labels = cms.vstring('two','ten'))

subprocess = cms.Process("SUB")
process.addSubProcess( cms.SubProcess(
    process = subprocess, 
    SelectEvents = cms.untracked.PSet(), 
    outputCommands = cms.untracked.vstring()
) )

subprocess.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))
# module, reads products from 'adder' in the parent process
subprocess.final = cms.EDProducer("AddIntsProducer", labels = cms.vstring('adder'))

subprocess.subpath = cms.Path( subprocess.final )
