import FWCore.ParameterSet.Config as cms

#
# Once the FakeAlignmentProducer is an ESSource and not only an ESProducer,
# all the following should in future be replaced by a simple
# include "Alignment/CommonAlignmentProducer/data/fakeAlignmentProducer.cff" # or cfi?
#
# Producer module
# ===============
# Dummy alignment producer (creating empty Aligments and AlignmentErrors)
from Alignment.CommonAlignmentProducer.fakeAlignmentProducer_cfi import *
# Sources that actually put the records into ESetup
# (Here with unlimited IOV, taken from producer above.)
# =====================================================
# global position
from Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff import *
# DT with errors
fakeDTAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('DTAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeDTAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('DTAlignmentErrorRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# CSC with errors
fakeCSCAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('CSCAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeCSCAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('CSCAlignmentErrorRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


