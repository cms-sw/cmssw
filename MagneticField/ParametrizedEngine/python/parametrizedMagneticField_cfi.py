import FWCore.ParameterSet.Config as cms

idealMagneticFieldRecordSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('IdealMagneticFieldRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    #    string version = "OAE_85l_030919"
    #    PSet parameters = {
    #	double b0 = 40.681
    #	double l  = 15.284
    #	double a  = 4.6430
    #    }
    #    string version = "OAE_1103l_071212"
    #    PSet parameters = {
    #      string BValue = "3_8T"
    #        string BValue = "4_0T"
    #    }
    #    string version = "MTCC2DPoly"
    #    PSet parameters = {
    #    }
    # Legacy implementation - to be removed!!!
    version = cms.string('OAE_85l_030919_t'),
    parameters = cms.PSet(
        paramFieldLength = cms.double(15.284),
        paramFieldRadius = cms.double(4.643)
    ),
    label = cms.untracked.string('')
)


