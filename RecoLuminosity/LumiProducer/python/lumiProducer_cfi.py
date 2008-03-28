import FWCore.ParameterSet.Config as cms

# In any real application the values of the parameters will
# need to be respecified completely for every job.  The values
# here are used in the unit test for the producer module.
# The numbers below have no meaning and the vectors contain
# only a few elements for test purposes.  In a real application
# there would be an element per bunch crossing in each vector,
# which would mean all the vectors would have thousands of elements.
lumiProducer = cms.EDProducer("LumiProducer",
    # The name is the letters "LS" followed by the actual number of
    # the luminosity section.  There is only one of these here, but
    # just add more parameter sets to extend this to handle the case
    # of multiple luminosity sections in the job.
    LS1 = cms.untracked.PSet(
        hltratecounter = cms.untracked.vint32(30, 31, 32, 33, 34),
        lumietsum = cms.untracked.vdouble(100.0, 101.0, 102.0, 103.0, 104.0),
        lumisecqual = cms.untracked.int32(3),
        l1scaler = cms.untracked.vint32(20, 21, 22, 23, 24),
        deadfrac = cms.untracked.double(0.05),
        lumietsumqual = cms.untracked.vint32(300, 301, 302, 303, 304),
        hltinput = cms.untracked.vint32(50, 51, 52, 53, 54),
        lumiocc = cms.untracked.vdouble(400.0, 401.0, 402.0, 403.0, 404.0),
        lumioccerr = cms.untracked.vdouble(500.0, 501.0, 502.0, 503.0, 504.0),
        hltscaler = cms.untracked.vint32(40, 41, 42, 43, 44),
        avginsdellumi = cms.untracked.double(1.0),
        lumietsumerr = cms.untracked.vdouble(200.0, 201.0, 202.0, 203.0, 204.0),
        lsnumber = cms.untracked.int32(5),
        l1ratecounter = cms.untracked.vint32(10, 11, 12, 13, 14),
        avginsdellumierr = cms.untracked.double(2.0),
        lumioccqual = cms.untracked.vint32(600, 601, 602, 603, 604)
    )
)


