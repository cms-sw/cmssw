import FWCore.ParameterSet.Config as cms

# trigger mask: disable L1A result if the corresponding technical trigger
# has bit value equal to 1 for the respective DAQ partition



l1GtTriggerMaskVetoTechTrig = cms.ESProducer("L1GtTriggerMaskVetoTechTrigTrivialProducer",
    TriggerMask = cms.vuint32(
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00,
           0x00
           )
)


# foo bar baz
# xvT4CTiEc1wzv
# TWxhELS25IJYi
