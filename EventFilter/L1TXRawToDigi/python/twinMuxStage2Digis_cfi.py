import FWCore.ParameterSet.Config as cms

twinMuxStage2Digis = cms.EDProducer("L1TTwinMuxRawToDigi",
                               DTTM7_FED_Source = cms.InputTag("rawDataCollector"),
                               feds     = cms.untracked.vint32( 1395,           1391,           1390,           1393,           1394           ),
                               wheels   = cms.untracked.vint32( -2,             -1,             0,              +1,             +2             ),
                               # Sector = '1' to '12' in HEX ('1' -> 'C'); 'F' if the AmcId is not associated to any Sector
                               # AmcId                         (  123456789...)
                               # Mapping                       (  FFFFF3FFF...)
                               # Sector                        (  -----3---...)
                               amcsecmap= cms.untracked.vint64( 0x123456789ABC, 0x123456789ABC, 0x123456789ABC, 0x123456789ABC, 0x123456789ABC ),
                               debug    = cms.untracked.bool(False),
                               )
