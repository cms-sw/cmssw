#
# cfi for various mappings of the L1 GT boards
#

import FWCore.ParameterSet.Config as cms

l1GtBoardMaps = cms.ESProducer("L1GtBoardMapsTrivialProducer",

    # list of all boards in the L1 GT system
    # multiple boards must appear as many time as they exist in the system
    BoardList = cms.vstring(
        'GTFE', 
        'FDL', 
        'PSB', 'PSB', 'PSB', 'PSB', 'PSB', 'PSB', 'PSB', 
        'GMT', 
        'TCS', 
        'TIM'),

    # board index - it starts with 0  
    BoardIndex = cms.vint32(
        0, 
        0, 
        0, 1, 2, 3, 4, 5, 6,
        0, 
        0, 
        0),

    # L1 GT DAQ record map
    # boards not in the record have negative index
    BoardPositionDaqRecord = cms.vint32(
         1, 
         2, 
         3, 4, 5, 6, 7, 8, 9, 
        10, 
        -1, 
        -1),

    # L1 GT EVM record map
    # boards not in the record have negative index
    BoardPositionEvmRecord = cms.vint32(
         1, 
         3, 
        -1, -1, -1, -1, -1, -1, -1,
        -1, 
         2, 
        -1),

    # L1 GT "active boards" map for DAQ record
    # boards not in the record have negative index
    ActiveBoardsDaqRecord = cms.vint32(
        -1, 
         0, 
         1, 2, 3, 4, 5, 6, 7, 
         8, 
        -1, 
        -1),
        
    # L1 GT "active boards" map for EVM record
    # boards not in the record have negative index
    ActiveBoardsEvmRecord = cms.vint32(
        -1, 
         1, 
        -1, -1, -1, -1, -1, -1, -1, 
        -1, 
         0,
        -1),

    # L1 GT board - slot map
    # boards not in the map have negative index
    BoardSlotMap = cms.vint32(
        17, 
        10, 
         9, 13, 14, 15, 19, 20, 21, 
        18, 
         7, 
        16),
        
    # L1 GT board name in hw record map
    # boards not in the map have negative index
      BoardHexNameMap = cms.vint32(
        0x00,
        0xfd, 
        0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb,
        0xdd,
        0xcc,
        0xad),
    
    # L1 quadruplet (4x16 bits)(cable) to PSB input map
    # see L1GlobalTriggerPSB::fillPsbBlock
    # summary
    #CableList = cms.vstring( N/A,        N/A,         N/A,      TTr_ch0/1,  // PSB_0 slot  9
    #                    CA1_ch6/7    CA2_ch4/5 ,  CA3_ch2/3,    CA4_ch0/1,  // PSB_1 slot 13
    #                    CA5_ch6/7    CA6_ch4/5 ,  CA7_ch2/3,    CA8_ch0/1,  // PSB_2 slot 14
    #                    N/A,         N/A,         CA9_ch2/3,   CA10_ch0/1,  // PSB_3 slot 15
    #                   MQF4_ch6/7,  MQF3_ch4/5,  MQB2_ch2/3,   MQB1_ch0/1,  // PSB_4 slot 19
    #                   MQF8_ch6/7,  MQF7_ch4/5,  MQB6_ch2/3,   MQB5_ch0/1,  // PSB_5 slot 20
    #                   MQF12_ch6/7,MQF11_ch4/5, MQB10_ch2/3,   MQB9_ch0/1   // PSB_6 slot 21
    #),
    CableList = cms.vstring('Free',      'Free',       'Free',    'TechTr', 
                          'IsoEGQ',  'NoIsoEGQ',    'CenJetQ',   'ForJetQ', 
                         'TauJetQ',    'ESumsQ',        'HfQ',      'Free', 
                            'Free',      'Free',       'Free',      'Free', 
                            'MQF4',      'MQF3',       'MQB2',      'MQB1', 
                            'MQF8',      'MQF7',       'MQB6',      'MQB5', 
                           'MQF12',     'MQF11',      'MQB10',      'MQB9'),
        
    # gives the mapping of cables to GT PSBs via PSB index
    # 4 infinicables per PSB 
    CableToPsbMap = cms.vint32(0, 0, 0, 0,
                               1, 1, 1, 1, 
                               2, 2, 2, 2, 
                               3, 3, 3, 3, 
                               4, 4, 4, 4, 
                               5, 5, 5, 5, 
                               6, 6, 6, 6),
    
    # detailed input configuration for PSB (objects pro channel) 
    PsbInput = cms.VPSet(
                         cms.PSet(
                                  Slot = cms.int32(9),
                                  Ch0 = cms.vstring('TechTrig'),
                                  Ch1 = cms.vstring('TechTrig'),
                                  Ch2 = cms.vstring(),
                                  Ch3 = cms.vstring(),
                                  Ch4 = cms.vstring(),
                                  Ch5 = cms.vstring(),
                                  Ch6 = cms.vstring(),
                                  Ch7 = cms.vstring()
                                  ), 
                         cms.PSet(
                                  Slot = cms.int32(13),
                                  Ch0 = cms.vstring('ForJet', 'ForJet'),
                                  Ch1 = cms.vstring('ForJet', 'ForJet'),
                                  Ch2 = cms.vstring('CenJet', 'CenJet'),
                                  Ch3 = cms.vstring('CenJet', 'CenJet'),
                                  Ch4 = cms.vstring('NoIsoEG', 'NoIsoEG'),
                                  Ch5 = cms.vstring('NoIsoEG', 'NoIsoEG'),
                                  Ch6 = cms.vstring('IsoEG', 'IsoEG'),
                                  Ch7 = cms.vstring('IsoEG', 'IsoEG')
                                  ),                                  
                         cms.PSet(
                                  Slot = cms.int32(14),
                                  Ch0 = cms.vstring(),
                                  Ch1 = cms.vstring(),
                                  Ch2 = cms.vstring('HfBitCounts', 'HfRingEtSums'),
                                  Ch3 = cms.vstring(),
                                  Ch4 = cms.vstring('ETT','HTT'),
                                  Ch5 = cms.vstring('ETM', 'ETM'),
                                  Ch6 = cms.vstring('TauJet', 'TauJet'),
                                  Ch7 = cms.vstring('TauJet', 'TauJet')
                                  ),
                         cms.PSet(
                                  Slot = cms.int32(15),
                                  Ch0 = cms.vstring(),
                                  Ch1 = cms.vstring(),
                                  Ch2 = cms.vstring(),
                                  Ch3 = cms.vstring(),
                                  Ch4 = cms.vstring(),
                                  Ch5 = cms.vstring(),
                                  Ch6 = cms.vstring(),
                                  Ch7 = cms.vstring()
                                  ),
                         cms.PSet(
                                  Slot = cms.int32(19),
                                  Ch0 = cms.vstring(),
                                  Ch1 = cms.vstring(),
                                  Ch2 = cms.vstring(),
                                  Ch3 = cms.vstring(),
                                  Ch4 = cms.vstring(),
                                  Ch5 = cms.vstring(),
                                  Ch6 = cms.vstring(),
                                  Ch7 = cms.vstring()
                                  ),
                         cms.PSet(
                                  Slot = cms.int32(20),
                                  Ch0 = cms.vstring(),
                                  Ch1 = cms.vstring(),
                                  Ch2 = cms.vstring(),
                                  Ch3 = cms.vstring(),
                                  Ch4 = cms.vstring(),
                                  Ch5 = cms.vstring(),
                                  Ch6 = cms.vstring(),
                                  Ch7 = cms.vstring()
                                  ),
                         cms.PSet(
                                  Slot = cms.int32(21),
                                  Ch0 = cms.vstring(),
                                  Ch1 = cms.vstring(),
                                  Ch2 = cms.vstring(),
                                  Ch3 = cms.vstring(),
                                  Ch4 = cms.vstring(),
                                  Ch5 = cms.vstring(),
                                  Ch6 = cms.vstring(),
                                  Ch7 = cms.vstring()
                                  )
                        ) 
    
)


