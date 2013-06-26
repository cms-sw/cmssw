#
# cfi for the setup of L1 GT PSB boards.
#

import FWCore.ParameterSet.Config as cms

l1GtPsbSetup = cms.ESProducer("L1GtPsbSetupTrivialProducer",

    # actual input to PSB  - see l1GtBoardMaps_cfi.py
    #CableList = cms.vstring( N/A,        N/A,         N/A,      TTr_ch0/1,  // PSB_0 slot  9
    #                    CA1_ch6/7    CA2_ch4/5 ,  CA3_ch2/3,    CA4_ch0/1,  // PSB_1 slot 13
    #                    CA5_ch6/7    CA6_ch4/5 ,  CA7_ch2/3,    CA8_ch0/1,  // PSB_2 slot 14
    #                    N/A,         N/A,         CA9_ch2/3,   CA10_ch0/1,  // PSB_3 slot 15
    #                   MQF4_ch6/7,  MQF3_ch4/5,  MQB2_ch2/3,   MQB1_ch0/1,  // PSB_4 slot 19
    #                   MQF8_ch6/7,  MQF7_ch4/5,  MQB6_ch2/3,   MQB5_ch0/1,  // PSB_5 slot 20
    #                   MQF12_ch6/7,MQF11_ch4/5, MQB10_ch2/3,   MQB9_ch0/1   // PSB_6 slot 21
    #),
    #CableList = cms.vstring('Free',      'Free',       'Free',    'TechTr', 
    #                      'IsoEGQ',  'NoIsoEGQ',    'CenJetQ',   'ForJetQ', 
    #                     'TauJetQ',    'ESumsQ',        'HfQ',      'Free', 
    #                        'Free',      'Free',       'Free',      'Free', 
    #                        'MQF4',      'MQF3',       'MQB2',      'MQB1', 
    #                        'MQF8',      'MQF7',       'MQB6',      'MQB5', 
    #                       'MQF12',     'MQF11',      'MQB10',      'MQB9'),
    
    # for vector<bool> use: 1 for True and 0 for false
    PsbSetup = cms.VPSet(
                         cms.PSet(
                                  Slot = cms.int32(9),
                                  Ch0SendLvds = cms.bool(True),
                                  Ch1SendLvds = cms.bool(True),
                                  EnableRecLvds = cms.vuint32(
                                           1, 1, 1, 1, 
                                           1, 1, 1, 1,
                                           1, 1, 1, 1,
                                           1, 1, 1, 1),
                                  EnableRecSerLink = cms.vuint32(
                                           0, 0, 
                                           0, 0,
                                           0, 0,
                                           0, 0)                                   
                                  ), 
                         cms.PSet(
                                  Slot = cms.int32(13),
                                  Ch0SendLvds = cms.bool(False),
                                  Ch1SendLvds = cms.bool(False),
                                  EnableRecLvds = cms.vuint32(
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0),
                                  EnableRecSerLink = cms.vuint32(
                                           1, 1, 
                                           1, 1,
                                           1, 1,
                                           1, 1)                                   
                                  ),                                  
                         cms.PSet(
                                  Slot = cms.int32(14),
                                  Ch0SendLvds = cms.bool(False),
                                  Ch1SendLvds = cms.bool(False),
                                  EnableRecLvds = cms.vuint32(
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0),
                                  EnableRecSerLink = cms.vuint32(
                                           1, 1, 
                                           1, 1,
                                           1, 1,
                                           1, 1)                                   
                                  ),
                         cms.PSet(
                                  Slot = cms.int32(15),
                                  Ch0SendLvds = cms.bool(True),
                                  Ch1SendLvds = cms.bool(True),
                                  EnableRecLvds = cms.vuint32(
                                           1, 1, 1, 1, 
                                           1, 1, 1, 1,
                                           1, 1, 1, 1,
                                           1, 1, 1, 1),
                                  EnableRecSerLink = cms.vuint32(
                                           0, 0, 
                                           0, 0,
                                           0, 0,
                                           0, 0)                                   
                                  ),
                         cms.PSet(
                                  Slot = cms.int32(19),
                                  Ch0SendLvds = cms.bool(False),
                                  Ch1SendLvds = cms.bool(False),
                                  EnableRecLvds = cms.vuint32(
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0),
                                  EnableRecSerLink = cms.vuint32(
                                           0, 0, 
                                           0, 0,
                                           0, 0,
                                           0, 0)                                   
                                  ),
                         cms.PSet(
                                  Slot = cms.int32(20),
                                  Ch0SendLvds = cms.bool(False),
                                  Ch1SendLvds = cms.bool(False),
                                  EnableRecLvds = cms.vuint32(
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0),
                                  EnableRecSerLink = cms.vuint32(
                                           0, 0, 
                                           0, 0,
                                           0, 0,
                                           0, 0)                                   
                                  ),
                         cms.PSet(
                                  Slot = cms.int32(21),
                                  Ch0SendLvds = cms.bool(False),
                                  Ch1SendLvds = cms.bool(False),
                                  EnableRecLvds = cms.vuint32(
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0, 0),
                                  EnableRecSerLink = cms.vuint32(
                                           0, 0, 
                                           0, 0,
                                           0, 0,
                                           0, 0)                                   
                                  )
                        ) 
    
)


