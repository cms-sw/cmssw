import FWCore.ParameterSet.Config as cms

############################################################
# L1 Global Trigger Emulation
############################################################

# Conditions

from L1Trigger.Phase2L1GT.l1tGTProducer_cff import l1tGTProducer

from L1Trigger.Phase2L1GT.l1tGTSingleObjectCond_cfi import l1tGTSingleObjectCond
from L1Trigger.Phase2L1GT.l1tGTDoubleObjectCond_cfi import l1tGTDoubleObjectCond
from L1Trigger.Phase2L1GT.l1tGTTripleObjectCond_cfi import l1tGTTripleObjectCond
from L1Trigger.Phase2L1GT.l1tGTQuadObjectCond_cfi import l1tGTQuadObjectCond

from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import algorithms

####### MUON SEEDS ###########

#        regionsAbsEtaLowerBounds=cms.vdouble(0,1.2,3),
#        regionsMinPt=cms.vdouble(12,14,15)


SingleTkMuon22 = l1tGTSingleObjectCond.clone(
    tag =  cms.InputTag("l1tGTProducer", "GMTTkMuons"),
    #minPt = cms.double(20.3),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
    regionsMinPt=cms.vdouble(20.0,19.9,20.1)
)
pSingleTkMuon22 = cms.Path(SingleTkMuon22)
algorithms.append(cms.PSet(expression = cms.string("pSingleTkMuon22")))

DoubleTkMuon157 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        #minPt = cms.double(13.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(13.4,13.2,13.5)

    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        #minPt = cms.double(5.9),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        regionsMinPt=cms.vdouble(7,7,7)
    ),
    maxDz = cms.double(1),
)
pDoubleTkMuon15_7 = cms.Path(DoubleTkMuon157)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkMuon15_7")))

TripleTkMuon533 = l1tGTTripleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(5),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        qual = cms.vuint32(0b00000001, 0b00000010, 0b00000011, 0b00000100, 0b00000101, 0b00000110, 0b00000111, 0b00001000, 0b00001001, 0b00001010, 0b00001011, 0b00001100, 0b00001101, 0b00001110, 0b00001111, 0b00010000, 0b00010001, 0b00010010, 0b00010011, 0b00010100, 0b00010101, 0b00010110, 0b00010111, 0b00011000, 0b00011001, 0b00011010, 0b00011011, 0b00011100, 0b00011101, 0b00011110, 0b00011111, 0b00100000, 0b00100001, 0b00100010, 0b00100011, 0b00100100, 0b00100101, 0b00100110, 0b00100111, 0b00101000, 0b00101001, 0b00101010, 0b00101011, 0b00101100, 0b00101101, 0b00101110, 0b00101111, 0b00110000, 0b00110001, 0b00110010, 0b00110011, 0b00110100, 0b00110101, 0b00110110, 0b00110111, 0b00111000, 0b00111001, 0b00111010, 0b00111011, 0b00111100, 0b00111101, 0b00111110, 0b00111111, 0b01000000, 0b01000001, 0b01000010, 0b01000011, 0b01000100, 0b01000101, 0b01000110, 0b01000111, 0b01001000, 0b01001001, 0b01001010, 0b01001011, 0b01001100, 0b01001101, 0b01001110, 0b01001111, 0b01010000, 0b01010001, 0b01010010, 0b01010011, 0b01010100, 0b01010101, 0b01010110, 0b01010111, 0b01011000, 0b01011001, 0b01011010, 0b01011011, 0b01011100, 0b01011101, 0b01011110, 0b01011111, 0b01100000, 0b01100001, 0b01100010, 0b01100011, 0b01100100, 0b01100101, 0b01100110, 0b01100111, 0b01101000, 0b01101001, 0b01101010, 0b01101011, 0b01101100, 0b01101101, 0b01101110, 0b01101111, 0b01110000, 0b01110001, 0b01110010, 0b01110011, 0b01110100, 0b01110101, 0b01110110, 0b01110111, 0b01111000, 0b01111001, 0b01111010, 0b01111011, 0b01111100, 0b01111101, 0b01111110, 0b01111111, 0b10000000, 0b10000001, 0b10000010, 0b10000011, 0b10000100, 0b10000101, 0b10000110, 0b10000111, 0b10001000, 0b10001001, 0b10001010, 0b10001011, 0b10001100, 0b10001101, 0b10001110, 0b10001111, 0b10010000, 0b10010001, 0b10010010, 0b10010011, 0b10010100, 0b10010101, 0b10010110, 0b10010111, 0b10011000, 0b10011001, 0b10011010, 0b10011011, 0b10011100, 0b10011101, 0b10011110, 0b10011111, 0b10100000, 0b10100001, 0b10100010, 0b10100011, 0b10100100, 0b10100101, 0b10100110, 0b10100111, 0b10101000, 0b10101001, 0b10101010, 0b10101011, 0b10101100, 0b10101101, 0b10101110, 0b10101111, 0b10110000, 0b10110001, 0b10110010, 0b10110011, 0b10110100, 0b10110101, 0b10110110, 0b10110111, 0b10111000, 0b10111001, 0b10111010, 0b10111011, 0b10111100, 0b10111101, 0b10111110, 0b10111111, 0b11000000, 0b11000001, 0b11000010, 0b11000011, 0b11000100, 0b11000101, 0b11000110, 0b11000111, 0b11001000, 0b11001001, 0b11001010, 0b11001011, 0b11001100, 0b11001101, 0b11001110, 0b11001111, 0b11010000, 0b11010001, 0b11010010, 0b11010011, 0b11010100, 0b11010101, 0b11010110, 0b11010111, 0b11011000, 0b11011001, 0b11011010, 0b11011011, 0b11011100, 0b11011101, 0b11011110, 0b11011111, 0b11100000, 0b11100001, 0b11100010, 0b11100011, 0b11100100, 0b11100101, 0b11100110, 0b11100111, 0b11101000, 0b11101001, 0b11101010, 0b11101011, 0b11101100, 0b11101101, 0b11101110, 0b11101111, 0b11110000, 0b11110001, 0b11110010, 0b11110011, 0b11110100, 0b11110101, 0b11110110, 0b11110111, 0b11111000, 0b11111001, 0b11111010, 0b11111011, 0b11111100, 0b11111101, 0b11111110, 0b11111111)
        #regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        #regionsMinPt=cms.vdouble(3.9,3.9,4.0)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(3),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        qual = cms.vuint32(0b00000001, 0b00000010, 0b00000011, 0b00000100, 0b00000101, 0b00000110, 0b00000111, 0b00001000, 0b00001001, 0b00001010, 0b00001011, 0b00001100, 0b00001101, 0b00001110, 0b00001111, 0b00010000, 0b00010001, 0b00010010, 0b00010011, 0b00010100, 0b00010101, 0b00010110, 0b00010111, 0b00011000, 0b00011001, 0b00011010, 0b00011011, 0b00011100, 0b00011101, 0b00011110, 0b00011111, 0b00100000, 0b00100001, 0b00100010, 0b00100011, 0b00100100, 0b00100101, 0b00100110, 0b00100111, 0b00101000, 0b00101001, 0b00101010, 0b00101011, 0b00101100, 0b00101101, 0b00101110, 0b00101111, 0b00110000, 0b00110001, 0b00110010, 0b00110011, 0b00110100, 0b00110101, 0b00110110, 0b00110111, 0b00111000, 0b00111001, 0b00111010, 0b00111011, 0b00111100, 0b00111101, 0b00111110, 0b00111111, 0b01000000, 0b01000001, 0b01000010, 0b01000011, 0b01000100, 0b01000101, 0b01000110, 0b01000111, 0b01001000, 0b01001001, 0b01001010, 0b01001011, 0b01001100, 0b01001101, 0b01001110, 0b01001111, 0b01010000, 0b01010001, 0b01010010, 0b01010011, 0b01010100, 0b01010101, 0b01010110, 0b01010111, 0b01011000, 0b01011001, 0b01011010, 0b01011011, 0b01011100, 0b01011101, 0b01011110, 0b01011111, 0b01100000, 0b01100001, 0b01100010, 0b01100011, 0b01100100, 0b01100101, 0b01100110, 0b01100111, 0b01101000, 0b01101001, 0b01101010, 0b01101011, 0b01101100, 0b01101101, 0b01101110, 0b01101111, 0b01110000, 0b01110001, 0b01110010, 0b01110011, 0b01110100, 0b01110101, 0b01110110, 0b01110111, 0b01111000, 0b01111001, 0b01111010, 0b01111011, 0b01111100, 0b01111101, 0b01111110, 0b01111111, 0b10000000, 0b10000001, 0b10000010, 0b10000011, 0b10000100, 0b10000101, 0b10000110, 0b10000111, 0b10001000, 0b10001001, 0b10001010, 0b10001011, 0b10001100, 0b10001101, 0b10001110, 0b10001111, 0b10010000, 0b10010001, 0b10010010, 0b10010011, 0b10010100, 0b10010101, 0b10010110, 0b10010111, 0b10011000, 0b10011001, 0b10011010, 0b10011011, 0b10011100, 0b10011101, 0b10011110, 0b10011111, 0b10100000, 0b10100001, 0b10100010, 0b10100011, 0b10100100, 0b10100101, 0b10100110, 0b10100111, 0b10101000, 0b10101001, 0b10101010, 0b10101011, 0b10101100, 0b10101101, 0b10101110, 0b10101111, 0b10110000, 0b10110001, 0b10110010, 0b10110011, 0b10110100, 0b10110101, 0b10110110, 0b10110111, 0b10111000, 0b10111001, 0b10111010, 0b10111011, 0b10111100, 0b10111101, 0b10111110, 0b10111111, 0b11000000, 0b11000001, 0b11000010, 0b11000011, 0b11000100, 0b11000101, 0b11000110, 0b11000111, 0b11001000, 0b11001001, 0b11001010, 0b11001011, 0b11001100, 0b11001101, 0b11001110, 0b11001111, 0b11010000, 0b11010001, 0b11010010, 0b11010011, 0b11010100, 0b11010101, 0b11010110, 0b11010111, 0b11011000, 0b11011001, 0b11011010, 0b11011011, 0b11011100, 0b11011101, 0b11011110, 0b11011111, 0b11100000, 0b11100001, 0b11100010, 0b11100011, 0b11100100, 0b11100101, 0b11100110, 0b11100111, 0b11101000, 0b11101001, 0b11101010, 0b11101011, 0b11101100, 0b11101101, 0b11101110, 0b11101111, 0b11110000, 0b11110001, 0b11110010, 0b11110011, 0b11110100, 0b11110101, 0b11110110, 0b11110111, 0b11111000, 0b11111001, 0b11111010, 0b11111011, 0b11111100, 0b11111101, 0b11111110, 0b11111111)
        #regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        #regionsMinPt=cms.vdouble(2.0,2.0,2.1)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(3),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        qual = cms.vuint32(0b00000001, 0b00000010, 0b00000011, 0b00000100, 0b00000101, 0b00000110, 0b00000111, 0b00001000, 0b00001001, 0b00001010, 0b00001011, 0b00001100, 0b00001101, 0b00001110, 0b00001111, 0b00010000, 0b00010001, 0b00010010, 0b00010011, 0b00010100, 0b00010101, 0b00010110, 0b00010111, 0b00011000, 0b00011001, 0b00011010, 0b00011011, 0b00011100, 0b00011101, 0b00011110, 0b00011111, 0b00100000, 0b00100001, 0b00100010, 0b00100011, 0b00100100, 0b00100101, 0b00100110, 0b00100111, 0b00101000, 0b00101001, 0b00101010, 0b00101011, 0b00101100, 0b00101101, 0b00101110, 0b00101111, 0b00110000, 0b00110001, 0b00110010, 0b00110011, 0b00110100, 0b00110101, 0b00110110, 0b00110111, 0b00111000, 0b00111001, 0b00111010, 0b00111011, 0b00111100, 0b00111101, 0b00111110, 0b00111111, 0b01000000, 0b01000001, 0b01000010, 0b01000011, 0b01000100, 0b01000101, 0b01000110, 0b01000111, 0b01001000, 0b01001001, 0b01001010, 0b01001011, 0b01001100, 0b01001101, 0b01001110, 0b01001111, 0b01010000, 0b01010001, 0b01010010, 0b01010011, 0b01010100, 0b01010101, 0b01010110, 0b01010111, 0b01011000, 0b01011001, 0b01011010, 0b01011011, 0b01011100, 0b01011101, 0b01011110, 0b01011111, 0b01100000, 0b01100001, 0b01100010, 0b01100011, 0b01100100, 0b01100101, 0b01100110, 0b01100111, 0b01101000, 0b01101001, 0b01101010, 0b01101011, 0b01101100, 0b01101101, 0b01101110, 0b01101111, 0b01110000, 0b01110001, 0b01110010, 0b01110011, 0b01110100, 0b01110101, 0b01110110, 0b01110111, 0b01111000, 0b01111001, 0b01111010, 0b01111011, 0b01111100, 0b01111101, 0b01111110, 0b01111111, 0b10000000, 0b10000001, 0b10000010, 0b10000011, 0b10000100, 0b10000101, 0b10000110, 0b10000111, 0b10001000, 0b10001001, 0b10001010, 0b10001011, 0b10001100, 0b10001101, 0b10001110, 0b10001111, 0b10010000, 0b10010001, 0b10010010, 0b10010011, 0b10010100, 0b10010101, 0b10010110, 0b10010111, 0b10011000, 0b10011001, 0b10011010, 0b10011011, 0b10011100, 0b10011101, 0b10011110, 0b10011111, 0b10100000, 0b10100001, 0b10100010, 0b10100011, 0b10100100, 0b10100101, 0b10100110, 0b10100111, 0b10101000, 0b10101001, 0b10101010, 0b10101011, 0b10101100, 0b10101101, 0b10101110, 0b10101111, 0b10110000, 0b10110001, 0b10110010, 0b10110011, 0b10110100, 0b10110101, 0b10110110, 0b10110111, 0b10111000, 0b10111001, 0b10111010, 0b10111011, 0b10111100, 0b10111101, 0b10111110, 0b10111111, 0b11000000, 0b11000001, 0b11000010, 0b11000011, 0b11000100, 0b11000101, 0b11000110, 0b11000111, 0b11001000, 0b11001001, 0b11001010, 0b11001011, 0b11001100, 0b11001101, 0b11001110, 0b11001111, 0b11010000, 0b11010001, 0b11010010, 0b11010011, 0b11010100, 0b11010101, 0b11010110, 0b11010111, 0b11011000, 0b11011001, 0b11011010, 0b11011011, 0b11011100, 0b11011101, 0b11011110, 0b11011111, 0b11100000, 0b11100001, 0b11100010, 0b11100011, 0b11100100, 0b11100101, 0b11100110, 0b11100111, 0b11101000, 0b11101001, 0b11101010, 0b11101011, 0b11101100, 0b11101101, 0b11101110, 0b11101111, 0b11110000, 0b11110001, 0b11110010, 0b11110011, 0b11110100, 0b11110101, 0b11110110, 0b11110111, 0b11111000, 0b11111001, 0b11111010, 0b11111011, 0b11111100, 0b11111101, 0b11111110, 0b11111111)
        #regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
        #regionsMinPt=cms.vdouble(2.0,2.0,2.1)
    ),
    delta12 = cms.PSet(
        maxDz = cms.double(1)
    ),
    delta13 = cms.PSet(
        maxDz = cms.double(1)
    ),
    #delta23 = cms.PSet(
    #    maxDz = cms.double(1)
    #)
)
pTripleTkMuon5_3_3 = cms.Path(TripleTkMuon533)
algorithms.append(cms.PSet(expression = cms.string("pTripleTkMuon5_3_3")))

####### EG and PHO seeds ###########

SingleEGEle51 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
    #minPt = cms.double(29.9),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
    regionsMinPt=cms.vdouble(40.7,39.6),
    regionsQual=cms.vuint32(0b0010,0b0100),
    #qual = cms.vuint32(0b0010)
)
pSingleEGEle51 = cms.Path(SingleEGEle51) 
algorithms.append(cms.PSet(expression = cms.string("pSingleEGEle51")))

DoubleEGEle3724 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        #minPt = cms.double(20.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(28.9,28.4),
        regionsQual=cms.vuint32(0b0010,0b0100),
        #qual = cms.vuint32(0b0010)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        #minPt = cms.double(9.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(17.9,18.0),
        regionsQual=cms.vuint32(0b0010,0b0100),
        #qual = cms.vuint32(0b0010)
    ),
    minDR = cms.double(0.1),
)
pDoubleEGEle37_24 = cms.Path(DoubleEGEle3724)
algorithms.append(cms.PSet(expression = cms.string("pDoubleEGEle37_24")))

IsoTkEleEGEle2212 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        #minPt = cms.double(20.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(17.8,17.1),
        #regionsQual=cms.vuint32(0b0000,0b0010),
        regionsMaxIso = cms.vdouble(0.13,0.28)
        #qual = cms.vuint32(0b0010)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        #minPt = cms.double(9.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(7.8,8.3),
        regionsQual=cms.vuint32(0b0010,0b0100)
        #qual = cms.vuint32(0b0010)
    ),
    minDR = cms.double(0.1),
)
pIsoTkEleEGEle22_12 = cms.Path(IsoTkEleEGEle2212)
algorithms.append(cms.PSet(expression = cms.string("pIsoTkEleEGEle22_12")))

SingleTkEle36 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
    #minPt = cms.double(29.9),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
    regionsMinPt=cms.vdouble(29.8,28.5),
    regionsQual=cms.vuint32(0b0010,0b0010)
    #qual = cms.vuint32(0b0010)
)
pSingleTkEle36 = cms.Path(SingleTkEle36) 
algorithms.append(cms.PSet(expression = cms.string("pSingleTkEle36")))

SingleIsoTkEle28 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
    #minPt = cms.double(29.9),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
    regionsMinPt=cms.vdouble(23,22.1),
    #regionsQual=cms.vuint32(0b0000,0b0010),
    regionsMaxIso = cms.vdouble(0.13,0.28)
    #qual = cms.vuint32(0b0010)
)
pSingleIsoTkEle28 = cms.Path(SingleIsoTkEle28) 
algorithms.append(cms.PSet(expression = cms.string("pSingleIsoTkEle28")))

#SingleIsoTkEle28Barrel = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
#    minPt = cms.double(23), 
#    minEta = cms.double(-1.479),
#    maxEta = cms.double(1.479),
    #maxIso = cms.double(0.13),
#)
#pSingleIsoTkEle28Barrel = cms.Path(SingleIsoTkEle28Barrel)
#algorithms.append(cms.PSet(expression = cms.string("pSingleIsoTkEle28Barrel")))

#SingleIsoTkEle28BarrelQual = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
#    minPt = cms.double(23), 
#    minEta = cms.double(-1.479),
#    maxEta = cms.double(1.479),
#    qual = cms.vuint32(0b0000),
    #maxIso = cms.double(0.13),
#)
#pSingleIsoTkEle28BarrelQual = cms.Path(SingleIsoTkEle28BarrelQual)
#algorithms.append(cms.PSet(expression = cms.string("pSingleIsoTkEle28BarrelQual")))

#SingleIsoTkEle28Endcap = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
#    minPt = cms.double(21.9),
#    minEtaAbs = cms.double(1.479),
#    maxEtaAbs = cms.double(2.4),
#    qual = cms.vuint32(0b0010,0b0011,0b0110,0b1010,0b0111,0b1011,0b1110,0b1111),
    #maxIso = cms.double(0.28)
#)
#pSingleIsoTkEle28Endcap = cms.Path(SingleIsoTkEle28Endcap) 
#algorithms.append(cms.PSet(expression = cms.string("pSingleIsoTkEle28Endcap")))

#algorithms.append(cms.PSet(name=cms.string("pSingleIsoTkEle28OLD"),
#                       expression=cms.string("pSingleIsoTkEle28Barrel or pSingleIsoTkEle28Endcap")))


SingleIsoTkPho36 = l1tGTSingleObjectCond.clone(
    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
    #minPt = cms.double(30.8),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
    regionsMinPt=cms.vdouble(30.4,29.0),
    regionsQual=cms.vuint32(0b0010,0b0100),
    regionsMaxIso = cms.vdouble(0.25,0.205)
    #qual = cms.vuint32(0b0100),
    #maxIso = cms.double(0.205)
)
pSingleIsoTkPho36 = cms.Path(SingleIsoTkPho36) 

algorithms.append(cms.PSet(expression=cms.string("pSingleIsoTkPho36")))

#SingleIsoTkPho36Barrel = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
#    minPt = cms.double(30.8),
#    minEta = cms.double(-1.479), 
#    maxEta = cms.double(1.479),
#    qual = cms.vuint32(0b0010),
#    maxIso = cms.double(0.25)
#)
#pSingleIsoTkPho36Barrel = cms.Path(SingleIsoTkPho36Barrel) 

#SingleIsoTkPho36Endcap = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
#    minPt = cms.double(30.8),
#    minEtaAbs = cms.double(1.479),
#    maxEtaAbs = cms.double(2.4),
#    qual = cms.vuint32(0b0100),
#    maxIso = cms.double(0.205)
#)
#pSingleIsoTkPho36Endcap = cms.Path(SingleIsoTkPho36Endcap) 
#
#algorithms.append(cms.PSet(name=cms.string("pSingleIsoTkPho36"),
#                       expression=cms.string("pSingleIsoTkPho36Barrel or pSingleIsoTkPho36Endcap")))

DoubleTkEle2512 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        #minPt = cms.double(20.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(20.3,19.5),
        regionsQual=cms.vuint32(0b0010,0b0000)
        #qual = cms.vuint32(0b0010)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        #minPt = cms.double(9.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(9.1,8.8),
        regionsQual=cms.vuint32(0b0010,0b0000)
        #qual = cms.vuint32(0b0010)
    ),
    maxDz = cms.double(1),
)
pDoubleTkEle25_12 = cms.Path(DoubleTkEle2512)
algorithms.append(cms.PSet(expression = cms.string("pDoubleTkEle25_12")))

DoubleIsoTkPho2212 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        #minPt = cms.double(20.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(17.6,15.9),
        regionsQual=cms.vuint32(0b0010,0b0100),
        regionsMaxIso = cms.vdouble(0.25,0.205)
        #qual = cms.vuint32(0b0010)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
        #minPt = cms.double(9.6),
        minEta = cms.double(-2.4),
        maxEta = cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
        regionsMinPt=cms.vdouble(8.5,6.0),
        regionsQual=cms.vuint32(0b0010,0b0100),
        regionsMaxIso = cms.vdouble(0.25,0.205)
        #qual = cms.vuint32(0b0010)
    ),
)
pDoubleIsoTkPho22_12 = cms.Path(DoubleIsoTkPho2212)
algorithms.append(cms.PSet(expression = cms.string("pDoubleIsoTkPho22_12")))



#SingleIsoTkPho36 = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
#    #minPt = cms.double(30.8),
#    minEta = cms.double(-2.4),
#    maxEta = cms.double(2.4),
#    regionsAbsEtaLowerBounds=cms.vdouble(0,1.479),
#    regionsMinPt=cms.vdouble(30.8,29.2),
#    regionsQual=cms.vuint32(0b0010,0b0100)
    #qual = cms.vuint32(0b0100),
    #maxIso = cms.double(0.205)
#)
#pSingleIsoTkPho36 = cms.Path(SingleIsoTkPho36) 

#algorithms.append(cms.PSet(expression=cms.string("pSingleIsoTkPho36")))





#SingleIsoTkPho22Barrel = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
#    minPt = cms.double(17.1),
#    minEta = cms.double(-1.479), 
#    maxEta = cms.double(1.479),
#    qual = cms.vuint32(0b0010),
#    maxIso = cms.double(0.25)
#)
#pSingleIsoTkPho22Barrel = cms.Path(SingleIsoTkPho22Barrel) 

#SingleIsoTkPho22Endcap = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
#    minPt = cms.double(17.1),
#    minEtaAbs = cms.double(1.479),
#    maxEtaAbs = cms.double(2.4),
#    qual = cms.vuint32(0b0100),
#    maxIso = cms.double(0.205)
#)
#pSingleIsoTkPho22Endcap = cms.Path(SingleIsoTkPho22Endcap) 

#SingleIsoTkPho12Barrel = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
#    minPt = cms.double(8.8),
#    minEta = cms.double(-1.479), 
#    maxEta = cms.double(1.479),
#    qual = cms.vuint32(0b0010),
#    maxIso = cms.double(0.25)
#)
#pSingleIsoTkPho12Barrel = cms.Path(SingleIsoTkPho12Barrel) 

#SingleIsoTkPho12EndcapPos = l1tGTSingleObjectCond.clone(
#    tag = cms.InputTag("l1tGTProducer", "CL2Photons"),
#    minPt = cms.double(8.8),
#    minEtaAbs = cms.double(1.479),
#    maxEtaAbs = cms.double(2.4),
#    qual = cms.vuint32(0b0100),
#    maxIso = cms.double(0.205)
#)
#pSingleIsoTkPho12EndcapPos = cms.Path(SingleIsoTkPho12EndcapPos) 

#algorithms.append(cms.PSet(name=cms.string("pDoubleTkIsoPho22_12"),
#                       expression=cms.string("(pSingleIsoTkPho22Barrel or pSingleIsoTkPho22EndcapPos or pSingleIsoTkPho22EndcapNeg) and (pSingleIsoTkPho12Barrel or pSingleIsoTkPho12EndcapPos or pSingleIsoTkPho12EndcapNeg)")))



DoublePuppiTau5252 = l1tGTDoubleObjectCond.clone(
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
        minEta = cms.double(-2.172),
        maxEta = cms.double(2.172),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(28.6,19.6),
        minHwIso = cms.int32(286),
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Taus"),
        minEta = cms.double(-2.172),
        maxEta = cms.double(2.172),
        regionsAbsEtaLowerBounds=cms.vdouble(0,1.5),
        regionsMinPt=cms.vdouble(28.6,19.6),
        minHwIso = cms.int32(286),
    ),
    minDR = cms.double(0.5),
)
pDoublePuppiTau52_52 = cms.Path(DoublePuppiTau5252)
algorithms.append(cms.PSet(expression = cms.string("pDoublePuppiTau52_52")))

