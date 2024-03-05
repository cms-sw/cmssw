import FWCore.ParameterSet.Config as cms

gmtStubs = cms.EDProducer("Phase2L1TGMTStubProducer",
   verbose = cms.int32(0),
   srcCSC = cms.InputTag("simCscTriggerPrimitiveDigis"),
   srcDT = cms.InputTag("dtTriggerPhase2PrimitiveDigis"),
   srcDTTheta = cms.InputTag("simDtTriggerPrimitiveDigis"),
   srcRPC = cms.InputTag("simMuonRPCDigis"),
   Endcap =cms.PSet(
        verbose              = cms.uint32(0),
        minBX                = cms.int32(0),
        maxBX                = cms.int32(0),
        coord1LSB            = cms.double(0.00076660156*8),#was 32
        eta1LSB              = cms.double(7.68334e-04*32),
        coord2LSB            = cms.double(0.00076660156*8),# was 32
        eta2LSB              = cms.double(7.68334e-04*32),
        phiMatch             = cms.double(0.05),
        etaMatch             = cms.double(0.1)
    ),
   Barrel = cms.PSet(
       verbose            = cms.int32(0),#2 for patterns
       minPhiQuality      = cms.int32(0),#was 0!!!
       minThetaQuality    = cms.int32(0),
       minBX              = cms.int32(0),
       maxBX              = cms.int32(0),
       phiLSB             = cms.double(2.3968450e-05),
       phiBDivider        = cms.int32(1),
       etaLSB             = cms.double(7.68334e-04*32),
       eta_1              = cms.vint32(int(-1503/32),int(-1446/32),int(-1387/32),int(-1327/32),int(-1266/32),int(-1194/32),int(-1125/32),int(-985/32),int(-916/32),int(-839/32),int(-752/32),int(-670/32),int(-582/32),int(-489/32),int(-315/32),int(-213/32),int(-115/32),int(-49/32),int(49/32),int(115/32),int(213/32),int(315/32),int(489/32),int(582/32),int(670/32),int(752/32),int(839/32),int(916/32),int(985/32),int(1125/32),int(1194/32),int(1266/32),int(1327/32),int(1387/32),int(1446/32), 1503),
       eta_2              = cms.vint32(int(-1334/32),int(-1279/32),int(-1227/32),int(-1168/32),int(-1109/32),int(-1044/32),int(-982/32),int(-861/32),int(-793/32),int(-720/32),int(-648/32),int(-577/32),int(-496/32),int(-425/32),int(-268/32),int(-185/32),int(-97/32),int(-51/32),int(51/32),int(97/32),int(185/32),int(268/32),int(425/32),int(496/32),int(577/32),int(648/32),int(720/32),int(793/32),int(861/32),int(982/32),int(1044/32),int(1109/32),int(1168/32),int(1227/32),int(1279/32),1334),
       eta_3              = cms.vint32(int(-1148/32),int(-1110/32),int(-1051/32),int(-1004/32),int(-947/32),int(-895/32),int(-839/32),int(-728/32),int(-668/32),int(-608/32),int(-546/32),int(-485/32),int(-425/32),int(-366/32),int(-222/32),int(-155/32),int(-87/32),int(-40/32),int(40/32),int(87/32),int(155/32),int(222/32),int(366/32),int(425/32),int(485/32),int(546/32),int(608/32),int(668/32),int(728/32),int(839/32),int(895/32),int(947/32),int(1004/32),int(1051/32),int(1110/32), 1148),

       coarseEta_1        = cms.vint32(int(0/32),int(758/32),int(1336/32)),
       coarseEta_2        = cms.vint32(int(0/32),int(653/32),int(1168/32)),
       coarseEta_3        = cms.vint32(int(0/32),int(552/32),int(1001/32)),
       coarseEta_4        = cms.vint32(int(0/32),int(478/32),int(878/32)),
       phiOffset          = cms.vint32(0,0,0,0)
   )
)

