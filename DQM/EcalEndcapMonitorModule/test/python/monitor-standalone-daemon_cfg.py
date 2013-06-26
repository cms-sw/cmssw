import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALDQM")

import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()

#import RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi
#process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi.ecalGlobalUncalibRecHit.clone()

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi")

process.load("DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi")

process.load("DQM.EcalEndcapMonitorClient.EcalEndcapMonitorClient_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.ecalEBunpacker = cms.EDProducer("EcalDCCTB07UnpackingModule",
    fedRawDataCollectionTag = cms.InputTag('rawDataCollector'),
    produceEBdigi = cms.untracked.bool(False),
    produceEEdigi = cms.untracked.bool(True),

    tbName = cms.untracked.string('h2'),
    ccuIDs = cms.untracked.vint32(1, 71, 80, 45),
    statusIDs = cms.untracked.vint32(1, 2, 3, 4),
    positionIDs = cms.untracked.vint32(6, 2, 5, 1),
    stripIDs = cms.untracked.vint32(1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1),
    towerIDs = cms.untracked.vint32(1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6),
    ics = cms.untracked.vint32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190),
    channelIDs = cms.untracked.vint32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)

#    tbName = cms.untracked.string('h4'),
#    ccuIDs = cms.untracked.vint32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 69, 70),
#    statusIDs = cms.untracked.vint32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70),
#    positionIDs = cms.untracked.vint32(4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17, 69, 70),
#    stripIDs = (cms.untracked.vint32(4, 4, 3, 3, 3, 4, 4, 2, 2, 2, 4, 4, 4, 1, 1, 5, 5, 3, 1, 1, 4, 4, 3, 2, 3, 4, 4, 4, 2, 2, 5, 4, 4, 1, 1, 5, 5, 3, 1, 1, 5, 4, 2, 2, 2, 5, 5, 1, 1, 1, 5, 5, 3, 2, 1, 5, 3, 3, 3, 1, 5, 5, 2, 1, 1, 5, 5, 3, 1, 1, 5, 3, 3, 2, 2, 4, 4, 4, 2, 2, 5, 5, 1, 1, 1, 5, 3, 3, 3, 3, 5, 3, 3, 2, 2, 4, 4, 2, 2, 2, 5, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 5, 5, 2, 2, 2, 5, 4, 4, 2, 2, 3, 3, 3, 1, 1, 5, 4, 4, 1, 1, 5, 4, 4, 2, 2, 5, 4, 1, 1, 1, 5, 3, 3, 1, 1, 5, 5, 3, 2, 1, 5, 4, 4, 1, 1, 5, 3, 3, 1, 1, 5, 5, 2, 2, 1, 5, 3, 3, 2, 2, 5, 4, 1, 1, 1, 5, 4, 3, 3, 3, 5, 5, 2, 2, 2, 5, 3, 3, 2, 2, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4, 2, 2, 2, 5, 5, 3, 3, 1, 4, 4, 4, 2, 2, 4, 4, 3, 1, 1, 4, 4, 4, 2, 2, 5, 5, 3, 3, 1, 5, 4, 4, 2, 2, 5, 4, 4, 1, 1, 5, 5, 1, 1, 1, 5, 4, 3, 1, 1)+cms.untracked.vint32(5, 5, 3, 2, 1, 5, 5, 2, 2, 1, 5, 5, 3, 1, 1, 4, 4, 2, 2, 1, 5, 3, 3, 1, 1, 5, 5, 2, 2, 2, 5, 3, 3, 3, 3, 4, 4, 2, 2, 2, 5, 3, 3, 1, 1, 5, 2, 2, 2, 2, 5, 5, 3, 2, 2, 4, 4, 4, 1, 1, 5, 5, 3, 2, 1, 5, 4, 4, 2, 1, 5, 5, 3, 2, 2, 4, 4, 3, 1, 1, 5, 3, 3, 2, 1, 5, 4, 4, 1, 1, 5, 3, 3, 2, 1, 5, 3, 3, 2, 1, 5, 3, 3, 2, 1, 5, 4, 3, 1, 1, 4, 4, 3, 1, 1, 5, 5, 3, 2, 2, 5, 4, 2, 2, 1, 5, 3, 3, 3, 3, 4, 4, 4, 1, 1, 5, 5, 3, 2, 2, 4, 4, 4, 4, 1, 5, 5, 3, 1, 1, 5, 4, 3, 3, 3, 4, 4, 4, 2, 2, 4, 4, 4, 2, 1, 5, 5, 3, 1, 1, 5, 4, 3, 1, 1, 5, 4, 2, 2, 1, 4, 4, 2, 2, 1, 5, 3, 3, 3, 1, 5, 4, 3, 1, 1, 5, 4, 2, 1, 1, 5, 5, 2, 2, 1, 4, 4, 4, 2, 2, 5, 4, 2, 2, 1, 5, 5, 3, 3, 1, 5, 5, 3, 1, 1, 4, 4, 2, 2, 2, 5, 4, 2, 2, 2, 5, 3, 3, 3, 1, 5, 3, 3, 3, 3)),
#    towerIDs = (cms.untracked.vint32(1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11)+cms.untracked.vint32(12, 12, 12, 12, 12, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20)),
#    ics = (cms.untracked.vint32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255)+cms.untracked.vint32(256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500)),
#    channelIDs = (cms.untracked.vint32(4, 5, 3, 4, 5, 4, 5, 3, 4, 5, 3, 4, 5, 4, 5, 4, 5, 5, 4, 5, 2, 3, 1, 5, 2, 1, 2, 3, 1, 2, 5, 1, 2, 2, 3, 2, 3, 4, 2, 3, 5, 1, 2, 3, 4, 4, 5, 3, 4, 5, 3, 4, 5, 5, 1, 1, 1, 2, 3, 1, 3, 4, 1, 4, 5, 2, 3, 5, 1, 2, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 1, 2, 1, 2, 3, 1, 1, 2, 3, 4, 1, 1, 2, 1, 2, 1, 2, 1, 2, 3, 5, 5, 3, 4, 5, 1, 2, 3, 4, 5, 3, 4, 5, 4, 5, 4, 5, 3, 4, 5, 4, 3, 4, 1, 2, 3, 4, 5, 4, 5, 5, 1, 2, 2, 3, 3, 4, 5, 1, 2, 3, 2, 3, 4, 5, 5, 1, 2, 2, 3, 3, 4, 5, 5, 1, 2, 2, 3, 4, 5, 2, 4, 5, 1, 2, 3, 4, 4, 5, 1, 2, 3, 4, 3, 4, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 1, 2, 3, 1, 1, 2, 1, 2, 1, 2, 3, 4, 5, 5, 2, 3, 4, 5, 4, 5, 3, 4, 5, 4, 5, 4, 5, 5, 3, 4, 5, 4, 5, 3, 4, 1, 4, 5, 1, 2, 3, 1, 2, 2, 3, 2, 3, 4, 5, 1, 2, 2, 3, 5, 1, 2, 2, 3, 4, 5, 3, 4, 5, 1, 5, 1, 2, 3)+cms.untracked.vint32(3, 4, 5, 1, 5, 3, 4, 4, 5, 1, 2, 3, 5, 1, 2, 3, 4, 4, 5, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 1, 1, 2, 3, 4, 1, 2, 1, 2, 3, 1, 1, 2, 1, 2, 5, 2, 3, 4, 5, 4, 5, 5, 4, 5, 3, 4, 5, 4, 5, 4, 5, 5, 5, 5, 4, 4, 5, 1, 5, 2, 3, 4, 2, 3, 1, 2, 5, 2, 3, 3, 3, 4, 4, 4, 3, 2, 3, 3, 4, 1, 2, 3, 1, 5, 5, 3, 4, 5, 1, 2, 1, 2, 3, 3, 2, 1, 5, 1, 2, 4, 5, 1, 3, 4, 3, 4, 2, 3, 4, 1, 5, 1, 2, 2, 1, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 2, 1, 1, 2, 1, 2, 3, 4, 1, 4, 5, 5, 4, 5, 5, 5, 3, 4, 5, 3, 4, 5, 4, 5, 3, 4, 5, 5, 5, 2, 3, 4, 2, 3, 4, 4, 2, 4, 5, 5, 2, 2, 3, 5, 1, 2, 3, 4, 4, 1, 1, 2, 3, 1, 3, 3, 1, 2, 3, 4, 1, 1, 3, 4, 4, 5, 1, 2, 3, 3, 4, 5, 4, 5, 2, 2, 4, 5, 1, 2, 3, 4, 5, 2, 2, 3, 5, 1, 2, 1, 2, 1, 2, 3, 1, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 1, 2, 3, 4))

)

#process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
#process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("FWCore.Modules.preScaler_cfi")

process.dqmInfoEE = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalEndcap')
)

process.dqmQTestEE = cms.EDAnalyzer("QualityTester",
    reportThreshold = cms.untracked.string('red'),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/EcalEndcapMonitorModule/test/data/EcalEndcapQualityTests.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    dirName = cms.untracked.string('.'),
    convention = cms.untracked.string('Online')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(150)
)
process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
#---
    fileNames = cms.untracked.vstring('/store/user/dellaric/data/h2b.00016403.A.0.0.root')
#---
#    fileNames = cms.untracked.vstring('/store/user/dellaric/data/h2b.00016404.A.0.0.root')
#---
#    fileNames = cms.untracked.vstring('/store/user/dellaric/data/h2b.00020937.A.0.0.root')
#---
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_44_V1::All"

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True),
        noTimeStamps = cms.untracked.bool(True),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalTBRawToDigi = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiTriggerType = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiTpg = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiNumTowerBlocks = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiTowerId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiTowerSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiChId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiGainZero = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiGainSwitch = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiDccBlockSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigi = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTriggerType = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTpg = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiNumTowerBlocks = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTowerId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTowerSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiChId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiGainZero = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiGainSwitch = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiDccBlockSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemBlock = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemTowerId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemChId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemGain = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTCC = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiSRP = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalDCCHeaderRuntypeDecoder = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalEndcapMonitorModule = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('EcalTBRawToDigi', 
                                       'EcalTBRawToDigiTriggerType', 
                                       'EcalTBRawToDigiTpg', 
                                       'EcalTBRawToDigiNumTowerBlocks', 
                                       'EcalTBRawToDigiTowerId', 
                                       'EcalTBRawToDigiTowerSize', 
                                       'EcalTBRawToDigiChId', 
                                       'EcalTBRawToDigiGainZero', 
                                       'EcalTBRawToDigiGainSwitch', 
                                       'EcalTBRawToDigiDccBlockSize', 
                                       'EcalRawToDigi', 
                                       'EcalRawToDigiTriggerType', 
                                       'EcalRawToDigiTpg', 
                                       'EcalRawToDigiNumTowerBlocks', 
                                       'EcalRawToDigiTowerId', 
                                       'EcalRawToDigiTowerSize', 
                                       'EcalRawToDigiChId', 
                                       'EcalRawToDigiGainZero', 
                                       'EcalRawToDigiGainSwitch', 
                                       'EcalRawToDigiDccBlockSize', 
                                       'EcalRawToDigiMemBlock', 
                                       'EcalRawToDigiMemTowerId', 
                                       'EcalRawToDigiMemChId', 
                                       'EcalRawToDigiMemGain', 
                                       'EcalRawToDigiTCC', 
                                       'EcalRawToDigiSRP', 
                                       'EcalDCCHeaderRuntypeDecoder', 
                                       'EcalEndcapMonitorModule'),
    destinations = cms.untracked.vstring('cout')
)

process.preScaler.prescaleFactor = 1

process.ecalDataSequence = cms.Sequence(process.preScaler*process.ecalEBunpacker*process.ecalUncalibHit*process.ecalRecHit)
process.ecalEndcapMonitorSequence = cms.Sequence(process.ecalEndcapMonitorModule*process.dqmInfoEE*process.ecalEndcapMonitorClient*process.dqmQTestEE)

process.p = cms.Path(process.ecalDataSequence*process.ecalEndcapMonitorSequence*process.dqmSaver)
process.q = cms.EndPath(process.ecalEndcapCosmicTasksSequence)

#process.ecalEBunpacker.FEDs = [13]
#process.ecalEBunpacker.orderedFedList = [13]
#process.ecalEBunpacker.orderedDCCIdList = [28]

process.ecalUncalibHit.MinAmplBarrel = 12.
process.ecalUncalibHit.MinAmplEndcap = 16.
process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'

process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'

process.ecalEndcapMonitorClient.location = 'H4'
process.ecalEndcapMonitorClient.superModules = [10]
#process.ecalEndcapMonitorClient.superModules = [4, 5, 6]

