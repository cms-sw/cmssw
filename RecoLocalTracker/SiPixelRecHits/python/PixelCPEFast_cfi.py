import FWCore.ParameterSet.Config as cms

PixelCPEFastESProducer = cms.ESProducer("PixelCPEFastESProducer",

    ComponentName = cms.string('PixelCPEFast'),
    Alpha2Order = cms.bool(True),

    # Edge cluster errors in microns (determined by looking at residual RMS) 
    EdgeClusterErrorX = cms.double( 50.0 ),                                      
    EdgeClusterErrorY = cms.double( 85.0 ),                                                     

    # these for CPEBase
    useLAWidthFromDB = cms.bool(True),
    useLAAlignmentOffsets = cms.bool(False),


    # Can use errors predicted by the template code
    # If UseErrorsFromTemplates is False, must also set
    # TruncatePixelCharge and LoadTemplatesFromDB to be False                                        
    UseErrorsFromTemplates = cms.bool(True),
    LoadTemplatesFromDB = cms.bool(True),

    # When set True this gives a slight improvement in resolution at no cost 
    TruncatePixelCharge = cms.bool(True),

    # petar, for clusterProbability() from TTRHs
    ClusterProbComputationFlag = cms.int32(0),

    #MagneticFieldRecord: e.g. "" or "ParabolicMF"
    MagneticFieldRecord = cms.ESInputTag(""),
)
