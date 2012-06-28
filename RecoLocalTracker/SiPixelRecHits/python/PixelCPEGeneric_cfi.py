import FWCore.ParameterSet.Config as cms

PixelCPEGenericESProducer = cms.ESProducer("PixelCPEGenericESProducer",

    ComponentName = cms.string('PixelCPEGeneric'),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    PixelErrorParametrization = cms.string('NOTcmsim'),

    # Allows cuts to be optimized
    eff_charge_cut_lowX = cms.double(0.0),
    eff_charge_cut_lowY = cms.double(0.0),
    eff_charge_cut_highX = cms.double(1.0),
    eff_charge_cut_highY = cms.double(1.0),
    size_cutX = cms.double(3.0),
    size_cutY = cms.double(3.0),

    # Edge cluster errors in microns (determined by looking at residual RMS) 
    EdgeClusterErrorX = cms.double( 50.0 ),                                      
    EdgeClusterErrorY = cms.double( 85.0 ),                                                     

    # ggiurgiu@jhu.edu
    inflate_errors = cms.bool(False),
    inflate_all_errors_no_trk_angle = cms.bool(False),

    # Can use errors predicted by the template code
    # If UseErrorsFromTemplates is False, must also set
    # TruncatePixelCharge, IrradiationBiasCorrection, DoCosmics and LoadTemplatesFromDB to be False                                        
    UseErrorsFromTemplates = cms.bool(True),

    # When set True this gives a slight improvement in resolution at no cost 
    TruncatePixelCharge = cms.bool(True),

    # Turn this ON later
    IrradiationBiasCorrection = cms.bool(False),                                       

    # When set to True we use templates with extended angular acceptance   
    DoCosmics = cms.bool(False),                                      

    LoadTemplatesFromDB = cms.bool(True),                                       

    # petar, for clusterProbability() from TTRHs
    ClusterProbComputationFlag = cms.int32(0),         

    Upgrade = cms.bool(False),
    SmallPitch = cms.bool(False)
                                           
)


