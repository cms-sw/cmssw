import FWCore.ParameterSet.Config as cms

PixelCPEGenericESProducer = cms.ESProducer("PixelCPEGenericESProducer",

    ComponentName = cms.string('PixelCPEGeneric'),
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

    # new parameters added in 1/14, dk
    # LA defined by hand, FOR TESTING ONLY, not for production   
    # 0.0 means that the offset is taken from DB        
    #lAOffset = cms..double(0.0),
    #lAWidthBPix = cms.double(0.0),
    #lAWidthFPix = cms.double(0.0),

    # Flag to select the source of LA-Width
    # Normal = True, use LA from DB
    useLAWidthFromDB = cms.bool(True),                             
    # if lAWith=0 and useLAWidthFromDB=false than width is calculated from lAOffset.         
    # Use the LA-Offsets from Alignment instead of our calibration
    useLAAlignmentOffsets = cms.bool(False),                             
                                           
    #MagneticFieldRecord: e.g. "" or "ParabolicMF"
    MagneticFieldRecord = cms.ESInputTag(""),
)

# This customizes the Run3 Pixel CPE generic reconstruction in order to activate the IrradiationBiasCorrection
# because of the expected resolution loss due to radiation damage
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(PixelCPEGenericESProducer, IrradiationBiasCorrection = True)

# This customization will be removed once we get the templates for phase2 pixel
# FIXME::Is the Upgrade variable actually used?
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(PixelCPEGenericESProducer, 
  UseErrorsFromTemplates = False,
  LoadTemplatesFromDB = False,
  TruncatePixelCharge = False,
  IrradiationBiasCorrection = False,
  DoCosmics = False,
  Upgrade = cms.bool(True)
)
