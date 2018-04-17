import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Demo")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "100X_dataRun2_Express_v2",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  306054,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,           # string, int, or float
                  "run number")

options.parseArguments()


##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.categories.append("SiStripNoisesFromDBMiscalibrator")  
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripNoisesFromDBMiscalibrator = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )
process.MessageLogger.statistics.append('cout') 

process.load("Configuration.Geometry.GeometryRecoDB_cff") # Ideal geometry and interface 
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

##
## Example smearing configurations
##

# Layer: Layer.TIB1 ratio of ratios: 1.02533166915
# Layer: Layer.TIB2 ratio of ratios: 1.01521093183
# Layer: Layer.TIB3 ratio of ratios: 1.01552419364
# Layer: Layer.TIB4 ratio of ratios: 0.95224779507
# Layer: Layer.TOB1 ratio of ratios: 1.01219411074
# Layer: Layer.TOB2 ratio of ratios: 1.00835168635
# Layer: Layer.TOB3 ratio of ratios: 0.996159099354
# Layer: Layer.TOB4 ratio of ratios: 0.997676926445
# Layer: Layer.TOB5 ratio of ratios: 0.993886888572
# Layer: Layer.TOB6 ratio of ratios: 0.997490411188
# Layer: Layer.TIDP1 ratio of ratios: 1.0314881072
# Layer: Layer.TIDP2 ratio of ratios: 1.02853114088
# Layer: Layer.TIDP3 ratio of ratios: 1.0518768914
# Layer: Layer.TIDM1 ratio of ratios: 1.03421675878
# Layer: Layer.TIDM2 ratio of ratios: 1.04546785025
# Layer: Layer.TIDM3 ratio of ratios: 1.0311586591
# Layer: Layer.TECP1 ratio of ratios: 1.04989866792
# Layer: Layer.TECP2 ratio of ratios: 1.03711260343
# Layer: Layer.TECP3 ratio of ratios: 1.04297992451
# Layer: Layer.TECP4 ratio of ratios: 1.04669045804
# Layer: Layer.TECP5 ratio of ratios: 1.03838249025
# Layer: Layer.TECP6 ratio of ratios: 1.04727471357
# Layer: Layer.TECP7 ratio of ratios: 1.03632636024
# Layer: Layer.TECP8 ratio of ratios: 1.04860504406
# Layer: Layer.TECP9 ratio of ratios: 1.03398568113
# Layer: Layer.TECM1 ratio of ratios: 1.04750199121
# Layer: Layer.TECM2 ratio of ratios: 1.03771633506
# Layer: Layer.TECM3 ratio of ratios: 1.0409554129
# Layer: Layer.TECM4 ratio of ratios: 1.03630204118
# Layer: Layer.TECM5 ratio of ratios: 1.0417988699
# Layer: Layer.TECM6 ratio of ratios: 1.03864754217
# Layer: Layer.TECM7 ratio of ratios: 1.03868976393
# Layer: Layer.TECM8 ratio of ratios: 1.03942709841
# Layer: Layer.TECM9 ratio of ratios: 1.03678940814

byLayer = cms.VPSet(

    ################## TIB ##################

    cms.PSet(partition = cms.string("TIB_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.02533166915),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIB_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.01521093183),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIB_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.01552419364),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIB_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.95224779507),
             smearFactor = cms.double(0.0)
             ),

    ################## TOB ##################

    cms.PSet(partition = cms.string("TOB_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.01219411074),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.00835168635),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.996159099354),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.997676926445),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_5"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.993886888572),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_6"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.997490411188),
             smearFactor = cms.double(0.0)
             ),

    ################## TID Plus ##################

    cms.PSet(partition = cms.string("TIDP_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0314881072),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIDP_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.02853114088),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIDP_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0518768914),
             smearFactor = cms.double(0.0)
             ),

    ################## TID Minus ##################

    cms.PSet(partition = cms.string("TIDM_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03421675878),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIDM_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04546785025),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIDM_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0311586591),
             smearFactor = cms.double(0.0)
             ),
    
    ################## TEC plus ##################

    cms.PSet(partition = cms.string("TECP_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04989866792),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03711260343),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04297992451),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04669045804),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_5"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03838249025),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_6"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04727471357),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_7"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03632636024),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_8"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04860504406),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_9"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03398568113),
             smearFactor = cms.double(0.0)
             ),

    ################## TEC Minus ##################
    cms.PSet(partition = cms.string("TECM_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04750199121),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03771633506),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0409554129),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03630204118),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_5"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0417988699),
             smearFactor = cms.double(0.0)
             ),    
    cms.PSet(partition = cms.string("TECM_6"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03864754217),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_7"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0386897639),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_8"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03942709841),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_9"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03678940814),
             smearFactor = cms.double(0.0)
             )
    ) 


########### Noise ##########
# TEC: new = 5.985 old = 5.684  => ratio: 1.052
# TOB: new = 6.628 old = 6.647  => ratio: 0.997
# TIB: new = 5.491 old = 5.392  => ratio: 1.018
# TID: new = 5.259 old = 5.080  => ratio: 1.035

##
## separately partition by partition
##
byPartition = cms.VPSet(
    cms.PSet(partition = cms.string("TEC"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.2),
             smearFactor = cms.double(0.04)
             ),
    cms.PSet(partition = cms.string("TOB"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.94),
             smearFactor = cms.double(0.03)
             ),
    cms.PSet(partition = cms.string("TIB"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.1),
             smearFactor = cms.double(0.02)
             ),
    cms.PSet(partition = cms.string("TID"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.22),
             smearFactor = cms.double(0.01)
             )
    )

##
## whole Strip tracker
##

wholeTracker = cms.VPSet(
    cms.PSet(partition = cms.string("Tracker"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.00),
             smearFactor = cms.double(0.00)
             )
    )

byLayer2 = cms.VPSet(

    ################## TIB ##################

    cms.PSet(partition = cms.string("TIB_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1),
             smearFactor = cms.double(0.1)
             ),
    cms.PSet(partition = cms.string("TIB_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(2),
             smearFactor = cms.double(0.25)
             ),
    cms.PSet(partition = cms.string("TIB_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(3),
             smearFactor = cms.double(0.2)
             ),
    cms.PSet(partition = cms.string("TIB_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(4),
             smearFactor = cms.double(0.01)
             )
    )

subsets =  cms.VPSet(
    cms.PSet(partition = cms.string("Tracker"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(0.65),
             smearFactor = cms.double(0.05)
             ),
    cms.PSet(partition = cms.string("TEC"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.15),
             smearFactor = cms.double(0.02)
             ),
    cms.PSet(partition = cms.string("TECP"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.35),
             smearFactor = cms.double(0.02)
             ),
    cms.PSet(partition = cms.string("TECP_9"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.55),
             smearFactor = cms.double(0.02)
             )
    )


autoparams=[]
listOfLayers=["TIB_1","TIB_2","TIB_3","TIB_4","TOB_1","TOB_2","TOB_3","TOB_4","TOB_5","TOB_6","TIDM_1","TIDM_2","TIDM_3","TECM_1","TECM_2","TECM_3","TECM_4","TECM_5","TECM_6","TECM_7","TECM_8","TECM_9","TIDP_1","TIDP_2","TIDP_3","TECP_1","TECP_2","TECP_3","TECP_4","TECP_5","TECP_6","TECP_7","TECP_8","TECP_9"]

for i,ll in enumerate(listOfLayers):
    autoparams.append(
        cms.PSet(
            partition = cms.string(ll),
            doScale   = cms.bool(True),
            doSmear   = cms.bool(True),
            scaleFactor = cms.double(i*0.1),
            smearFactor = cms.double((len(listOfLayers)-i)*0.01)
            )
        )


# process.demo = cms.EDAnalyzer('SiStripChannelGainFromDBMiscalibrator',
#                               record = cms.untracked.string("SiStripApvGainRcd"),
#                               gainType = cms.untracked.uint32(1), #0 for G1, 1 for G2
#                               params = subsets # as a cms.VPset
#                               )

process.load("CondTools.SiStrip.scaleAndSmearSiStripNoises_cfi")
#process.scaleAndSmearSiStripNoises.params  = wholeTracker  # as a cms.VPset
#process.scaleAndSmearSiStripNoises.params  = byPartition   # as a cms.VPset
#process.scaleAndSmearSiStripNoises.params = subsets        # as a cms.VPset
#process.scaleAndSmearSiStripNoises.params = byLayer2       # as a cms.VPset
process.scaleAndSmearSiStripNoises.params  = autoparams
process.scaleAndSmearSiStripNoises.fillDefaults = True     # to fill uncabled DetIds with default

##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:modifiedNoise_'+options.globalTag+'_IOV_'+str(options.runNumber)+".db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiStripNoisesRcd'),
                                                                     tag = cms.string('modifiedNoise')
                                                                     )
                                                            )
                                          )

process.p = cms.Path(process.scaleAndSmearSiStripNoises)
