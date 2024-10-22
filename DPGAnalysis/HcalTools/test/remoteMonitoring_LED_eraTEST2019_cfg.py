#eoscms ls -l /eos/cms/store/group/dpg_hcal/comm_hcal/USC/
# choose run in /store/group/dpg_hcal/comm_hcal/USC/
#
# with GT: 104X_dataRun2_v1
#how to run: cmsRun remoteMonitoring_LED_eraTEST2019_cfg.py 320117 /store/group/dpg_hcal/comm_hcal/USC/ your_working_dir 
#
# with GT: 106X_dataRun3_HLT_v3
#how to run: cmsRun remoteMonitoring_LED_eraTEST2019_cfg.py 331370 /store/group/dpg_hcal/comm_hcal/USC/ your_working_dir
#
#
import sys
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
#process = cms.Process("TEST", eras.Run2_2018)
process = cms.Process("TEST", eras.Run3)
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("CondCore.CondDB.CondDB_cfi")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.l1GtUnpack.DaqGtInputTag = 'source'
# from RelValAlCaPedestal_cfg_2018.py
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
#process.load('RecoLocalCalo.Configuration.hcalLocalReco_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

runnumber = sys.argv[1]
rundir = sys.argv[2]
histodir = sys.argv[3]

#print 'RUN = '+runnumber
#print 'Input file = '+rundir+'/run'+runnumber+'/USC_'+runnumber+'.root'
##print 'Input file = '+rundir+'/USC_'+runnumber+'.root'
#print 'Output file = '+histodir+'/LED_'+runnumber+'.root'

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
  input = cms.untracked.int32(-1)
  )

# process.TFileService = cms.Service("TFileService",
#     fileName = cms.string('histoTFileService.root'),
#     closeFileFast = cms.untracked.bool(False)
# )
#process.TFileService = cms.Service('TFileService',
#   fileName=cms.string("testhisto.root"),
# closeFileFast = cms.untracked.bool(True)
#)
#process.TFileService = cms.Service("TFileService",
#      fileName = cms.string("analysis_minbias_Full.root"),
#      closeFileFast = cms.untracked.bool(True)
#  )
process.TFileService = cms.Service("TFileService",
      fileName = cms.string(histodir+'/LED_'+runnumber+'.root')
#      ,closeFileFast = cms.untracked.bool(True)
  )


# process.TFileService = cms.Service("TFileService", fileName = cms.string("plotting%03d.root" % jobnumber))

#process.source = cms.Source("PoolSource",
process.source = cms.Source("HcalTBSource",
                            skipBadFiles=cms.untracked.bool(True),
                            firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID([]),
                            firstRun = cms.untracked.uint32(331370),
#                            firstRun = cms.untracked.uint32(330153),
#                            firstRun = cms.untracked.uint32(329416),
                            fileNames = cms.untracked.vstring(
rundir+'/run'+runnumber+'/USC_'+runnumber+'.root'
#rundir+'/USC_'+runnumber+'.root'
#                       '/store/group/dpg_hcal/comm_hcal/USC/run331370/USC_331370.root'

), 
                            secondaryFileNames = cms.untracked.vstring()
                            )

#process.Analyzer = cms.EDAnalyzer("VeRawAnalyzer",
process.Analyzer = cms.EDAnalyzer("CMTRawAnalyzer",
                                  #
                                  Verbosity = cms.untracked.int32(0),
                                  #
                                  MapCreation = cms.untracked.int32(1),
                                  #
                                  recordNtuples = cms.untracked.bool(False),
                                  maxNeventsInNtuple = cms.int32(1),
                                  recordHistoes = cms.untracked.bool(True),
                                  studyRunDependenceHist = cms.untracked.bool(True),
                                  studyCapIDErrorsHist = cms.untracked.bool(True),
                                  studyRMSshapeHist = cms.untracked.bool(True),
                                  studyRatioShapeHist = cms.untracked.bool(True),
                                  studyADCAmplHist = cms.untracked.bool(True),
                                  studyTSmeanShapeHist = cms.untracked.bool(True),
                                  studyTSmaxShapeHist = cms.untracked.bool(True),
                                  studyCalibCellsHist = cms.untracked.bool(True),
                                  studyDiffAmplHist = cms.untracked.bool(True),
                                  studyPedestalsHist = cms.untracked.bool(True),
                                  studyPedestalCorrelations = cms.untracked.bool(True),
                                  #         Normal channels:
                                  ratioHBMin = cms.double(0.70),
                                  ratioHBMax = cms.double(0.94),
                                  ratioHEMin = cms.double(0.60),
                                  ratioHEMax = cms.double(0.95),
                                  ratioHFMin = cms.double(0.45),
                                  ratioHFMax = cms.double(1.02),
                                  ratioHOMin = cms.double(0.40),
                                  ratioHOMax = cms.double(1.04),
                                  rmsHBMin = cms.double(0.7),
                                  rmsHBMax = cms.double(2.5),
                                  rmsHEMin = cms.double(0.7),
                                  rmsHEMax = cms.double(2.2),
                                  rmsHFMin = cms.double(0.1),
                                  rmsHFMax = cms.double(2.6),
                                  rmsHOMin = cms.double(0.1),
                                  rmsHOMax = cms.double(2.8),
                                  TSmeanHBMin = cms.double(2.5),
                                  TSmeanHBMax = cms.double(5.5),
                                  TSmeanHEMin = cms.double(1.0),
                                  TSmeanHEMax = cms.double(5.2),
                                  TSmeanHFMin = cms.double(1.0),
                                  TSmeanHFMax = cms.double(4.2),
                                  TSmeanHOMin = cms.double(1.0),
                                  TSmeanHOMax = cms.double(4.8),
                                  TSpeakHBMin = cms.double(2.2),
                                  TSpeakHBMax = cms.double(5.5),
                                  TSpeakHEMin = cms.double(1.5),
                                  TSpeakHEMax = cms.double(6.5),
                                  TSpeakHFMin = cms.double(0.5),
                                  TSpeakHFMax = cms.double(4.5),
                                  TSpeakHOMin = cms.double(0.5),
                                  TSpeakHOMax = cms.double(7.5),
                                  ADCAmplHBMin = cms.double(10000.),
                                  ADCAmplHBMax = cms.double(300000.),
                                  ADCAmplHEMin = cms.double(20000.),  
                                  ADCAmplHEMax = cms.double(300000.),
                                  ADCAmplHFMin = cms.double(50.),
                                  ADCAmplHFMax = cms.double(9000.),
                                  ADCAmplHOMin = cms.double(50.),
                                  ADCAmplHOMax = cms.double(9000.),
                                  pedestalwHBMax = cms.double(0.1),
                                  pedestalwHEMax = cms.double(0.1),
                                  pedestalwHFMax = cms.double(0.4),
                                  pedestalwHOMax = cms.double(0.1),
                                  pedestalHBMax = cms.double(0.1),
                                  pedestalHEMax = cms.double(0.6),
                                  pedestalHFMax = cms.double(0.8),
                                  pedestalHOMax = cms.double(0.1),
                                  #             CALIBRATION channels:
                                  calibrADCHBMin = cms.double(1000.),
				  calibrADCHBMax = cms.double(100000000.),
                                  calibrADCHEMin = cms.double(1000.),
				  calibrADCHEMax = cms.double(100000000.),
                                  calibrADCHOMin = cms.double(1000.),
				  calibrADCHOMax = cms.double(100000000.),
                                  calibrADCHFMin = cms.double(100.),
				  calibrADCHFMax = cms.double(100000000.),
                                  calibrRatioHBMin = cms.double(0.76),
				  calibrRatioHBMax = cms.double(0.94),
                                  calibrRatioHEMin = cms.double(0.76),
				  calibrRatioHEMax = cms.double(0.94),
                                  calibrRatioHOMin = cms.double(0.85),
				  calibrRatioHOMax = cms.double(0.99),
                                  calibrRatioHFMin = cms.double(0.5),
				  calibrRatioHFMax = cms.double(0.8),
                                  calibrTSmaxHBMin = cms.double(1.50),
                                  calibrTSmaxHBMax = cms.double(2.50),
                                  calibrTSmaxHEMin = cms.double(1.50),
                                  calibrTSmaxHEMax = cms.double(2.50),
                                  calibrTSmaxHOMin = cms.double(1.50),
                                  calibrTSmaxHOMax = cms.double(2.50),
                                  calibrTSmaxHFMin = cms.double(3.50),
                                  calibrTSmaxHFMax = cms.double(4.50),
                                  calibrTSmeanHBMin = cms.double(2.40),
                                  calibrTSmeanHBMax = cms.double(3.70),
                                  calibrTSmeanHEMin = cms.double(2.40),
                                  calibrTSmeanHEMax = cms.double(3.70),
                                  calibrTSmeanHOMin = cms.double(1.50),
                                  calibrTSmeanHOMax = cms.double(2.70),
                                  calibrTSmeanHFMin = cms.double(3.50),
                                  calibrTSmeanHFMax = cms.double(4.50),
                                  calibrWidthHBMin = cms.double(1.30),
                                  calibrWidthHBMax = cms.double(1.90),
                                  calibrWidthHEMin = cms.double(1.30),
                                  calibrWidthHEMax = cms.double(1.90),
                                  calibrWidthHOMin = cms.double(0.70),
                                  calibrWidthHOMax = cms.double(1.65),
                                  calibrWidthHFMin = cms.double(0.30),
                                  calibrWidthHFMax = cms.double(1.50),
                                  # Special task of run or LS quality:
                                  #
                                  # flag for ask runs of LSs for RMT & CMT accordingly:
                                  #=0-runs, =1-LSs
                                  # keep for LED runs this flags =0 always
                                  flagtoaskrunsorls = cms.int32(0),
                                  #
                                  # flag for choice of criterion of bad channels:
                                  #=0-CapIdErr, =1-Ratio, =2-Width, =3-TSmax, =4-TSmean, =5-adcAmplitud
                                  # keep for CMT (global runs) this flags =0 always
                                  flagtodefinebadchannel = cms.int32(0),
                                  #how many bins you want on the plots:better to choice (#LS+1)
                                  howmanybinsonplots = cms.int32(25),
                                  #
                                  # ls - range for RBX study (and ??? perhaps for gain stability via abort gap):
                                  lsmin = cms.int32(1),
                                  #lsmax = cms.int32(620),
                                  lsmax = cms.int32(2600),
                                  #
                                  flagabortgaprejected = cms.int32(1),
                                  bcnrejectedlow = cms.int32(3446),
                                  bcnrejectedhigh= cms.int32(3564),
                                  #
                                  # flag cpu time reducing
                                  #=0-all plots, =1-optimized number of plots (for Global runs)
                                  flagcpuoptimization = cms.int32(0),
                                  #
                                  # flag for ask type of Normalization for CMT estimators:
                                  #=0-normalizationOn#evOfLS;   =1-averageVariable-normalizationOn#entriesInLS;
                                  flagestimatornormalization = cms.int32(1),
                                  #
                                  #
                                  # cuts on Nbadchannels to see LS dependences:
                                  # Verbosity = cms.untracked.int32(-77),
                                  # to select abnormal events,for which Nbcs > this limits
                                  lsdep_cut1_peak_HBdepth1 = cms.int32(20),
                                  lsdep_cut1_peak_HBdepth2 = cms.int32(7),
                                  lsdep_cut1_peak_HEdepth1 = cms.int32(16),
                                  lsdep_cut1_peak_HEdepth2 = cms.int32(13),
                                  lsdep_cut1_peak_HEdepth3 = cms.int32(4),
                                  lsdep_cut1_peak_HFdepth1 = cms.int32(10),
                                  lsdep_cut1_peak_HFdepth2 = cms.int32(5),
                                  lsdep_cut1_peak_HOdepth4 = cms.int32(45),
                                  # to select events with Nbcs > this limits
                                  lsdep_cut3_max_HBdepth1 = cms.int32(19),
                                  lsdep_cut3_max_HBdepth2 = cms.int32(6),
                                  lsdep_cut3_max_HEdepth1 = cms.int32(15),
                                  lsdep_cut3_max_HEdepth2 = cms.int32(12),
                                  lsdep_cut3_max_HEdepth3 = cms.int32(3),
                                  lsdep_cut3_max_HFdepth1 = cms.int32(9),
                                  lsdep_cut3_max_HFdepth2 = cms.int32(4),
                                  lsdep_cut3_max_HOdepth4 = cms.int32(40),
                                  #Verbosity = cms.untracked.int32(-79),
                                  # cuts on Estimator1 to see LS dependences:
                                  lsdep_estimator1_HBdepth1 = cms.double(2500.),
                                  lsdep_estimator1_HBdepth2 = cms.double(2500.),
                                  lsdep_estimator1_HBdepth3 = cms.double(2500.),
                                  lsdep_estimator1_HBdepth4 = cms.double(2500.),
                                  lsdep_estimator1_HEdepth1 = cms.double(2500.),
                                  lsdep_estimator1_HEdepth2 = cms.double(2500.),
                                  lsdep_estimator1_HEdepth3 = cms.double(2500.),
                                  lsdep_estimator1_HEdepth4 = cms.double(2500.),
                                  lsdep_estimator1_HEdepth5 = cms.double(2500.),
                                  lsdep_estimator1_HEdepth6 = cms.double(2500.),
                                  lsdep_estimator1_HEdepth7 = cms.double(2500.),
                                  lsdep_estimator1_HFdepth1 = cms.double(2500.),
                                  lsdep_estimator1_HFdepth2 = cms.double(2500.),
                                  lsdep_estimator1_HFdepth3 = cms.double(2500.),
                                  lsdep_estimator1_HFdepth4 = cms.double(2500.),
                                  lsdep_estimator1_HOdepth4 = cms.double(2500.),
                                  # cuts on Estimator2 to see LS dependences:
                                  lsdep_estimator2_HBdepth1 = cms.double(7.),
                                  lsdep_estimator2_HBdepth2 = cms.double(7.),
                                  lsdep_estimator2_HEdepth1 = cms.double(7.),
                                  lsdep_estimator2_HEdepth2 = cms.double(7.),
                                  lsdep_estimator2_HEdepth3 = cms.double(7.),
                                  lsdep_estimator2_HFdepth1 = cms.double(7.),
                                  lsdep_estimator2_HFdepth2 = cms.double(7.),
                                  lsdep_estimator2_HOdepth4 = cms.double(7.),
                                  # cuts on Estimator3 to see LS dependences:
                                  lsdep_estimator3_HBdepth1 = cms.double(7.),
                                  lsdep_estimator3_HBdepth2 = cms.double(7.),
                                  lsdep_estimator3_HEdepth1 = cms.double(7.),
                                  lsdep_estimator3_HEdepth2 = cms.double(7.),
                                  lsdep_estimator3_HEdepth3 = cms.double(7.),
                                  lsdep_estimator3_HFdepth1 = cms.double(7.),
                                  lsdep_estimator3_HFdepth2 = cms.double(7.),
                                  lsdep_estimator3_HOdepth4 = cms.double(7.),
                                  # cuts on Estimator4 to see LS dependences:
                                  lsdep_estimator4_HBdepth1 = cms.double(5.),
                                  lsdep_estimator4_HBdepth2 = cms.double(5.),
                                  lsdep_estimator4_HEdepth1 = cms.double(5.),
                                  lsdep_estimator4_HEdepth2 = cms.double(5.),
                                  lsdep_estimator4_HEdepth3 = cms.double(5.),
                                  lsdep_estimator4_HFdepth1 = cms.double(5.),
                                  lsdep_estimator4_HFdepth2 = cms.double(5.),
                                  lsdep_estimator4_HOdepth4 = cms.double(5.),
                                  # cuts on Estimator5 to see LS dependences:
                                  lsdep_estimator5_HBdepth1 = cms.double(1.8),
                                  lsdep_estimator5_HBdepth2 = cms.double(1.8),
                                  lsdep_estimator5_HEdepth1 = cms.double(1.8),
                                  lsdep_estimator5_HEdepth2 = cms.double(1.8),
                                  lsdep_estimator5_HEdepth3 = cms.double(1.8),
                                  lsdep_estimator5_HFdepth1 = cms.double(1.8),
                                  lsdep_estimator5_HFdepth2 = cms.double(1.8),
                                  lsdep_estimator5_HOdepth4 = cms.double(1.8),
                                  #
                                  # use ADC amplitude:
                                  useADCmassive = cms.untracked.bool(True),
                                  useADCfC = cms.untracked.bool(False),
                                  useADCcounts = cms.untracked.bool(False),
                                  # 
                                  # Pedestals in fC
                                  #usePedestalSubtraction = cms.untracked.bool(True),
                                  usePedestalSubtraction = cms.untracked.bool(False),
                                  #
                                  # for possible ignoring of channels w/o signal, apply same cut for
                                  # HBHEHFHO on Amplitude, usable for all Estimators 1,2,3,4,5:
                                  # forallestimators_amplitude_bigger = cms.double(10.),
                                  forallestimators_amplitude_bigger = cms.double(-100.),
                                  #
                                  #usecontinuousnumbering = cms.untracked.bool(False),
                                  usecontinuousnumbering = cms.untracked.bool(True),
                                  #
                                  #
                                  #
                                  #
                                  # if 0 - do not use digis at all
                                  flagToUseDigiCollectionsORNot = cms.int32(1),
                                  #
                                  #
                                  #
                                  #
                                  #
                                  hcalCalibDigiCollectionTag = cms.InputTag('hcalDigis'),
                                  hbheDigiCollectionTag = cms.InputTag('hcalDigis'),
                                  hoDigiCollectionTag = cms.InputTag('hcalDigis'),
                                  hfDigiCollectionTag = cms.InputTag('hcalDigis'),
                                  #
                                  #for upgrade: ---------------------------------------------------------
                                  hbheQIE11DigiCollectionTag = cms.InputTag('hcalDigis'),
                                  hbheQIE10DigiCollectionTag = cms.InputTag('hcalDigis'),
                                  # flag to use either only old QIE8 digiCollections or only new QIE10,11 digiCollections
                                  #=0-all digiCollections(default for normal running), =1-only old QIE8 digiCollections, 
                                  #=2-only new QIE1011 digiCollections, =3-only new QIE1011 digiCollections w/o new high depthes
                                  #=4-2016fall, =5-2016fall w/o new high depthes, =6-2017bebin, =7-2017bebin w/o new high depthes in HEonly
                                  #=8--2017bebin w/o new high depthes, =9-all digiCollections  w/o new high depthes
                                  # flag   HBHE8    HBHE11   HF8   HF10  comments:
                                  #  0       +        +       +     +     all
                                  #  1       +        -       +     -     old
                                  #  2       -        +       -     +     new
                                  #  3       -        +       -     +     new w/o high depthes
                                  #  4       +        -       +     +     2016fall
                                  #  5       +        -       +     +     2016fall w/o high depthes
                                  #  6       +        +       -     +     2017 && 2018 && 2021
                                  #  7       +        +       -     +     2017begin w/o high depthes in HEonly
                                  #  8       +        +       -     +     2017begin w/o high depthes
                                  #  9       +        +       +     +     all  w/o high depthes
                                  # 10       +        -       -     +     2017 w/o HEP17
                                  # 
                                  flagupgradeqie1011 = cms.int32(6),
                                  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                                  # flaguseshunt = 1 or 6 (6 is default for global runs) 
                                  flaguseshunt = cms.int32(6),
                                  # flagsipmcorrection: != 0 yes,apply; = 0 do not use;
                                  flagsipmcorrection = cms.int32(1),
                                  #end upgrade: --------------------------------------------------------- end upgrade
                                  #
                                  #
                                  # for local LASER runs ONLY!!! to be > 0    (,else = 0)
                                  flagLaserRaddam = cms.int32(0),
                                  # for gaussian fit for local shunt1 (Gsel0) led low-intensity or ped ONLY!!! to be  > 0    (,else = 0)
                                  flagfitshunt1pedorledlowintensity = cms.int32(0),
                                  #
                                  splashesUpperLimit = cms.int32(10000),
                                  #
                                  #
                                  # for use in IterativeMethod of CalibrationGroup!!! to be > 1    (,else = 0) This flag name is only present in early version of main cc-code & visualiz.script
                                  #flagIterativeMethodCalibrationGroup = cms.int32(0),
                                  #
                                  #
                                  # for use in IterativeMethod of CalibrationGroup!!! to be > 1    (,else = 0)
                                  flagIterativeMethodCalibrationGroupDigi = cms.int32(0),
                                  #
                                  # for use in IterativeMethod of CalibrationGroup!!! to be > 1    (,else = 0)
                                  flagIterativeMethodCalibrationGroupReco = cms.int32(0),
                                  #
                                  hbheInputSignalTag = cms.InputTag('hbherecoMBNZS'),
                                  hbheInputNoiseTag = cms.InputTag('hbherecoNoise'),
                                  hfInputSignalTag = cms.InputTag('hfrecoMBNZS'),
                                  hfInputNoiseTag = cms.InputTag('hfrecoNoise'),
                                  #
                                  #HistOutFile = cms.untracked.string('LED_331370.root'),
                                  #HistOutFile = cms.untracked.string(histodir+'/LED_'+runnumber+'.root'),
                                  #MAPOutFile = cms.untracked.string('LogEleMapdb.h')
                                  #
                                  ##OutputFilePath = cms.string('/tmp/zhokin/'),        
                                  ##OutputFileExt = cms.string(''),
                                  #
                                  )		
##################################################################################################
process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)
##################################################################################################
process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('QIEShape',
        'QIEData',
        'ChannelQuality',
        'HcalQIEData',
        'Pedestals',
        'PedestalWidths',
        'Gains',
        'GainWidths',
        'ZSThresholds',
        'RespCorrs')
)
##################################################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
# 2018:
process.GlobalTag.globaltag = '104X_dataRun2_v1'
######process.GlobalTag.globaltag = '105X_postLS2_design_v4'
# 2019:
#process.GlobalTag.globaltag = '106X_dataRun3_HLT_v3'
##################################################################################################
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.hcalDigis.FilterDataQuality = cms.bool(False)
process.hcalDigis.InputLabel = cms.InputTag("source")
############################################################################
#process.hcalDigis= cms.EDProducer("HcalRawToDigi",
#    FilterDataQuality = cms.bool(False),
#    HcalFirstFED = cms.untracked.int32(700),
#    InputLabel = cms.InputTag("source"),
#)
process.load('EventFilter.HcalRawToDigi.hcalRawToDigi_cfi')
process.hcalDigis= process.hcalRawToDigi.clone(
    FilterDataQuality = False,
    InputLabel = "source",
    #InputLabel = cms.InputTag("rawDataCollector"),
)
##################################################################################################
#process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalPedestalLocal_cff")
process.p = cms.Path(process.hcalDigis*process.Analyzer)
#process.p = cms.Path(process.hcalDigis*process.seqALCARECOHcalCalMinBiasDigiNoHLT*process.seqALCARECOHcalCalMinBias*process.Analyzer)
##################################################################################################
process.MessageLogger = cms.Service("MessageLogger",
     categories   = cms.untracked.vstring(''),
     destinations = cms.untracked.vstring('cout'),
     debugModules = cms.untracked.vstring('*'),
     cout = cms.untracked.PSet(
         threshold = cms.untracked.string('WARNING'),
	 WARNING = cms.untracked.PSet(limit = cms.untracked.int32(0))
     )
 )
##################################################################################################





