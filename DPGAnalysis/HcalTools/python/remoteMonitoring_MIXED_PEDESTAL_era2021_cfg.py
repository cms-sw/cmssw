#eoscms ls -l /eos/cms/store/group/dpg_hcal/comm_hcal/USC/run327785/USC_327785.root
# choose run in /store/group/dpg_hcal/comm_hcal/USC/
#how to run: cmsRun remoteMonitoring_PEDESTAL_era2019_cfg.py 331301 /store/group/dpg_hcal/comm_hcal/USC/ /afs/cern.ch/work/z/zhokin/hcal/voc2/CMSSW_11_1_0_pre3/src/DPGAnalysis/HcalTools/scripts/rmt

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
#print 'Output file = '+histodir+'/PEDESTAL_'+runnumber+'.root'

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
  input = cms.untracked.int32(1000)
  )

process.TFileService = cms.Service("TFileService",
      fileName = cms.string(histodir+'/MIXED_PEDESTAL_'+runnumber+'.root')
#      ,closeFileFast = cms.untracked.bool(True)
  )

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

process.Analyzer = cms.EDAnalyzer("CMTRawAnalyzer",
                                  #
                                  Verbosity = cms.untracked.int32(0),
                                  #Verbosity = cms.untracked.int32(-9062),
                                  #Verbosity = cms.untracked.int32(-9063),
                                  #Verbosity = cms.untracked.int32(-9064),
                                  #Verbosity = cms.untracked.int32(-9065),
                                  #Verbosity = cms.untracked.int32(-84),
                                  #Verbosity = cms.untracked.int32(-91),
                                  #Verbosity = cms.untracked.int32(-92),
                                  #
                                  MapCreation = cms.untracked.int32(1),
                                  #
                                  recordNtuples = cms.untracked.bool(False),
                                  #recordNtuples = cms.untracked.bool(True),
                                  maxNeventsInNtuple = cms.int32(1),
                                  #
                                  #recordHistoes = cms.untracked.bool(False),
                                  recordHistoes = cms.untracked.bool(True),
                                  #
                                  ##scripts: zRunRatio34.C, zRunNbadchan.C
                                  studyRunDependenceHist = cms.untracked.bool(True),
                                  #studyRunDependenceHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: zerrors.C
                                  studyCapIDErrorsHist = cms.untracked.bool(True),
                                  #studyCapIDErrorsHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: zrms.C
                                  studyRMSshapeHist = cms.untracked.bool(True),
                                  #studyRMSshapeHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: zratio34.C
                                  studyRatioShapeHist = cms.untracked.bool(True),
                                  #studyRatioShapeHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: zadcamplitude.C
                                  studyADCAmplHist = cms.untracked.bool(True),
                                  #studyADCAmplHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: ztsmean.C
                                  studyTSmeanShapeHist = cms.untracked.bool(True),
                                  #studyTSmeanShapeHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: ztsmaxa.C
                                  studyTSmaxShapeHist = cms.untracked.bool(True),
                                  #studyTSmaxShapeHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: zcalib....C
                                  studyCalibCellsHist = cms.untracked.bool(True),
                                  #studyCalibCellsHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: zdifampl.C
                                  studyDiffAmplHist = cms.untracked.bool(True),
                                  #studyDiffAmplHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: zadcamplitude.C
                                  studyPedestalsHist = cms.untracked.bool(True),
                                  #studyPedestalsHist = cms.untracked.bool(False),
                                  #
                                  ##scripts: zamplpedcorr.C
                                  studyPedestalCorrelations = cms.untracked.bool(True),
                                  #studyPedestalsHist = cms.untracked.bool(False),
                                  #
                                  #
                                  ##DigiCollectionLabel = cms.untracked.InputTag("hcalDigis"),
                                  #Verbosity = cms.untracked.int32(-54),
                                  #Verbosity = cms.untracked.int32(-22),
                                  #Verbosity = cms.untracked.int32(-11),
                                  #Verbosity = cms.untracked.int32(-12),
                                  #Verbosity = cms.untracked.int32(-13),
                                  #Verbosity = cms.untracked.int32(-51),
                                  #Verbosity = cms.untracked.int32(-24),
                                  #Verbosity = cms.untracked.int32(-244),
                                  #Verbosity = cms.untracked.int32(-233),
                                  #
                                  #
                                  #         Normal channels:
                                  #
                                  # -53 for  BAD HBHEHF channels from study on shape Ratio
                                  #Verbosity = cms.untracked.int32(-53),
                                  ratioHBMin = cms.double(0.31),
                                  ratioHBMax = cms.double(0.95),
                                  ratioHEMin = cms.double(0.31),
                                  ratioHEMax = cms.double(1.00),
                                  ratioHFMin = cms.double(0.05),
                                  ratioHFMax = cms.double(0.98),
                                  ratioHOMin = cms.double(0.15),
                                  ratioHOMax = cms.double(1.00),
                                  # -54 for  BAD HBHEHF channels from study on RMS of shapes
                                  #Verbosity = cms.untracked.int32(-54),
                                  rmsHBMin = cms.double(2.7),
                                  rmsHBMax = cms.double(3.0),
                                  rmsHEMin = cms.double(2.7),
                                  rmsHEMax = cms.double(3.0),
                                  rmsHFMin = cms.double(0.2),
                                  rmsHFMax = cms.double(5.0),
                                  rmsHOMin = cms.double(2.7),
                                  rmsHOMax = cms.double(3.0),
                                  # -55 for  BAD HBHEHF channels from study on TSmean of shapes
                                  #Verbosity = cms.untracked.int32(-55),
                                  TSmeanHBMin = cms.double(4.5),
                                  TSmeanHBMax = cms.double(4.6),
                                  TSmeanHEMin = cms.double(4.5),
                                  TSmeanHEMax = cms.double(4.6),
                                  TSmeanHFMin = cms.double(2.0),
                                  TSmeanHFMax = cms.double(7.0),
                                  TSmeanHOMin = cms.double(4.5),
                                  TSmeanHOMax = cms.double(4.6),
                                  # -55 for  BAD HBHEHF channels from study on TSmax of shapes
                                  #Verbosity = cms.untracked.int32(-55),
                                  TSpeakHBMin = cms.double(0.5),
                                  TSpeakHBMax = cms.double(9.5),
                                  TSpeakHEMin = cms.double(0.5),
                                  TSpeakHEMax = cms.double(9.5),
                                  TSpeakHFMin = cms.double(0.5),
                                  TSpeakHFMax = cms.double(8.5),
                                  TSpeakHOMin = cms.double(0.5),
                                  TSpeakHOMax = cms.double(8.5),
                                  # -56 for  BAD HBHEHOHF channels from study on ADC Amplitude
                                  #Verbosity = cms.untracked.int32(-56),
                                  ADCAmplHBMin = cms.double(10000.),
                                  ADCAmplHBMax = cms.double(300000.),
                                  ADCAmplHEMin = cms.double(20000.),  
                                  ADCAmplHEMax = cms.double(300000.),
                                  ADCAmplHFMin = cms.double(50.),
                                  ADCAmplHFMax = cms.double(9000.),
                                  ADCAmplHOMin = cms.double(50.),
                                  ADCAmplHOMax = cms.double(9000.),
                                  #
                                  # to see channels w/ PedestalSigma < cut
                                  #Verbosity = cms.untracked.int32(-57),
                                  pedestalwHBMax = cms.double(0.1),
                                  pedestalwHEMax = cms.double(0.1),
                                  pedestalwHFMax = cms.double(0.4),
                                  pedestalwHOMax = cms.double(0.1),
                                  #
                                  # to see channels for pedestal < cut
                                  pedestalHBMax = cms.double(0.1),
                                  pedestalHEMax = cms.double(0.6),
                                  pedestalHFMax = cms.double(0.8),
                                  pedestalHOMax = cms.double(0.1),
                                  #
                                  #
                                  #             CALIBRATION channels:
                                  #
                                  # for  BAD HBHEHOHF CALIBRATION channels from study on ADC amplitude
                                  # cuts for Laser runs:
                                  #calibrADCHBMin = cms.double(15.0),
                                  #calibrADCHEMin = cms.double(15.0),
                                  #calibrADCHOMin = cms.double(15.0),
                                  #calibrADCHFMin = cms.double(15.0),
                                  # cuts for PEDESTAL runs:
                                  calibrADCHBMin = cms.double(1000.),
				  calibrADCHBMax = cms.double(100000000.),
                                  calibrADCHEMin = cms.double(1000.),
				  calibrADCHEMax = cms.double(100000000.),
                                  calibrADCHOMin = cms.double(1000.),
				  calibrADCHOMax = cms.double(100000000.),
                                  calibrADCHFMin = cms.double(100.),
				  calibrADCHFMax = cms.double(100000000.),
				  
                                  # for  BAD HBHEHOHF CALIBRATION channels from study on shape Ratio
                                  calibrRatioHBMin = cms.double(0.76),
				  calibrRatioHBMax = cms.double(0.94),
                                  calibrRatioHEMin = cms.double(0.76),
				  calibrRatioHEMax = cms.double(0.94),
                                  calibrRatioHOMin = cms.double(0.85),
				  calibrRatioHOMax = cms.double(0.99),
                                  calibrRatioHFMin = cms.double(0.5),
				  calibrRatioHFMax = cms.double(0.8),
                                  # for  BAD HBHEHOHF CALIBRATION channels from study on TSmax
                                  calibrTSmaxHBMin = cms.double(1.50),
                                  calibrTSmaxHBMax = cms.double(2.50),
                                  calibrTSmaxHEMin = cms.double(1.50),
                                  calibrTSmaxHEMax = cms.double(2.50),
                                  calibrTSmaxHOMin = cms.double(1.50),
                                  calibrTSmaxHOMax = cms.double(2.50),
                                  calibrTSmaxHFMin = cms.double(3.50),
                                  calibrTSmaxHFMax = cms.double(4.50),
                                  # for  BAD HBHEHOHF CALIBRATION channels from study on TSmean
                                  calibrTSmeanHBMin = cms.double(2.40),
                                  calibrTSmeanHBMax = cms.double(3.70),
                                  calibrTSmeanHEMin = cms.double(2.40),
                                  calibrTSmeanHEMax = cms.double(3.70),
                                  calibrTSmeanHOMin = cms.double(1.50),
                                  calibrTSmeanHOMax = cms.double(2.70),
                                  calibrTSmeanHFMin = cms.double(3.50),
                                  calibrTSmeanHFMax = cms.double(4.50),
                                  # for  BAD HBHEHOHF CALIBRATION channels from study on Width
                                  calibrWidthHBMin = cms.double(1.30),
                                  calibrWidthHBMax = cms.double(1.90),
                                  calibrWidthHEMin = cms.double(1.30),
                                  calibrWidthHEMax = cms.double(1.90),
                                  calibrWidthHOMin = cms.double(0.70),
                                  calibrWidthHOMax = cms.double(1.65),
                                  calibrWidthHFMin = cms.double(0.30),
                                  calibrWidthHFMax = cms.double(1.50),
                                  #
                                  # Special task of run or LS quality:
                                  #
                                  # flag for ask runs of LSs for RMT & CMT accordingly:
                                  #=0-runs, =1-LSs
                                  # keep for PEDESTAL runs this flags =0 always
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
                                  #
                                  #
                                  #old was for runs:
                                  #                                  nbadchannels1 = cms.int32(7),
                                  #                                  nbadchannels2 = cms.int32(12),
                                  #                                  nbadchannels3 = cms.int32(50),
                                  #
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
                                  # 
                                  #Verbosity = cms.untracked.int32(-81),
                                  #Verbosity = cms.untracked.int32(-82),
                                  #Verbosity = cms.untracked.int32(-83),
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
                                  #
                                  #
                                  #usecontinuousnumbering = cms.untracked.bool(False),
                                  usecontinuousnumbering = cms.untracked.bool(True),
                                  #
                                  # if 0 - do not use digis at all
                                  flagToUseDigiCollectionsORNot = cms.int32(1),
                                  #
                                  #
                                  hcalCalibDigiCollectionTag = cms.InputTag('hcalDigis'),
                                  hbheDigiCollectionTag = cms.InputTag('hcalDigis'),
                                  hoDigiCollectionTag = cms.InputTag('hcalDigis'),
                                  hfDigiCollectionTag = cms.InputTag('hcalDigis'),
                                  #
                                  #
                                  #
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
                                  # for use in IterativeMethod of CalibrationGroup!!! to be > 1    (,else = 0)
                                  flagIterativeMethodCalibrationGroupDigi = cms.int32(1),
                                  #
                                  # for use in IterativeMethod of CalibrationGroup!!! to be > 1    (,else = 0)
                                  flagIterativeMethodCalibrationGroupReco = cms.int32(1),
                                  #
                                  hbheInputSignalTag = cms.InputTag('hbherecoMBNZS'),
                                  hbheInputNoiseTag = cms.InputTag('hbherecoNoise'),
                                  hfInputSignalTag = cms.InputTag('hfrecoMBNZS'),
                                  hfInputNoiseTag = cms.InputTag('hfrecoNoise'),
                                  #
                                  #
                                  #
                                  #
                                  #
                                  #
                                  #HistOutFile = cms.untracked.string('PEDESTAL_331370.root'),
                                  #HistOutFile = cms.untracked.string(histodir+'/PEDESTAL_'+runnumber+'.root'),
                                  #MAPOutFile = cms.untracked.string('LogEleMapdb.h')
                                  #
                                  ##OutputFilePath = cms.string('/tmp/zhokin/'),        
                                  ##OutputFileExt = cms.string(''),
                                  #
                                  )		

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)
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

## Jula's recipe for too many files 
#process.options = cms.untracked.PSet(
#   wantSummary = cms.untracked.bool(False),
#   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
#   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
#)

######################################################################################## Global Tags for 2018 data taking :
# use twiki site to specify HLT reconstruction Global tags:
#   https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFrontierConditions
#
#   100X_dataRun2_HLT_v2        for CMSSW_10_0_3 onwards        CRUZET 2018     update of 0T templates for SiPixels
#   100X_dataRun2_HLT_v1        for CMSSW_10_0_0 onwards        MWGRs 2018      first HLT GT for 2018 
#
#
############################################################################ GlobalTag :1+ good as 5
#from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '100X_dataRun2_HLT_v2', '')

#from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data_FULL', '')


#from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '101X_dataRun2_HLT_v7', '')

# 2019 Ultra Legacy 2017
#process.GlobalTag.globaltag = '106X_dataRun2_trackerAlignment2017_v1'
# 2019 Ultra Legacy 2018 test TkAl
#process.GlobalTag.globaltag = '106X_dataRun2_v17'
# 2019 Ultra Legacy 2018 
#process.GlobalTag.globaltag = '106X_dataRun2_newTkAl_v18'
# 2019 Ultra Legacy 2016
#process.GlobalTag.globaltag = '106X_dataRun2_UL2016TkAl_v24'
#process.GlobalTag.globaltag = '105X_dataRun2_v8'

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = '104X_dataRun2_v1'
#process.GlobalTag.globaltag = '105X_postLS2_design_v4'
process.GlobalTag.globaltag = '106X_dataRun3_HLT_v3'


############################################################################
# V.EPSHTEIN:
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = '100X_dataRun2_Prompt_Candidate_2018_01_31_16_01_36'
###
#process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
#    dump = cms.untracked.vstring(''),
#    file = cms.untracked.string('')
#)
#
#process.hcalDigis= cms.EDProducer("HcalRawToDigi",
#    FilterDataQuality = cms.bool(True),
#    HcalFirstFED = cms.untracked.int32(700),
#    InputLabel = cms.InputTag("source"),
#    UnpackCalib = cms.untracked.bool(True),
#    FEDs = cms.untracked.vint32(1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117),
#)
###
############################################################################
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.hcalDigis.FilterDataQuality = cms.bool(False)
process.hcalDigis.InputLabel = cms.InputTag("source")
############################################################################
process.hcalDigis= cms.EDProducer("HcalRawToDigi",
#    FilterDataQuality = cms.bool(True),
    FilterDataQuality = cms.bool(False),
    HcalFirstFED = cms.untracked.int32(700),
    InputLabel = cms.InputTag("source"),
    #InputLabel = cms.InputTag("rawDataCollector"),
)
#process.hcalDigis.FilterDataQuality = cms.bool(False)
#process.hcalDigis.InputLabel = cms.InputTag("source")
############################################################################
##process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalPedestal_cff")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalPedestalLocal_cff")
##process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff")
#process.load("ALCARECOHcalCalPedestalLocal_cff")
############################################################################
#process.p = cms.Path(process.hcalDigis*process.Analyzer)
#process.p = cms.Path(process.seqALCARECOHcalCalMinBiasDigiNoHLT*process.seqALCARECOHcalCalMinBias*process.minbiasana)

process.p = cms.Path(process.hcalDigis*process.seqALCARECOHcalCalMinBiasDigiNoHLT*process.seqALCARECOHcalCalMinBias*process.Analyzer)
#process.p = cms.Path(process.seqALCARECOHcalCalMinBiasDigiNoHLT*process.seqALCARECOHcalCalMinBias*process.Analyzer)

# see   /afs/cern.ch/work/z/zhokin/public/CMSSW_10_4_0_patch1/src/Calibration/HcalAlCaRecoProducers/python/ALCARECOHcalCalMinBias_cff.py
############################################################################
process.MessageLogger = cms.Service("MessageLogger",
     categories   = cms.untracked.vstring(''),
     destinations = cms.untracked.vstring('cout'),
     debugModules = cms.untracked.vstring('*'),
     cout = cms.untracked.PSet(
         threshold = cms.untracked.string('WARNING'),
	 WARNING = cms.untracked.PSet(limit = cms.untracked.int32(0))
     )
 )
############################################################################



