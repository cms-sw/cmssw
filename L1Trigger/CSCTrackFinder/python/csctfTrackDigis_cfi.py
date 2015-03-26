import FWCore.ParameterSet.Config as cms

# This function makes all the changes to csctfTrackDigis required for it to work
# in Run 2. It is declared here to be obvious when this file is opened, but not
# applied until after csctfTrackDigis is declared (and then only if the "run2"
# era is active).
def _modifyCsctfTrackDigisForRun2( object ) :
	object.SectorProcessor.PTLUT.PtMethod = 34
	object.SectorProcessor.gangedME1a = False
	object.SectorProcessor.firmwareSP = 20140515
	object.SectorProcessor.initializeFromPSet = False 


from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
csctfTrackDigis = cms.EDProducer("CSCTFTrackProducer",
	DTproducer = cms.untracked.InputTag("dtTriggerPrimitiveDigis"),
	DtDirectProd = cms.untracked.InputTag("csctfunpacker","DT"),
	SectorReceiverInput = cms.untracked.InputTag("cscTriggerPrimitiveDigis","MPCSORTED"),
	SectorProcessor = cms.PSet(CSCCommonTrigger,
		# LUT Setup
		###########
		SRLUT = cms.PSet(
			Binary = cms.untracked.bool(False),
			ReadLUTs = cms.untracked.bool(False),
			LUTPath = cms.untracked.string('./'),
			UseMiniLUTs = cms.untracked.bool(True)
		),

		PTLUT = cms.PSet(
			LowQualityFlag = cms.untracked.uint32(4),
			ReadPtLUT = cms.bool(False),
			PtMethod = cms.untracked.uint32(32)
		),
		
		# Operational mode control
		##########################
		AllowALCTonly = cms.bool(False),
		AllowCLCTonly = cms.bool(False),
		rescaleSinglesPhi  = cms.bool(True),
		run_core = cms.bool(True),
		trigger_on_MB1a = cms.bool(False),
		trigger_on_MB1d = cms.bool(False),
		trigger_on_ME1a = cms.bool(False),
		trigger_on_ME1b = cms.bool(False),
		trigger_on_ME2 = cms.bool(False),
		trigger_on_ME3 = cms.bool(False),
		trigger_on_ME4 = cms.bool(False),
		singlesTrackOutput = cms.uint32(1),
		gangedME1a = cms.untracked.bool(True),
		CoreLatency = cms.uint32(7),
		PreTrigger = cms.uint32(2),
		BXAdepth = cms.uint32(2),
		widePhi = cms.uint32(0),
		
		# Control Registers to core,
		# Reordered to match firmware interface
		#######################################
		mindetap = cms.uint32(7),
		mindetap_halo = cms.uint32(8),
		
		EtaMin = cms.vuint32(0, 0, 0, 0, 0, 0, 0, 0),
		
		mindeta12_accp = cms.uint32(12),
		mindeta13_accp = cms.uint32(13),
		mindeta112_accp = cms.uint32(14),
		mindeta113_accp = cms.uint32(21),
		
		EtaMax = cms.vuint32(127, 127, 127, 127, 127, 24, 24, 127),
		
		maxdeta12_accp = cms.uint32(17),
		maxdeta13_accp = cms.uint32(27),
		maxdeta112_accp = cms.uint32(29),
		maxdeta113_accp = cms.uint32(38),
		
		EtaWindows = cms.vuint32(4, 4, 6, 6, 6, 6, 6), 
		
		maxdphi12_accp = cms.uint32(64),
		maxdphi13_accp = cms.uint32(64),
		maxdphi112_accp = cms.uint32(64),
		maxdphi113_accp = cms.uint32(64),
		
		mindphip = cms.uint32(180),
		mindphip_halo = cms.uint32(128),
		
		straightp = cms.uint32(19),
		curvedp = cms.uint32(15),
		
		mbaPhiOff = cms.uint32(0),
		mbbPhiOff = cms.uint32(1982),
		
		kill_fiber         = cms.uint32(0),
		QualityEnableME1a  = cms.uint32(65535),
		QualityEnableME1b  = cms.uint32(65535),
		QualityEnableME1c  = cms.uint32(65535),
		QualityEnableME1d  = cms.uint32(65535),
		QualityEnableME1e  = cms.uint32(65535),
		QualityEnableME1f  = cms.uint32(65535),
		QualityEnableME2a  = cms.uint32(65535),
		QualityEnableME2b  = cms.uint32(65535),
		QualityEnableME2c  = cms.uint32(65535),
		QualityEnableME3a  = cms.uint32(65535),
		QualityEnableME3b  = cms.uint32(65535),
		QualityEnableME3c  = cms.uint32(65535),
		QualityEnableME4a  = cms.uint32(65535),
		QualityEnableME4b  = cms.uint32(65535),
		QualityEnableME4c  = cms.uint32(65535),

		firmwareSP = cms.uint32(20120319),#core 20120313
		firmwareFA = cms.uint32(20091026),
		firmwareDD = cms.uint32(20091026),
		firmwareVM = cms.uint32(20091026),

                isCoreVerbose = cms.bool(False),
                                   
		#use firmware version and PTLUTs listed in this files if True
                #use firmware and PTLUTs from O2O if False 
                #initializeFromPSet = cms.bool(True)
                initializeFromPSet = cms.bool(False)
	),
                                 
	isTMB07 = cms.bool(True),
	useDT = cms.bool(True),
	readDtDirect = cms.bool(False),
)

#
# If the run2 era is active, make the required changes
#
from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify( csctfTrackDigis, _modifyCsctfTrackDigisForRun2 )

