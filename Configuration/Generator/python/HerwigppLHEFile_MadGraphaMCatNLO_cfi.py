import FWCore.ParameterSet.Config as cms

# Read in a LHE file from MadGraph5_aMC@NLO

herwigLHEFileSettingsBlock = cms.PSet(

	LHEFileMadGraph = cms.vstring(
		'cd /Herwig/Cuts',
		'create ThePEG::Cuts NoCuts',
		'cd /Herwig/EventHandlers',
		'create ThePEG::LesHouchesInterface LHEReader',
		'set LHEReader:Cuts /Herwig/Cuts/NoCuts',
		'create ThePEG::LesHouchesEventHandler LHEHandler',
                'set LHEReader:MomentumTreatment RescaleEnergy',
                'set LHEReader:WeightWarnings 0',
#                'set LHEReader:InitPDFs 1', # Do not try to derive PDFs from the LHE file
		'set LHEHandler:WeightOption VarNegWeight',
		'set LHEHandler:PartonExtractor /Herwig/Partons/QCDExtractor',
		'set LHEHandler:CascadeHandler /Herwig/Shower/ShowerHandler',
		'set LHEHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler',
		'set LHEHandler:DecayHandler /Herwig/Decays/DecayHandler',	
		'insert LHEHandler:LesHouchesReaders 0 LHEReader',
		'cd /Herwig/Generators',
		'set LHCGenerator:EventHandler /Herwig/EventHandlers/LHEHandler',
		'cd /Herwig/Shower',
		'set Evolver:HardVetoScaleSource Read',
		'set Evolver:MECorrMode No',
		'cd /',
                'set /Herwig/Shower/KinematicsReconstructor:ReconstructionOption General',
                'set /Herwig/Shower/KinematicsReconstructor:InitialInitialBoostOption LongTransBoost',
                'cd /Herwig/EventHandlers',
                'set LHEReader:PDFA /Herwig/Partons/cmsPDFSet',
                'set LHEReader:PDFB /Herwig/Partons/cmsPDFSet',
                'cd /',                 
	),
)

