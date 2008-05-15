import FWCore.ParameterSet.Config as cms

herwigDefaultsBlock = cms.PSet(
	dataLocation = cms.string('${HERWIGPATH}'),

	repository = cms.string('HerwigDefaults.rpo'),
	eventHandlers = cms.string('/Herwig/EventHandlers'),
	generatorModule = cms.string('/Herwig/Generators/LHCGenerator'),
	run = cms.string('LHC'),

	lheDefaultPDFs = cms.vstring(
		'cd /Herwig/EventHandlers', 
		'set LHEReader:PDFA /LHAPDF/cmsPDFSet', 
		'set LHEReader:PDFB /LHAPDF/cmsPDFSet'
	),

	lheDefaults = cms.vstring(
		'cd /Herwig/Cuts', 
		'create ThePEG::Cuts NoCuts', 
		'cd /Herwig/EventHandlers', 
		'create ThePEG::LesHouchesInterface LHEReader', 
		'set LHEReader:Cuts /Herwig/Cuts/NoCuts', 
		'set LHEReader:PDFA /LHAPDF/cmsPDFSet', 
		'set LHEReader:PDFB /LHAPDF/cmsPDFSet', 
		'create ThePEG::LesHouchesEventHandler LHEHandler', 
		'set LHEHandler:WeightOption VarWeight', 
		'set LHEHandler:PartonExtractor /Herwig/Partons/QCDExtractor', 
		'set LHEHandler:CascadeHandler /Herwig/Shower/ShowerHandler', 
		'set LHEHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler', 
		'set LHEHandler:DecayHandler /Herwig/Decays/DecayHandler', 
		'insert LHEHandler:LesHouchesReaders 0 LHEReader', 
		'cd /Herwig/Generators', 
		'set LHCGenerator:EventHandler /Herwig/EventHandlers/LHEHandler'
	),

	cmsDefaults = cms.vstring(
		'mkdir /LHAPDF', 
		'cd /LHAPDF', 
		'create ThePEG::LHAPDF CTEQ5l', 
		'set CTEQ5l:PDFName cteq5l.LHgrid', 
		'set CTEQ5l:RemnantHandler /Herwig/Partons/HadronRemnants', 
		'cp CTEQ5l cmsPDFSet', 
		'set /Herwig/Particles/p+:PDF cmsPDFSet', 
		'set /Herwig/Particles/pbar-:PDF cmsPDFSet', 
		'cd /Herwig/Generators', 
		'set LHCGenerator:NumberOfEvents 10000000', 
		'set LHCGenerator:DebugLevel 1', 
		'set LHCGenerator:PrintEvent 0', 
		'set LHCGenerator:MaxErrors 10000', 
		'set LHCGenerator:EventHandler:LuminosityFunction:Energy 14000.0', 
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 5.7*GeV'
	)
)
