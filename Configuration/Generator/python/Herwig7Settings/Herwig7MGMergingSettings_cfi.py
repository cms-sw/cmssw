import FWCore.ParameterSet.Config as cms

# Settings from $HERWIGPATH/LHE-MGMerging.in, including reading in the LHE file- do not use this together with Herwig7LHECommonSettings 
# User needs to include the following lines in their user settings block:
# 'set FxFxHandler:MergeMode TreeMG5' (for mlm merging) OR 'set FxFxHandler:MergeMode FxFx' (for FxFx merging)
# 'set FxFxHandler:njetsmax  MAXIMUM_NUMBER_OF_PARTONS_IN_LHE_FILE'

herwig7MGMergingSettingsBlock = cms.PSet(
    hw_mg_merging_settings = cms.vstring(
        'cd /Herwig/EventHandlers',
        'library HwFxFx.so',
        'create Herwig::FxFxEventHandler LesHouchesHandler',
        'set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor',
        'set LesHouchesHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler',
        'set LesHouchesHandler:DecayHandler /Herwig/Decays/DecayHandler',
        'set LesHouchesHandler:WeightOption VarNegWeight',
        'set LesHouchesHandler:EventNumbering LHE',
        'set /Herwig/Generators/EventGenerator:EventHandler  /Herwig/EventHandlers/LesHouchesHandler',
        'create ThePEG::Cuts /Herwig/Cuts/NoCuts',
        'cd /Herwig/EventHandlers',
        'create Herwig::FxFxFileReader FxFxLHReader',
        'insert LesHouchesHandler:FxFxReaders[0] FxFxLHReader',
        'set FxFxLHReader:FileName cmsgrid_final.lhe',
        'set FxFxLHReader:WeightWarnings false',
        'set FxFxLHReader:AllowedToReOpen No',
        'set FxFxLHReader:InitPDFs 0',
        'set FxFxLHReader:Cuts /Herwig/Cuts/NoCuts',
        'set FxFxLHReader:MomentumTreatment RescaleEnergy',

        'cd /Herwig/Shower',
        'library HwFxFxHandler.so',
        'create Herwig::FxFxHandler FxFxHandler',
        'set FxFxHandler:SplittingGenerator /Herwig/Shower/SplittingGenerator',
        'set FxFxHandler:KinematicsReconstructor /Herwig/Shower/KinematicsReconstructor',
        'set FxFxHandler:PartnerFinder /Herwig/Shower/PartnerFinder',
        'set /Herwig/EventHandlers/LesHouchesHandler:CascadeHandler /Herwig/Shower/FxFxHandler',
        'set /Herwig/Partons/RemnantDecayer:AllowTop Yes',
        
        'set FxFxHandler:MaxPtIsMuF Yes',
        'set FxFxHandler:RestrictPhasespace Yes',
        'set PartnerFinder:PartnerMethod Random',
        'set PartnerFinder:ScaleChoice Partner',
        'set KinematicsReconstructor:InitialInitialBoostOption LongTransBoost',
        'set KinematicsReconstructor:ReconstructionOption General',
        'set KinematicsReconstructor:InitialStateReconOption Rapidity',
        'set FxFxHandler:SpinCorrelations Yes',

        'set FxFxHandler:MPIHandler  /Herwig/UnderlyingEvent/MPIHandler',
        'set FxFxHandler:RemDecayer  /Herwig/Partons/RemnantDecayer',
        'set FxFxHandler:ShowerAlpha  AlphaQCD',
        'set FxFxHandler:IntrinsicPtGaussian 2.2*GeV', 
        'set FxFxHandler:HeavyQVeto Yes',
        'set FxFxHandler:HardProcessDetection Automatic',
        'set FxFxHandler:drjmin 0',

        'set FxFxHandler:VetoIsTurnedOff VetoingIsOn',
        'set FxFxHandler:ETClus 20*GeV', # Note this is the default, but this parameter should be tuned in future
        'set FxFxHandler:RClus 1.0',
        'set FxFxHandler:EtaClusMax 10',
        'set FxFxHandler:RClusFactor 1.5',
    ),
    herwig7_mg_merging_CH3PDF = cms.vstring(
        'cd /Herwig/Partons',
        'create ThePEG::LHAPDF PDFSet_nnlo ThePEGLHAPDF.so',
        'set PDFSet_nnlo:PDFName NNPDF31_nnlo_as_0118.LHgrid',
        'set PDFSet_nnlo:RemnantHandler HadronRemnants',
        'set /Herwig/Particles/p+:PDF PDFSet_nnlo',
        'set /Herwig/Particles/pbar-:PDF PDFSet_nnlo',

        'set /Herwig/Partons/PPExtractor:FirstPDF  PDFSet_nnlo',
        'set /Herwig/Partons/PPExtractor:SecondPDF PDFSet_nnlo',

        'set /Herwig/Shower/FxFxHandler:PDFA PDFSet_nnlo',
        'set /Herwig/Shower/FxFxHandler:PDFB PDFSet_nnlo',
        
        'create ThePEG::LHAPDF PDFSet_lo ThePEGLHAPDF.so',
        'set PDFSet_lo:PDFName NNPDF31_lo_as_0130.LHgrid',
        'set PDFSet_lo:RemnantHandler HadronRemnants',

        'set /Herwig/Shower/FxFxHandler:PDFARemnant PDFSet_lo',
        'set /Herwig/Shower/FxFxHandler:PDFBRemnant PDFSet_lo',
        'set /Herwig/Partons/MPIExtractor:FirstPDF PDFSet_lo',
        'set /Herwig/Partons/MPIExtractor:SecondPDF PDFSet_lo',

        'set /Herwig/EventHandlers/FxFxLHReader:PDFA /Herwig/Partons/PDFSet_nnlo',
        'set /Herwig/EventHandlers/FxFxLHReader:PDFB /Herwig/Partons/PDFSet_nnlo',

        'cd /',
        # Not technically PDF related, but part of the 7p1 settings which should be used with the CH3 tune
        'set /Herwig/Shower/FxFxHandler:EvolutionScheme Q2'
    )
)
