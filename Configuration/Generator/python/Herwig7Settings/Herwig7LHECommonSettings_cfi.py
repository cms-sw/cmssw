import FWCore.ParameterSet.Config as cms

# Settings from $HERWIGPATH/LHE.in

herwig7LHECommonSettingsBlock = cms.PSet(
    hw_lhe_common_settings = cms.vstring(
        'cd /Herwig/EventHandlers',
        'library LesHouches.so',
        'create ThePEG::LesHouchesEventHandler LesHouchesHandler',
        'set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor',
        'set LesHouchesHandler:CascadeHandler /Herwig/Shower/ShowerHandler',
        'set LesHouchesHandler:DecayHandler /Herwig/Decays/DecayHandler',
        'set LesHouchesHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler',

        # set the weight option (e.g. for MC@NLO)
        'set LesHouchesHandler:WeightOption VarNegWeight',
        'set LesHouchesHandler:EventNumbering LHE',

        'set /Herwig/Generators/EventGenerator:EventHandler /Herwig/EventHandlers/LesHouchesHandler',
        'create ThePEG::Cuts /Herwig/Cuts/NoCuts',
        'create ThePEG::LHAPDF /Herwig/Partons/LHAPDF ThePEGLHAPDF.so',
        'set /Herwig/Partons/LHAPDF:PDFName NNPDF31_nnlo_as_0118',
        'set /Herwig/Partons/LHAPDF:RemnantHandler /Herwig/Partons/HadronRemnants',
        'set /Herwig/Particles/p+:PDF /Herwig/Partons/LHAPDF',
        'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/LHAPDF',
        'set /Herwig/Partons/PPExtractor:FirstPDF  /Herwig/Partons/LHAPDF',
        'set /Herwig/Partons/PPExtractor:SecondPDF /Herwig/Partons/LHAPDF',
        'create ThePEG::LesHouchesFileReader LesHouchesReader',
        'set LesHouchesReader:FileName cmsgrid_final.lhe',
        'set LesHouchesReader:AllowedToReOpen No',
        'set LesHouchesReader:InitPDFs 0',
        'set LesHouchesReader:Cuts /Herwig/Cuts/NoCuts',
        'set LesHouchesReader:MomentumTreatment RescaleEnergy',
        'set LesHouchesReader:PDFA /Herwig/Partons/LHAPDF',
        'set LesHouchesReader:PDFB /Herwig/Partons/LHAPDF',
        'insert LesHouchesHandler:LesHouchesReaders 0 LesHouchesReader',
        'set /Herwig/Shower/ShowerHandler:MaxPtIsMuF Yes',
        'set /Herwig/Shower/ShowerHandler:RestrictPhasespace Yes',
        'set /Herwig/Shower/PartnerFinder:PartnerMethod Random',
        'set /Herwig/Shower/PartnerFinder:ScaleChoice Partner'
    )
)
