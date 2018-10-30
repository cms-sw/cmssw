import FWCore.ParameterSet.Config as cms

from Configuration.Generator.TTbar_Pow_LHE_13TeV_cff import externalLHEProducer

generator = cms.EDFilter("Herwig7GeneratorFilter",

    configFiles = cms.vstring(),
                                                                                                                                                             
   hw_nnpdf31 = cms.vstring(
            'cd /Herwig/Partons',
            'create ThePEG::LHAPDF PDFSet ThePEGLHAPDF.so',
            'set PDFSet:PDFName NNPDF31_nnlo_as_0118.LHgrid',
            'set PDFSet:RemnantHandler HadronRemnants',
            'set /Herwig/Particles/p+:PDF PDFSet',
            'set /Herwig/Particles/pbar-:PDF PDFSet',
            'set /Herwig/Shower/ShowerHandler:PDFA PDFSet',
            'set /Herwig/Shower/ShowerHandler:PDFB PDFSet',
            'set /Herwig/DipoleShower/DipoleShowerHandler:PDFA PDFSet',
            'set /Herwig/DipoleShower/DipoleShowerHandler:PDFB PDFSet',

            'set /Herwig/Shower/ShowerHandler:PDFARemnant PDFSet',
            'set /Herwig/Shower/ShowerHandler:PDFBRemnant PDFSet',
            'set /Herwig/DipoleShower/DipoleShowerHandler:PDFARemnant PDFSet',
            'set /Herwig/DipoleShower/DipoleShowerHandler:PDFBRemnant PDFSet',
            'set /Herwig/Partons/MPIExtractor:FirstPDF PDFSet',
            'set /Herwig/Partons/MPIExtractor:SecondPDF PDFSet',

            'cd /',
    ),
    hw_alphas = cms.vstring(
        'cd /Herwig/Shower',
        'set AlphaQCD:AlphaMZ 0.118',
        'cd /Herwig/DipoleShower',
        'set NLOAlphaS:input_alpha_s 0.118'
    ),
    hw_LHEsettings = cms.vstring(
            'read snippets/PPCollider.in',

            'cd /Herwig/Generators',
            'set EventGenerator:NumberOfEvents 10000000',
            #'set EventGenerator:RandomNumberGenerator:Seed 31122001'
            'set EventGenerator:DebugLevel 0',
            'set EventGenerator:PrintEvent 10',
            'set EventGenerator:MaxErrors 10000',

            'cd /Herwig/EventHandlers',
            'library LesHouches.so',
            'create ThePEG::LesHouchesEventHandler LesHouchesHandler',

            'set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor',
            'set LesHouchesHandler:CascadeHandler /Herwig/Shower/ShowerHandler',
            'set LesHouchesHandler:DecayHandler /Herwig/Decays/DecayHandler',
            'set LesHouchesHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler',

            'set LesHouchesHandler:WeightOption VarNegWeight',

            'set /Herwig/Generators/EventGenerator:EventHandler /Herwig/EventHandlers/LesHouchesHandler',

            'create ThePEG::Cuts /Herwig/Cuts/NoCuts',

            'create ThePEG::LHAPDF /Herwig/Partons/LHAPDF ThePEGLHAPDF.so',
            'set /Herwig/Partons/LHAPDF:PDFName NNPDF30_nlo_as_0118',
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
            'set /Herwig/Shower/PartnerFinder:ScaleChoice Partner',

            'set /Herwig/Shower/GtoQQbarSplitFn:AngularOrdered Yes',
            'set /Herwig/Shower/GammatoQQbarSplitFn:AngularOrdered Yes',
            'set /Herwig/Particles/t:NominalMass 172.5',
            'cd /Herwig/Generators',
            'saverun LHE EventGenerator'),                                                                                                                                                                         
    parameterSets = cms.vstring('hw_LHEsettings', 'hw_nnpdf31','hw_alphas'),
    crossSection = cms.untracked.double(-1),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    filterEfficiency = cms.untracked.double(1.0),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    run = cms.string('InterfaceMatchboxTest'),
    runModeList = cms.untracked.string("read,run"),
    seed = cms.untracked.int32(12345)
)


ProductionFilterSequence = cms.Sequence(generator)
