import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/powheg/V2/TT_weight_NNPDF3.0/TT_weight_NNPDF3.0_tarball.tar.gz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

generator = cms.EDFilter("Herwig7GeneratorFilter",

     configFiles = cms.vstring('/nfs/dust/cms/user/gvonsem/cms/TOPMC/herwig7/powheg/hadronize/pretuning/CMSSW_10_0_0/src/LHE_custom.in'),
                                                                                                                                                             
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
    parameterSets = cms.vstring('hw_nnpdf31','hw_alphas'),
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
