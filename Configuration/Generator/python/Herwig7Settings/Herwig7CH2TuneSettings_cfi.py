import FWCore.ParameterSet.Config as cms

herwig7CH2SettingsBlock = cms.PSet(
    herwig7CH2PDF = cms.vstring(
            'cd /Herwig/Partons',
            'create ThePEG::LHAPDF PDFSet_nnlo ThePEGLHAPDF.so',
            'set PDFSet_nnlo:PDFName NNPDF31_nnlo_as_0118.LHgrid',
            'set PDFSet_nnlo:RemnantHandler HadronRemnants',
            'set /Herwig/Particles/p+:PDF PDFSet_nnlo',
            'set /Herwig/Particles/pbar-:PDF PDFSet_nnlo',

            'set /Herwig/Partons/PPExtractor:FirstPDF  PDFSet_nnlo',
            'set /Herwig/Partons/PPExtractor:SecondPDF PDFSet_nnlo',

            'set /Herwig/Shower/ShowerHandler:PDFA PDFSet_nnlo',
            'set /Herwig/Shower/ShowerHandler:PDFB PDFSet_nnlo',
            
            'create ThePEG::LHAPDF PDFSet_lo ThePEGLHAPDF.so',
            'set PDFSet_lo:PDFName NNPDF31_lo_as_0118.LHgrid',
            'set PDFSet_lo:RemnantHandler HadronRemnants',

            'set /Herwig/Shower/ShowerHandler:PDFARemnant PDFSet_lo',
            'set /Herwig/Shower/ShowerHandler:PDFBRemnant PDFSet_lo',
            'set /Herwig/Partons/MPIExtractor:FirstPDF PDFSet_lo',
            'set /Herwig/Partons/MPIExtractor:SecondPDF PDFSet_lo',

            'cd /',
        ),
    herwig7CH2AlphaS = cms.vstring(
        'cd /Herwig/Shower',
        'set AlphaQCD:AlphaMZ 0.118',
        'cd /'
        ),
    herwig7CH2MPISettings = cms.vstring(
        'read snippets/SoftModel.in',
        'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.479',
        'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 3.138',
        'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.174',
        'set /Herwig/UnderlyingEvent/MPIHandler:Power 0.1203',
        'set /Herwig/Partons/RemnantDecayer:ladderPower -0.08',
        'set /Herwig/Partons/RemnantDecayer:ladderNorm 0.95',
                                )
)
