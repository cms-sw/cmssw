import FWCore.ParameterSet.Config as cms

herwig7CH3SettingsBlock = cms.PSet(
    herwig7CH3PDF = cms.vstring(
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
            'set PDFSet_lo:PDFName NNPDF31_lo_as_0130.LHgrid',
            'set PDFSet_lo:RemnantHandler HadronRemnants',

            'set /Herwig/Shower/ShowerHandler:PDFARemnant PDFSet_lo',
            'set /Herwig/Shower/ShowerHandler:PDFBRemnant PDFSet_lo',
            'set /Herwig/Partons/MPIExtractor:FirstPDF PDFSet_lo',
            'set /Herwig/Partons/MPIExtractor:SecondPDF PDFSet_lo',

            'cd /',
        ),
    herwig7CH3AlphaS = cms.vstring(
        'cd /Herwig/Shower',
        'set AlphaQCD:AlphaIn 0.118',
        'cd /'
        ),
    herwig7CH3MPISettings = cms.vstring(
        'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.4712',
        'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 3.04',
        'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.284',
        'set /Herwig/UnderlyingEvent/MPIHandler:Power 0.1362',
                                )
)
