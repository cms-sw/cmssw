import FWCore.ParameterSet.Config as cms

# CT10 PDF

herwigppPDFSettingsBlock = cms.PSet(

        pdfCT10 = cms.vstring(
                'cd /Herwig/Partons',
                'create ThePEG::LHAPDF cmsPDFSet ThePEGLHAPDF.so',
                'set cmsPDFSet:PDFName CT10.LHgrid',
                'set cmsPDFSet:RemnantHandler HadronRemnants',
                'set /Herwig/Particles/p+:PDF cmsPDFSet',
                'set /Herwig/Particles/pbar-:PDF cmsPDFSet',
                'cd /',
        ),
)

