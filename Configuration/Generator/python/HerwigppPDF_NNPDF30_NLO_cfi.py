import FWCore.ParameterSet.Config as cms

#NNPDF30 PDF

herwigppPDFSettingsBlock = cms.PSet(

        pdfNNPDF30NLO = cms.vstring(
                'cd /Herwig/Partons',
                'create ThePEG::LHAPDF cmsPDFSet ThePEGLHAPDF.so',
                'set cmsPDFSet:PDFName NNPDF30_nlo_as_0118.LHgrid',
                'set cmsPDFSet:RemnantHandler HadronRemnants',
                'set /Herwig/Particles/p+:PDF cmsPDFSet',
                'set /Herwig/Particles/pbar-:PDF cmsPDFSet',
                'cd /',
        ),
)

