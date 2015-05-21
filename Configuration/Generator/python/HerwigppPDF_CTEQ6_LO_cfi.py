import FWCore.ParameterSet.Config as cms

# CTEQ6L PDF

herwigppPDFSettingsBlock = cms.PSet(

	hwpp_pdf_CTEQ6LL = cms.vstring(
                'cd /Herwig/Partons',
                'create ThePEG::LHAPDF cmsPDFSet ThePEGLHAPDF.so',
                'set cmsPDFSet:PDFName cteq6ll.LHpdf',
                'set cmsPDFSet:RemnantHandler HadronRemnants',
                'set /Herwig/Particles/p+:PDF cmsPDFSet',
                'set /Herwig/Particles/pbar-:PDF cmsPDFSet',
                '+hwpp_ue_EE5C', # Tune for CTEQ6L1 from 2.7, see HerwigppUE_EE_5C
                'cd /',
        ),

        hwpp_pdf_pdfCTEQ6L1 = cms.vstring(
                'cd /Herwig/Partons',
                'create ThePEG::LHAPDF cmsPDFSet ThePEGLHAPDF.so',
                'set cmsPDFSet:PDFName cteq6ll.LHpdf',
                'set cmsPDFSet:RemnantHandler HadronRemnants',
                'set /Herwig/Particles/p+:PDF cmsPDFSet',
                'set /Herwig/Particles/pbar-:PDF cmsPDFSet',
                '+hwpp_ue_EE5C', # Tune for CTEQ6L1 from 2.7, see HerwigppUE_EE_5C                 
                'cd /',
        ),
)

