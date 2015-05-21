import FWCore.ParameterSet.Config as cms

# CTEQ6L PDF

herwigppPDFSettingsBlock = cms.PSet(

	# PDF for shower
        hwpp_pdf_pdfCTEQ6L1 = cms.vstring(
                '+hwpp_ue_EE5C', 							# Tune for CTEQ6L1 from 2.7, see HerwigppUE_EE_5C 
                'create ThePEG::LHAPDF /Herwig/Partons/cmsPDFSet ThePEGLHAPDF.so',	# cmsPDFSet Default name for shower PDF
                'set /Herwig/Partons/cmsPDFSet:PDFName cteq6ll.LHpdf',	
                'set /Herwig/Partons/cmsPDFSet:RemnantHandler HadronRemnants',
                'set /Herwig/Particles/p+:PDF /Herwig/Partons/cmsPDFSet',		# Use PDF in shower
                'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/cmsPDFSet',
        ),

	hwpp_pdf_CTEQ6LL = cms.vstring(
                '+hwpp_pdf_CTEQ6L1',							# Correct mixing up the name
        ),

	# PDF for hard subprocess
        hwpp_pdf_pdfCTEQ6L1_Hard = cms.vstring(
                '+hwpp_ue_EE5C', 							# Tune for CTEQ6L1 from 2.7, see HerwigppUE_EE_5C 
                'create ThePEG::LHAPDF /Herwig/Partons/cmsHardPDFSet ThePEGLHAPDF.so',	# cmsHardPDFSet Default name for hard subprocess PDF
                'set /Herwig/Partons/cmsHardPDFSet:PDFName cteq6ll.LHpdf',	
                'set /Herwig/Partons/cmsHardPDFSet:RemnantHandler HadronRemnants',
        ),

	hwpp_pdf_CTEQ6LL_Hard = cms.vstring(
                '+hwpp_pdf_CTEQ6L1_Hard',						# Correct mixing up the name
        ),
)

