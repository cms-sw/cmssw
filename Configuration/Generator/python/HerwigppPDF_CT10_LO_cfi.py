import FWCore.ParameterSet.Config as cms

# CT10 PDF

herwigppPDFSettingsBlock = cms.PSet(

	# PDF for shower
        hwpp_pdf_CT10 = cms.vstring(
                'create ThePEG::LHAPDF /Herwig/Partons/cmsPDFSet ThePEGLHAPDF.so',			# cmsPDFSet Default name for shower PDF
                'set /Herwig/Partons/cmsPDFSet:PDFName CT10.LHgrid',
                'set /Herwig/Partons/cmsPDFSet:RemnantHandler /Herwig/Partons/HadronRemnants',
                'set /Herwig/Particles/p+:PDF /Herwig/Partons/cmsPDFSet',				# Use PDF in shower
                'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/cmsPDFSet',
        ),

	# PDF for hard subprocess
        hwpp_pdf_CT10_Hard = cms.vstring(
                'cd /Herwig/Partons',
                'create ThePEG::LHAPDF /Herwig/Partons/cmsHardPDFSet ThePEGLHAPDF.so',			# cmsHardPDFSet Default name for hard subprocess PDF
                'set /Herwig/Partons/cmsHardPDFSet:PDFName CT10.LHgrid',
                'set /Herwig/Partons/cmsHardPDFSet:RemnantHandler /Herwig/Partons/HadronRemnants',
        ),

)

