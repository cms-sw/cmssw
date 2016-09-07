import FWCore.ParameterSet.Config as cms

# Read in a LHE file from MadGraph5_aMC@NLO

herwigppLHEFileSettingsBlock = cms.PSet(

    hwpp_LHE_Common = cms.vstring(

        'create ThePEG::Cuts /Herwig/Cuts/NoCuts', 									# Use no cuts in shower step, just shower
 
        'create ThePEG::LesHouchesInterface /Herwig/EventHandlers/LHEReader', 
        'set /Herwig/EventHandlers/LHEReader:Cuts /Herwig/Cuts/NoCuts', 						# No cuts while read in of LHE file
        'set /Herwig/EventHandlers/LHEReader:MomentumTreatment RescaleEnergy', 	
        'set /Herwig/EventHandlers/LHEReader:WeightWarnings 0', 							# Suppress weight warnings 
        'set /Herwig/EventHandlers/LHEReader:InitPDFs 0',								# Explicitly set PDF of hard subprocess / Do not retrieve PDF from LHEReader

        'create ThePEG::LesHouchesEventHandler /Herwig/EventHandlers/LHEHandler',
        'insert /Herwig/EventHandlers/LHEHandler:LesHouchesReaders 0 /Herwig/EventHandlers/LHEReader', 
        'set /Herwig/EventHandlers/LHEHandler:WeightOption VarNegWeight', 						# Allow positive and negative event weight
        'set /Herwig/EventHandlers/LHEHandler:PartonExtractor /Herwig/Partons/QCDExtractor', 
        'set /Herwig/EventHandlers/LHEHandler:CascadeHandler /Herwig/Shower/ShowerHandler', 
        'set /Herwig/EventHandlers/LHEHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler', 		# Switch Hadronization on
        'set /Herwig/EventHandlers/LHEHandler:DecayHandler /Herwig/Decays/DecayHandler', 				# Switch Decay on
        'insert /Herwig/EventHandlers/LHEHandler:PreCascadeHandlers 0 /Herwig/NewPhysics/DecayHandler', 		# Needed in 2.7, must be removed in 3.0

        'set /Herwig/Generators/LHCGenerator:EventHandler /Herwig/EventHandlers/LHEHandler', 				# Activate LHEHandler

        'set /Herwig/Shower/Evolver:MaxTry 100',									# Try to shower an event maximum 100 times
        'set /Herwig/Shower/Evolver:HardVetoScaleSource Read', 								# Read event to define hard veto scale

        'set /Herwig/Shower/KinematicsReconstructor:ReconstructionOption General', 
        'set /Herwig/Shower/KinematicsReconstructor:InitialInitialBoostOption LongTransBoost', 

        '+hwpp_MECorr_Common', 												# Switch off ME corrections
															# Or at least require that the user chooses a MECorrection option
    ),

    # Showering MadGraph5_aMC@NLO LHE files: The same PDF for the hard subprocess and the shower must be used
    hwpp_LHE_MadGraph = cms.vstring(

    	'+hwpp_LHE_Common',      
	'set /Herwig/EventHandlers/LHEReader:PDFA /Herwig/Partons/cmsPDFSet', 						# Shower PDF defined by HerwigppPDF_
        'set /Herwig/EventHandlers/LHEReader:PDFB /Herwig/Partons/cmsPDFSet', 						
    ),


    # Showering LO MadGraph5_aMC@NLO LHE files with a different PDF for the hard subprocess 
    ############ WARNING ######
    # This option should only be used with LO MadGraph5_aMC@NLO LHE files.
    # In case of NLO, MC@NLO matched LHE files this results most likely in a mismatch of phase space
    ############ WARNING ######
    # The shower pdf is the standard PDF which one can get including a predefined PDF using HerwigppPDF_
    # The additional pdf of the hard subprocess is also predefined in HerwigppPDF_. However it has the additional suffix _Hard
    # E.g. hwpp_pdf_NNPDF30NLO shower pdf, hwpp_pdf_NNPDF30NLO_Hard hard subprocess pdf
    hwpp_LHE_MadGraph_DifferentPDFs = cms.vstring(

    	'+hwpp_LHE_Common',      
	'set /Herwig/EventHandlers/LHEReader:PDFA /Herwig/Partons/cmsHardPDFSet', 					# Hard subprocess PDF defined by HerwigppPDF_
        'set /Herwig/EventHandlers/LHEReader:PDFB /Herwig/Partons/cmsHardPDFSet', 						
    ),


    # Additional common block for Powheg
    hwpp_LHE_Powheg_Common = cms.vstring(

    	'+hwpp_LHE_Common',
        'set /Herwig/Shower/Evolver:HardVetoMode Yes',									# Treat hardest emission differently
        'set /Herwig/Shower/Evolver:HardVetoReadOption PrimaryCollision',			
    ),

    # Showering Powheg LHE files with the same PDF for the hard subprocess and the shower  
    hwpp_LHE_Powheg = cms.vstring(

    	'+hwpp_LHE_Powheg_Common',       
	'set /Herwig/EventHandlers/LHEReader:PDFA /Herwig/Partons/cmsPDFSet', 						# Shower PDF defined by HerwigppPDF_
        'set /Herwig/EventHandlers/LHEReader:PDFB /Herwig/Partons/cmsPDFSet', 						
    ),

    # Showering Powheg LHE files with a different PDF for the hard subprocess 
    # The shower pdf is the standard PDF which one can get including a predefined PDF using HerwigppPDF_
    # The additional pdf of the hard subprocess is also predefined in HerwigppPDF_. However it has the additional suffix _Hard
    # E.g. hwpp_pdf_NNPDF30NLO shower pdf, hwpp_pdf_NNPDF30NLO_Hard hard subprocess pdf
    hwpp_LHE_Powheg_DifferentPDFs = cms.vstring(

    	'+hwpp_LHE_Powheg_Common', 
	'set /Herwig/EventHandlers/LHEReader:PDFA /Herwig/Partons/cmsHardPDFSet', 					# Hard subprocess PDF defined by HerwigppPDF_
        'set /Herwig/EventHandlers/LHEReader:PDFB /Herwig/Partons/cmsHardPDFSet', 						
    ),
)

