import FWCore.ParameterSet.Config as cms

# Switch off MPI

herwigppMPISettingsBlock = cms.PSet(

     hwpp_mpi_switchOff =  cms.vstring(
	'set /Herwig/Shower/ShowerHandler:MPIHandler NULL',
        ),
)
