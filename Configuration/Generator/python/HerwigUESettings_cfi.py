import FWCore.ParameterSet.Config as cms

herwigUESettingsBlock = cms.PSet(
   herwigUESettings = cms.vstring(
	    'MODPDF(1)  = 10041      ! PDF set according to LHAGLUE',
            'MODPDF(2)  = 10041      ! CTEQ6L',
            'JMUEO      = 1          ! multiparton interaction model',
            'PTJIM      = 4.449      ! 2.8x(sqrt(s)/1.8TeV)^0.27 @ 10 TeV',
            'JMRAD(73)  = 1.8        ! inverse proton radius squared',
            'PRSOF      = 0.0        ! prob. of a soft underlying event',
            'MAXER      = 1000000    ! max error')
)
