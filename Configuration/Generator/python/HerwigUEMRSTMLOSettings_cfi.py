import FWCore.ParameterSet.Config as cms

herwigUESettingsBlock = cms.PSet(
   herwigUESettings = cms.vstring(
	    'MODPDF(1)  = 20650      ! PDF set according to LHAGLUE',
            'MODPDF(2)  = 20650      ! MRST LO*',
            'JMUEO      = 1          ! multiparton interaction model',
            'PTJIM      = 6.323      ! 3.98x(sqrt(s)/1.8TeV)^0.27 @ 10 TeV',
            'JMRAD(73)  = 2.80       ! inverse proton radius squared',
	    'ISPAC      = 2          ! treatment of force parton branchings in backward evolution',
            'PRSOF      = 0.0        ! prob. of a soft underlying event',
            'MAXER      = 1000000    ! max error')
)
