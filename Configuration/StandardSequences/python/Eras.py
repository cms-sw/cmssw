import FWCore.ParameterSet.Config as cms

class Eras (object):
    """
    Dummy container for all the cms.Modifier instances that config fragments
    can use to selectively configure depending on what scenario is active.
    """
    def __init__(self):
        # These eras should not be set directly by the user.
        self.run2_common = cms.Modifier()
        self.run2_25ns_specific = cms.Modifier()
        self.run2_50ns_specific = cms.Modifier()
        self.run2_HI_specific = cms.Modifier()
        self.run2_HE_2017 = cms.Modifier()
        self.stage1L1Trigger = cms.Modifier()
        self.stage2L1Trigger = cms.Modifier()
        self.phase1Pixel = cms.Modifier()
        # Implementation note: When this was first started, stage1L1Trigger wasn't in all
        # of the eras. Now that it is, it could in theory be dropped if all changes are
        # converted to run2_common (i.e. a search and replace of "stage1L1Trigger" to
        # "run2_common" over the whole python tree). In practice, I don't think it's worth
        # it, and this also gives the flexibilty to take it out easily.
        self.run3_GEM = cms.Modifier()

        # Phase 2 sub-eras for stable features
        self.phase2_common = cms.Modifier()
        self.phase2_tracker = cms.Modifier()
        self.phase2_hgcal = cms.Modifier()
        self.phase2_muon = cms.Modifier()
        # Phase 2 sub-eras for in-development features
        self.phase2dev_common = cms.Modifier()
        self.phase2dev_tracker = cms.Modifier()
        self.phase2dev_hgcal = cms.Modifier()
        self.phase2dev_muon = cms.Modifier()

        # These eras are used to specify the tracking configuration
        # when it should differ from the default (which is Run2). This
        # way the tracking configuration is decoupled from the
        # detector geometry to allow e.g. running Run2 tracking on
        # phase1Pixel detector.
        self.trackingPhase1 = cms.Modifier()
        self.trackingPhase1PU70 = cms.Modifier()
        self.trackingLowPU = cms.Modifier()
        self.trackingPhase2PU140 = cms.Modifier()
        
        # This era should not be set by the user with the "--era" command, it's
        # activated automatically if the "--fast" command is used.
        self.fastSim = cms.Modifier()
        
        #
        # These are the eras that the user should specify
        #
        # Run1 currently does nothing. It's useful to use as a no-operation era commands when scripting,
        # but also retains the flexibility to add Run1 specific commands at a later date.
        self.Run1 = cms.Modifier()
        # The various Run2 scenarios for 2015 startup.
        self.Run2_25ns = cms.ModifierChain( self.run2_common, self.run2_25ns_specific, self.stage1L1Trigger )
        self.Run2_50ns = cms.ModifierChain( self.run2_common, self.run2_50ns_specific )
        self.Run2_HI = cms.ModifierChain( self.run2_common, self.run2_HI_specific, self.stage1L1Trigger )
        # Future Run 2 scenarios.
        self.Run2_2016 = cms.ModifierChain( self.run2_common, self.run2_25ns_specific, self.stage2L1Trigger )
        self.Run2_2017 = cms.ModifierChain( self.Run2_2016, self.phase1Pixel, self.trackingPhase1, self.run2_HE_2017 )
        # Scenarios further afield.
        # Run3 includes the GE1/1 upgrade
        self.Run3 = cms.ModifierChain( self.Run2_2017,self.run3_GEM )
        # Phase2 is everything for the 2023 (2026?) detector that works so far in this release.
        # include phase 1 stuff until phase 2 tracking is fully defined....
        self.Phase2 = cms.ModifierChain( self.phase2_common, self.phase2_tracker, self.trackingPhase2PU140, self.phase2_hgcal, self.phase2_muon, self.run3_GEM )
        # Phase2dev is everything for the 2023 (2026?) detector that is still in development.
        self.Phase2dev = cms.ModifierChain( self.Phase2, self.phase2dev_common, self.phase2dev_tracker, self.trackingPhase2PU140, self.phase2dev_hgcal, self.phase2dev_muon )

        # Scenarios with low-PU tracking (for B=0T reconstruction)
        self.Run2_2016_trackingLowPU = cms.ModifierChain(self.Run2_2016, self.trackingLowPU)

        # 2017 scenarios with customized tracking for expert use
        # Will be used as reference points for 2017 tracking development
        self.Run2_2017_trackingPhase1PU70 = cms.ModifierChain( self.Run2_2016, self.phase1Pixel, self.trackingPhase1PU70 )
        self.Run2_2017_trackingRun2 = cms.ModifierChain( self.Run2_2016, self.phase1Pixel ) # no tracking-era = Run2 tracking
        
        # The only thing this collection is used for is for cmsDriver to
        # warn the user if they specify an era that is discouraged from being
        # set directly. It also stops these eras being printed in the error
        # message of available values when an invalid era is specified.
        self.internalUseEras = [self.run2_common, self.run2_25ns_specific,
                                self.run2_50ns_specific, self.run2_HI_specific,
                                self.stage1L1Trigger, self.fastSim,
                                self.run2_HE_2017, self.stage2L1Trigger,
                                self.phase1Pixel, self.run3_GEM,
                                self.phase2_common, self.phase2_tracker,
                                self.phase2_hgcal, self.phase2_muon,
                                self.phase2dev_common, self.phase2dev_tracker,
                                self.phase2dev_hgcal, self.phase2dev_muon,
                                self.trackingLowPU, self.trackingPhase1, self.trackingPhase1PU70,
                               ]

eras=Eras()
