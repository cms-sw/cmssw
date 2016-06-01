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
        self.run2_HF_2016 = cms.Modifier()
        self.ctpps_2016 = cms.Modifier()
        self.stage1L1Trigger = cms.Modifier()
        self.stage2L1Trigger = cms.Modifier()
        self.phase1Pixel = cms.Modifier()
        # Implementation note: When this was first started, stage1L1Trigger wasn't in all
        # of the eras. Now that it is, it could in theory be dropped if all changes are
        # converted to run2_common (i.e. a search and replace of "stage1L1Trigger" to
        # "run2_common" over the whole python tree). In practice, I don't think it's worth
        # it, and this also gives the flexibilty to take it out easily.

        # Phase 2 sub-eras for stable features
        self.phase2_common = cms.Modifier()
        self.phase2_tracker = cms.Modifier()
        self.phase2_hgc = cms.Modifier()
        self.phase2_muon = cms.Modifier()
        # Phase 2 sub-eras for in-development features
        self.phase2dev_common = cms.Modifier()
        self.phase2dev_tracker = cms.Modifier()
        self.phase2dev_hgc = cms.Modifier()
        self.phase2dev_muon = cms.Modifier()


        # These eras are used to specify the tracking configuration
        # when it should differ from the default (which is Run2). This
        # way the tracking configuration is decoupled from the
        # detector geometry to allow e.g. running Run2 tracking on
        # phase1Pixel detector.
        self.trackingLowPU = cms.Modifier()

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
        self.Run2_2016 = cms.ModifierChain( self.run2_common, self.run2_25ns_specific, self.stage2L1Trigger, self.run2_HF_2016, self.ctpps_2016 )
        self.Run2_2017 = cms.ModifierChain( self.Run2_2016, self.phase1Pixel )
        # Scenarios further afield.
        # Phase2 is everything for the 2023 (2026?) detector that works so far in this release.
        self.Phase2 = cms.ModifierChain( self.phase2_common, self.phase2_tracker, self.phase2_hgc, self.phase2_muon )
        # Phase2dev is everything for the 2023 (2026?) detector that is still in development.
        self.Phase2dev = cms.ModifierChain( self.Phase2, self.phase2dev_common, self.phase2dev_tracker, self.phase2dev_hgc, self.phase2dev_muon )
        
        # Scenarios with low-PU tracking (for B=0T reconstruction)
        self.Run2_2016_trackingLowPU = cms.ModifierChain(self.Run2_2016, self.trackingLowPU)

        # The only thing this collection is used for is for cmsDriver to
        # warn the user if they specify an era that is discouraged from being
        # set directly. It also stops these eras being printed in the error
        # message of available values when an invalid era is specified.
        self.internalUseEras = [self.run2_common, self.run2_25ns_specific,
                                self.run2_50ns_specific, self.run2_HI_specific,
                                self.stage1L1Trigger, self.fastSim,
                                self.run2_HF_2016, self.stage2L1Trigger,
                                self.phase1Pixel,
                                self.phase2_common, self.phase2_tracker,
                                self.phase2_hgc, self.phase2_muon,
                                self.phase2dev_common, self.phase2dev_tracker,
                                self.phase2dev_hgc, self.phase2dev_muon,
                                self.trackingLowPU
                               ]

eras=Eras()
