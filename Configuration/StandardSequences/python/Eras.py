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
        self.stage1L1Trigger = cms.Modifier()
        self.stage2L1Trigger = cms.Modifier()
        
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
        
        # The only thing this collection is used for is for cmsDriver to
        # warn the user if they specify an era that is discouraged from being
        # set directly. It also stops these eras being printed in the error
        # message of available values when an invalid era is specified.
        self.internalUseEras = [self.run2_common, self.run2_25ns_specific,
                                self.run2_50ns_specific, self.run2_HI_specific,
                                self.stage1L1Trigger, self.fastSim ]

eras=Eras()
