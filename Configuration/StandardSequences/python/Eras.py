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
        # Implementation note: When this was first started, stage1L1Trigger wasn't in all
        # of the eras. Now that it is, it could in theory be dropped if all changes are
        # converted to run2_common (i.e. a search and replace of "stage1L1Trigger" to
        # "run2_common" over the whole python tree). In practice, I don't think it's worth
        # it, and this also gives the flexibilty to take it out easily.
        
        # This era should not be set by the user with the "--era" command, it's
        # activated automatically if the "--fast" command is used.
        self.fastSim = cms.Modifier()
        
        # These are the eras that the user should specify
        self.Run2_25ns = cms.ModifierChain( self.run2_common, self.run2_25ns_specific, self.stage1L1Trigger )
        self.Run2_50ns = cms.ModifierChain( self.run2_common, self.run2_50ns_specific, self.stage1L1Trigger )
        self.Run2_HI = cms.ModifierChain( self.run2_common, self.run2_HI_specific, self.stage1L1Trigger )
        
        # The only thing this collection is used for is for cmsDriver to
        # warn the user if they specify an era that is discouraged from being
        # set directly. It also stops these eras being printed in the error
        # message of available values when an invalid era is specified.
        self.internalUseEras = [self.run2_common, self.run2_25ns_specific,
                                self.run2_50ns_specific, self.run2_HI_specific,
                                self.stage1L1Trigger, self.fastSim ]

eras=Eras()
