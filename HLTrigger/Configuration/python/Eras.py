import FWCore.ParameterSet.Config as cms

def modifyHLTforEras(fragment):
    """load all Eras-based customisations for the HLT configuration"""

    # modify the HLT configuration for the 2016 configuration
    from HLTrigger.Configuration.customizeHLTfor2016_cff import modifyHLTfor2016
    modifyHLTfor2016(fragment)

