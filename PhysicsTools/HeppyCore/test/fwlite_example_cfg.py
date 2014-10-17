import os
import PhysicsTools.HeppyCore.framework.config as cfg

# input component 
# several input components can be declared,
# and added to the list of selected components
ttbar = cfg.Component(
    'TTBar',
    files = ['root://eoscms//eos/cms/store/relval/CMSSW_7_2_0_pre5/RelValTTbarLepton_13/GEN-SIM-RECO/POSTLS172_V3-v1/00000/8289BD3A-1731-E411-8D63-002618FDA265.root'],
    # files = ['relval_ttbar_gensimreco.root']
    )

selectedComponents  = [ttbar]

printer = cfg.Analyzer(
    "Printer"
    )

cmstest = cfg.Analyzer(
    "CMSTestAnalyzer"
    )

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    printer,
    cmstest
    ] )

# finalization of the configuration object. 
config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

print config 
