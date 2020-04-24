import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

RazorEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_Rsq          'Rsq turnON;            Rsq;        efficiency'     Rsq_numerator          Rsq_denominator",
        "effic_MR           'MR turnON;             MR [GeV];   efficiency'     MR_numerator           MR_denominator",
        "effic_dPhiR        'dPhiR turnON;          dphiR;      efficiency'     dPhiR_numerator        dPhiR_denominator",
        "effic_MRVsRsq      'MR efficiency vs Rsq;  MR [GeV];   Rsq'            MRVsRsq_numerator      MRVsRsq_denominator",
    ),
)


susyHLTRazorClient = cms.Sequence(
    RazorEfficiency
)
