import FWCore.ParameterSet.Config as cms

# Name:   RecoMET.cff
# Original Author: R.Cavanaugh
# Date:   05.11.2006
# Notes:  CaloMET.cfi assumes that a product with label "caloTowers" is
#         already written into the event.
# Modification by F. Ratnikov and R. Remington
# Date: 10/21/08 
# Addition of MET significance by F.Blekman
# Date: 10/23/08
# Addition of HCAL noise by JP Chou
# Date:  3/26/09

from RecoMET.Configuration.RecoMET_cff import *

#tcMetP5 = tcMet.clone(trackInputTag = 'ctfWithMaterialTracksP5LHCNavigation')
tcMet.trackInputTag = 'ctfWithMaterialTracksP5LHCNavigation'

#hcalnoiseCosmics = hcalnoise.clone(fillTracks = False)
hcalnoise.fillTracks = False


metrecoCosmics = cms.Sequence(
      met+
      metNoHF+
      metHO+
      metNoHFHO+
      calotoweroptmaker+
      metOpt+
      metOptNoHF+
      calotoweroptmakerWithHO+
      metOptHO+metOptNoHFHO+
      htMetSC5+
      htMetSC7+
      htMetKT4+
      htMetKT6+
      htMetIC5+
      muonMETValueMapProducer+
      corMetGlobalMuons+
      muonTCMETValueMapProducer+
      hcalnoise)

metrecoCosmics_woHcalNoise = cms.Sequence(
    met+
    metNoHF+
    metHO+
    metNoHFHO+
    calotoweroptmaker+
    metOpt+
    metOptNoHF+
    calotoweroptmakerWithHO+
    metOptHO+metOptNoHFHO+
    htMetSC5+
    htMetSC7+
    htMetKT4+
    htMetKT6+
    htMetIC5+
    muonMETValueMapProducer+
    corMetGlobalMuons+
    muonTCMETValueMapProducer
)
    




