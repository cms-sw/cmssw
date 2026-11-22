import FWCore.ParameterSet.Config as cms

from DQM.HGCAL.hgcalfaststreamdqm_cfi import hgcalfaststreamdqm
from DQM.HGCAL.hgcaldigidqm_cfi import hgcaldigidqm
from DQM.HGCAL.hgcalrechitdqm_cfi import hgcalrechitdqm
from DQM.HGCAL.hgcallayerclusterdqm_cfi import hgcallayerclusterdqm
from DQM.HGCAL.hgCalDQMHarvester_cfi import hgCalDQMHarvester

# Clients reading RAW-level products (FED/ECON-D/ECON-T packet info, digis, trigger digis).
hgcalDQMSources = cms.Sequence(hgcalfaststreamdqm + hgcaldigidqm)

# Clients reading local-reconstruction products; schedule only when rechits/layer clusters are produced.
hgcalRecoDQMSources = cms.Sequence(hgcalrechitdqm + hgcallayerclusterdqm)

hgcalDQMHarvesting = cms.Sequence(hgCalDQMHarvester)
