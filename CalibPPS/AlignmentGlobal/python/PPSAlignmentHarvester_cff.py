import FWCore.ParameterSet.Config as cms
from DQMServices.Components.EDMtoMEConverter_cfi import EDMtoMEConverter

from CalibPPS.AlignmentGlobal.ppsAlignmentHarvester_cfi import ppsAlignmentHarvester as ppsAlignmentHarvester_

EDMtoMEConvertPPSAlignment = EDMtoMEConverter.clone()
EDMtoMEConvertPPSAlignment.lumiInputTag = cms.InputTag("MEtoEDMConvertPPSAlignment", "MEtoEDMConverterLumi")
EDMtoMEConvertPPSAlignment.runInputTag = cms.InputTag("MEtoEDMConvertPPSAlignment", "MEtoEDMConverterRun")

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
DQMInfoPPSAlignment = DQMEDHarvester('DQMHarvestingMetadata',
                                     subSystemFolder=cms.untracked.string('AlCaReco')
                                     )

ppsAlignmentHarvester = ppsAlignmentHarvester_.clone()
ppsAlignmentHarvester.text_results_path = cms.string("")
ppsAlignmentHarvester.write_sqlite_results = cms.bool(True)
ppsAlignmentHarvester.x_ali_rel_final_slope_fixed = cms.bool(False)
ppsAlignmentHarvester.y_ali_final_slope_fixed = cms.bool(False)

ALCAHARVESTPPSAlignment = cms.Task(
    EDMtoMEConvertPPSAlignment,
    DQMInfoPPSAlignment,
    ppsAlignmentHarvester
)
