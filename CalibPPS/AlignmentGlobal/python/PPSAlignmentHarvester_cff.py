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

ppsAlignmentHarvester = ppsAlignmentHarvester_.clone( 
                            text_results_path = "",
                            write_sqlite_results = True
                        )

ALCAHARVESTPPSAlignment = cms.Task(
    EDMtoMEConvertPPSAlignment,
    DQMInfoPPSAlignment,
    ppsAlignmentHarvester
)
