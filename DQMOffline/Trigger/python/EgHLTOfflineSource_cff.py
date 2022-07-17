import FWCore.ParameterSet.Config as cms

### 
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *

egHLTOffDQMSource_HEP17 = egHLTOffDQMSource.clone(
    subDQMDirName = 'HEP17',
    doHEP = True
)
