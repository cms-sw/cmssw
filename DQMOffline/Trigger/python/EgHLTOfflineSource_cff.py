import FWCore.ParameterSet.Config as cms

### 
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *

egHLTOffDQMSource_HEP17 = egHLTOffDQMSource.clone()
egHLTOffDQMSource_HEP17.subDQMDirName=cms.string('HEP17')
egHLTOffDQMSource_HEP17.doHEP =cms.bool(True)

