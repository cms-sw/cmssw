
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <DQM/EcalEndcapMonitorTasks/interface/EEBeamHodoTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEBeamHodoTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEBeamCaloTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEBeamCaloTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEClusterTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEClusterTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EECosmicTask.h>
DEFINE_ANOTHER_FWK_MODULE(EECosmicTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEStatusFlagsTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEStatusFlagsTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEIntegrityTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEIntegrityTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EELaserTask.h>
DEFINE_ANOTHER_FWK_MODULE(EELaserTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EELedTask.h>
DEFINE_ANOTHER_FWK_MODULE(EELedTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEOccupancyTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEOccupancyTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEPedestalOnlineTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEPedestalOnlineTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEPedestalTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEPedestalTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EETestPulseTask.h>
DEFINE_ANOTHER_FWK_MODULE(EETestPulseTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EETriggerTowerTask.h>
DEFINE_ANOTHER_FWK_MODULE(EETriggerTowerTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EETimingTask.h>
DEFINE_ANOTHER_FWK_MODULE(EETimingTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EESelectiveReadoutTask.h>
DEFINE_ANOTHER_FWK_MODULE(EESelectiveReadoutTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EERawDataTask.h>
DEFINE_ANOTHER_FWK_MODULE(EERawDataTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEHltTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEHltTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEDaqInfoTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEDaqInfoTask);

#include <DQM/EcalEndcapMonitorTasks/interface/EEDcsInfoTask.h>
DEFINE_ANOTHER_FWK_MODULE(EEDcsInfoTask);
