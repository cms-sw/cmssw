// Backward compatibility hack to allow the old, completely useless
// idiom "Service<MonitorDaemon>().operator->()", spread all around
// DQM subsystem, to continue to work.
#warning Please remove include of DQMServices/Core/interface/MonitorDaemon.h \
         and any use of MonitorDaemon (any use is most likely unnecessary)
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/src/DQMService.h"
typedef DQMService MonitorDaemon;
