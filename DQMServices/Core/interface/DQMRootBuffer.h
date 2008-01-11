#ifndef DQM_SERVICES_NODE_ROOT_DQM_ROOTBUFFER_H
#define DQM_SERVICES_NODE_ROOT_DQM_ROOTBUFFER_H

// Workaround to ease migration past ROOT TBuffer changes in ROOT 5.16
#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,15,0)
#include "TBufferFile.h"
typedef TBufferFile DQMRootBuffer;
#else
#include "TBuffer.h"
typedef TBuffer DQMRootBuffer;
#endif

#endif
