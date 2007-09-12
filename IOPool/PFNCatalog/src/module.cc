#include "IOPool/PFNCatalog/interface/PFNCatalog.h"
#include "FileCatalog/FCImplPluginFactory.h"
#include "FileCatalog/FCMetaImplPluginFactory.h"
#include "PluginManager/ModuleDef.h"

namespace pool
{
DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN( FCImplPluginFactory, PFNCatalog, "pfncatalog");
} // End of namespace pool
