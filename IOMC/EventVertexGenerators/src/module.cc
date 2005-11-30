#include "IOMC/EventVertexGenerators/interface/EventVertexGeneratorFactory.h"
#include "IOMC/EventVertexGenerators/interface/GaussianEventVertexGenerator.h"
#include "IOMC/EventVertexGenerators/interface/FlatEventVertexGenerator.h"
#include "IOMC/EventVertexGenerators/interface/BeamProfileVertexGenerator.h"

#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE ();
DEFINE_EVENTVERTEXGENERATOR(GaussianEventVertexGenerator);
DEFINE_EVENTVERTEXGENERATOR(FlatEventVertexGenerator);
DEFINE_EVENTVERTEXGENERATOR(BeamProfileVertexGenerator);

