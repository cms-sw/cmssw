#include "IOMC/EventVertexGenerators/interface/EventVertexGeneratorFactory.h"

EventVertexGeneratorFactory EventVertexGeneratorFactory::s_instance;

EventVertexGeneratorFactory::EventVertexGeneratorFactory()
    : seal::PluginFactory<EventVertexGeneratorMakerBase * ()>("CMS Simulation EventVertexGeneratorFactory")
{}

EventVertexGeneratorFactory::~EventVertexGeneratorFactory() {}

EventVertexGeneratorFactory * EventVertexGeneratorFactory::get() { return & s_instance; }
