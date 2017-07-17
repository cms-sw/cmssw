#include "CommonTools/UtilAlgos/interface/EventSelector.h"
EDM_REGISTER_PLUGINFACTORY(EventSelectorFactory, "EventSelectorFactory");
EDM_REGISTER_PLUGINFACTORY(EventSelectorFactoryFromHelper, "EventSelectorFactoryFromHelper");

#include "PhysicsTools/UtilAlgos/interface/NTupler.h"
EDM_REGISTER_PLUGINFACTORY(NTuplerFactory, "NTuplerFactory");

#include "PhysicsTools/UtilAlgos/interface/Plotter.h"
EDM_REGISTER_PLUGINFACTORY(PlotterFactory, "PlotterFactory");

#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"
EDM_REGISTER_PLUGINFACTORY(CachingVariableFactory, "CachingVariableFactory");
EDM_REGISTER_PLUGINFACTORY(VariableComputerFactory, "VariableComputerFactory");
