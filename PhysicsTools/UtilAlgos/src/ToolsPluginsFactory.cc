#include "PhysicsTools/CommonTools/interface/EventSelector.h"
EDM_REGISTER_PLUGINFACTORY(EventSelectorFactory, "EventSelectorFactory");

#include "PhysicsTools/CommonTools/interface/NTupler.h"
EDM_REGISTER_PLUGINFACTORY(NTuplerFactory, "NTuplerFactory");

#include "PhysicsTools/CommonTools/interface/Plotter.h"
EDM_REGISTER_PLUGINFACTORY(PlotterFactory, "PlotterFactory");

#include "PhysicsTools/CommonTools/interface/CachingVariable.h"
EDM_REGISTER_PLUGINFACTORY(CachingVariableFactory, "CachingVariableFactory");
