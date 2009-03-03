#include "CommonTools/UtilAlgos/interface/EventSelector.h"
EDM_REGISTER_PLUGINFACTORY(EventSelectorFactory, "EventSelectorFactory");

#include "CommonTools/UtilAlgos/interface/NTupler.h"
EDM_REGISTER_PLUGINFACTORY(NTuplerFactory, "NTuplerFactory");

#include "CommonTools/UtilAlgos/interface/Plotter.h"
EDM_REGISTER_PLUGINFACTORY(PlotterFactory, "PlotterFactory");

#include "CommonTools/UtilAlgos/interface/CachingVariable.h"
EDM_REGISTER_PLUGINFACTORY(CachingVariableFactory, "CachingVariableFactory");
EDM_REGISTER_PLUGINFACTORY(VariableComputerFactory, "VariableComputerFactory");
