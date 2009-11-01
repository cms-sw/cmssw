#include "Fireworks/Core/src/fwCintInterfaces.h"
#include "Fireworks/Core/src/CSGConnector.h"
#include "Fireworks/Core/interface/FWIntValueListenerBase.h"
#include "Fireworks/Core/interface/FWSummaryManager.h"
#include "Fireworks/Core/src/FWGUIEventDataAdder.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/src/FWCollectionSummaryWidget.h"
#include "Fireworks/Core/src/FWCompactVerticalLayout.h"

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ class FWGUIEventDataAdder;

#pragma link C++ function fwSetInCint(double);
#pragma link C++ function fwSetInCint(long);
#pragma link C++ function fwGetObjectPtr();
#pragma link C++ class CSGConnector;
#pragma link C++ class FWIntValueListenerBase;
#pragma link C++ class FWColorFrame;
#pragma link C++ class FWColorPopup;
#pragma link C++ class FWColorRow;
#pragma link C++ class FWColorSelect;

#pragma link C++ class FWCollectionSummaryWidget;
#pragma link C++ class FWSummaryManager;

#pragma link C++ class FWCompactVerticalLayout;




#endif
