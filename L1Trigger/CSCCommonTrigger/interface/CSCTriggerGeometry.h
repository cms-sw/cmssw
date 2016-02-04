#ifndef CSCCommonTrigger_CSCTriggerGeometry_h
#define CSCCommonTrigger_CSCTriggerGeometry_h

/**
 * \class CSCTriggerGeometry
 * Static wrapper for CSCTriggerGeomManager
 * Makes changing geometry per event easy.
 * \author L. Gray 3/10/05
 */

#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeomManager.h>

class CSCTriggerGeometry
{
 public:
  CSCTriggerGeometry() {}
  ~CSCTriggerGeometry() {}

  static void setGeometry(const edm::ESHandle<CSCGeometry>& thegeom) { mygeom.setGeometry(thegeom); }
  static CSCTriggerGeomManager* get() { return &mygeom; }

 private:
  static CSCTriggerGeomManager mygeom;
};

#endif
