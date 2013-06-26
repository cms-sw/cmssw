#ifndef METOBJECTS_TOWER_MET_H
#define METOBJECTS_TOWER_MET_H

/** \class TowerMET
 *
 * The TowerMET EDProduct type. It is a BaseMET type, but more
 * data may be added later (EEMF, ECHF, etc.)
 *
 * \author Michael Schmitt, The University of Florida
 *
 * \version   1st Version May 31st, 2005.
 *
 ************************************************************/

#include "DataFormats/METObjects/interface/BaseMET.h"

class TowerMETv0: public BaseMETv0 {
public:

  // Methods
  virtual void clearMET();

  virtual ~TowerMETv0() {};

};

#endif // METOBJECTS_TOWER_MET_H
