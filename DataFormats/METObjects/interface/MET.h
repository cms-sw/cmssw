#ifndef METOBJECTS_MET_H
#define METOBJECTS_MET_H

/** \class MET
 *
 * The MET EDProduct type. It is a BaseMET type, but more
 * data may be added later (EEMF, ECHF, etc.)
 *
 * \author Michael Schmitt, The University of Florida
 *
 * \version   1st Version May 31st, 2005.
 *
 ************************************************************/

#include "DataFormats/METObjects/interface/BaseMET.h"

class METv0: public BaseMETv0 {
public:

  // Additional Methods
  //virtual void clearMET();

};

#endif // METOBJECTS__MET_H
