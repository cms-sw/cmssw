#ifndef METOBJECTS_CORRMET_H
#define METOBJECTS_CORRMET_H

/** \class CorrMET
 *
 * The CorrMET EDProduct type. It is a BaseMET type, but more
 * data may be added later (EEMF, ECHF, etc.)
 *
 * \author Michael Schmitt, The University of Florida
 *
 * \version   1st Version May 31st, 2005.
 *
 ************************************************************/

#include "DataFormats/METObjects/interface/BaseMET.h"

class CorrMET: public BaseMET {
public:

  // Methods
  virtual void clearMET();

};

#endif // METOBJECTS_CORRMET_H
