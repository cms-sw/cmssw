#ifndef CalibMuon_DTT0FillDefaultFromDB_H
#define CalibMuon_DTT0FillDefaultFromDB_H

/** \class DTT0FillDefaultFromDB
 *  Concrete implementation of a DTT0BaseCorrection.
 *  Computes correction for t0
 *
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTT0BaseCorrection.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTT0;

namespace dtCalibration {

class DTT0FillDefaultFromDB: public DTT0BaseCorrection {
public:
  // Constructor
  DTT0FillDefaultFromDB(const edm::ParameterSet&);

  // Destructor
  virtual ~DTT0FillDefaultFromDB();

  virtual void setES(const edm::EventSetup& setup);
  virtual DTT0Data correction(const DTWireId&);

private:
  std::string dbLabelRef_;

  const DTT0 *t0MapRef_;
  const DTT0 *t0Map_;
};

} // namespace
#endif
