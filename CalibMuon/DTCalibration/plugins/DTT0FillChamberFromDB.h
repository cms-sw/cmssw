#ifndef CalibMuon_DTT0FillChamberFromDB_H
#define CalibMuon_DTT0FillChamberFromDB_H

/** \class DTT0FillChamberFromDB
 *  Concrete implementation of a DTT0BaseCorrection.
 *  Computes correction for t0
 *
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTT0BaseCorrection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include <string>

namespace edm {
  class ParameterSet;
}

class DTT0;

namespace dtCalibration {

class DTT0FillChamberFromDB: public DTT0BaseCorrection {
public:
  // Constructor
  DTT0FillChamberFromDB(const edm::ParameterSet&);

  // Destructor
  virtual ~DTT0FillChamberFromDB();

  virtual void setES(const edm::EventSetup& setup);
  virtual DTT0Data correction(const DTWireId&);

private:
  std::string dbLabelRef_;
  std::string chamberRef_;

  DTChamberId chosenChamberId_;

  const DTT0 *t0MapRef_;
  const DTT0 *t0Map_;
};

} // namespace
#endif
