#ifndef METOBJECTS_BASE_MET_H
#define METOBJECTS_BASE_MET_H

/** \class BaseMET
 *
 * The BaseMET EDProduct type. Stores a few basic variables
 * critical to all higher level MET products.
 *
 * \author Michael Schmitt, The University of Florida
 *
 * \version   1st Version May 31st, 2005.
 *
 ************************************************************/

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "DataFormats/METObjects/interface/CommonMETData.h"

#include <cstring>

//class BaseMET: public edm::EDProduct {
class BaseMET {

public:

  BaseMET();

  // Setters
  void setLabel(const char *Label) { strcpy(mdata.label,Label); }

  void setMET(double MET) { mdata.met = MET; }
  void setMETx(double METx) { mdata.metx = METx; }
  void setMETy(double METy) { mdata.mety = METy; }
  void setMETz(double METz) { mdata.metz = METz; }
  void setSumEt(double SumEt) { mdata.sumet = SumEt; }
  void setPhi(double Phi) { mdata.phi = Phi; }

  // Getters
  char *getLabel() const { return mdata.label; }

  double getMET() const { return mdata.met; }
  double getMETx() const { return mdata.metx; }
  double getMETy() const { return mdata.mety; }
  double getMETz() const { return mdata.metz; }
  double getSumEt() const { return mdata.sumet; }
  double getPhi() const { return mdata.phi; }

  // Methods
  virtual void clearMET();

private:

  CommonMETData mdata;

};

#endif // METOBJECTS_MET_H
