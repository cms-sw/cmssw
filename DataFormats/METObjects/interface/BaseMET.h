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
  void setLabel(const char *Label) { strcpy(data.label,Label); }

  void setMET(double MET) { data.met = MET; }
  void setMETx(double METx) { data.metx = METx; }
  void setMETy(double METy) { data.mety = METy; }
  void setMETz(double METz) { data.metz = METz; }
  void setSumEt(double SumEt) { data.sumet = SumEt; }
  void setPhi(double Phi) { data.phi = Phi; }

  // Getters
  char *getLabel() { return data.label; }

  double getMET() const { return data.met; }
  double getMETx() const { return data.metx; }
  double getMETy() const { return data.mety; }
  double getMETz() const { return data.metz; }
  double getSumEt() const { return data.sumet; }
  double getPhi() const { return data.phi; }

  // Methods
  virtual void clearMET();

private:

  CommonMETData data;

};

#endif // METOBJECTS_MET_H
