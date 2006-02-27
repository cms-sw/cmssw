#ifndef METOBJECTS_BASE_MET_H
#define METOBJECTS_BASE_MET_H

/** \class BaseMET
 *
 * The BaseMET EDProduct type. Stores a few basic variables
 * critical to all higher level MET products.
 *
 * \authors Michael Schmitt, Richard Cavanaugh The University of Florida
 *
 * \version   1st Version May 31st, 2005.
 *
 ************************************************************/

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "DataFormats/METObjects/interface/CommonMETData.h"

#include <vector>
#include <cstring>

class BaseMET 
{
public:
  BaseMET();
  // Setters
  void setLabel(const char *Label) { strcpy( data.label, Label ); }
  void setMET(double MET)     { data.met = MET; } //derived quantity 
  void setMEx(double MEx)     { data.mex = MEx; }
  void setMEy(double MEy)     { data.mey = MEy; }
  void setMEz(double MEz)     { data.mez = MEz; }
  void setSumET(double SumET) { data.sumet = SumET; }
  void setPhi(double Phi)     { data.phi = Phi; } //derived quantity
  void pushDelta() { corr.push_back( data ); }
  // Getters
  char *getLabel()        { return data.label; }
  double getMET()   const { return data.met; }
  double getMEx()   const { return data.mex; }
  double getMEy()   const { return data.mey; }
  double getMEz()   const { return data.mez; }
  double getSumET() const { return data.sumet; }
  double getPhi()   const { return data.phi; }
  std::vector<CommonMETData> getAllCorr() const {return corr;}
  // Methods
  virtual void clearMET();
private:
  CommonMETData data;
  std::vector<CommonMETData> corr;
};

#endif // METOBJECTS_MET_H
