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

#include "DataFormats/METObjects/interface/CommonMETData.h"

#include <vector>
#include <cstring>

class BaseMETv0 
{
public:
  BaseMETv0();
  // Setters
  //void setLabel(const char *Label) { strcpy( data.label, Label ); }
  void setMET(double MET)     { data.met = MET; } //derived quantity 
  void setMEx(double MEx)     { data.mex = MEx; }
  void setMEy(double MEy)     { data.mey = MEy; }
  void setMEz(double MEz)     { data.mez = MEz; }
  void setSumET(double SumET) { data.sumet = SumET; }
  void setPhi(double Phi)     { data.phi = Phi; } //derived quantity
  void pushDelta() { corr.push_back( data ); }
  // Getters
  //char *getLabel()        { return data.label; }
  double MET()   const { return data.met; }
  double MEx()   const { return data.mex; }
  double MEy()   const { return data.mey; }
  double MEz()   const { return data.mez; }
  double SumET() const { return data.sumet; }
  double phi()   const { return data.phi; }
  std::vector<CommonMETv0Data> getAllCorr() const {return corr;}
  // Methods
  void clearMET();
private:
  CommonMETv0Data data;
  std::vector<CommonMETv0Data> corr;
};

#endif // METOBJECTS_MET_H
