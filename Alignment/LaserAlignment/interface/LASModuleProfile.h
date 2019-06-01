
#ifndef __LASMODULEPROFILE_H
#define __LASMODULEPROFILE_H

#include <vector>

class LASModuleProfile {
  ///
  /// container class for a LAS
  /// SiStrip module's 512 strip signals
  ///

public:
  LASModuleProfile();
  LASModuleProfile(double*);
  LASModuleProfile(int*);
  void SetData(double*);
  void SetData(int*);
  double GetValue(unsigned int theStripNumber) const { return (data[theStripNumber]); }  // return an element
  void SetValue(unsigned int theStripNumber, const double& theValue) { data.at(theStripNumber) = theValue; }
  void SetAllValuesTo(const double&);
  void DumpToArray(double[512]);
  LASModuleProfile& operator=(const LASModuleProfile&);
  LASModuleProfile operator+(const LASModuleProfile&);
  LASModuleProfile operator-(const LASModuleProfile&);
  LASModuleProfile operator+(const double[512]);
  LASModuleProfile operator-(const double[512]);
  LASModuleProfile& operator+=(const LASModuleProfile&);
  LASModuleProfile& operator-=(const LASModuleProfile&);
  LASModuleProfile& operator+=(const double[512]);
  LASModuleProfile& operator-=(const double[512]);
  LASModuleProfile& operator+=(const int[512]);
  LASModuleProfile& operator-=(const int[512]);
  LASModuleProfile& operator/=(const double);

private:
  void Init(void);
  std::vector<double> data;
};

#endif
