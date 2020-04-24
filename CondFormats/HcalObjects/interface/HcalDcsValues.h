// -*- C++ -*-
#ifndef HcalDcsValues_h
#define HcalDcsValues_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>
//#include <set>
#include <vector>
#include "CondFormats/HcalObjects/interface/HcalDcsValue.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDcsDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

/*
  \class: HcalDcsValues
  \author: Jacob Anderson

  A container class for holding the dcs values from the database.  The
  values are organized by subdetector, and sorted by HcalDcsDetId.
  There is also checking functionality to be able to see that the
  values are witin their set bounds.
 */

class HcalDcsValues {
public:
  typedef std::vector<HcalDcsValue> DcsSet;

  //The subdetectors of interest to the Hcal run certification
  enum DcsSubDet{ HcalHB = 1, HcalHE = 2, HcalHO0 = 3, HcalHO12 = 4,
		  HcalHF = 5 };

  HcalDcsValues();

  virtual ~HcalDcsValues();

  // add a new value to the appropriate list
  bool addValue(HcalDcsValue const& newVal);
  void sortAll();
  // check if a given id has any entries in a list
  bool exists(HcalDcsDetId const& fid);
  // get a list of values that are for the give id
  DcsSet getValues(HcalDcsDetId const& fid);

  DcsSet const & getAllSubdetValues(DcsSubDet subd) const;

  std::string myname() const { return (std::string)"HcalDcsValues"; }

  //Check the values of a subdetector.  If LS is -1 then for the whole run
  //otherwise for the given LS.
  bool DcsValuesOK(DcsSubDet subd, int LS = -1);
  //bool DcsValuesOK(HcalDetID dataId, DcsMap, int LS = -1) const;

protected:
  bool foundDcsId(DcsSet const& valList, HcalDcsDetId const& fid) const;
  bool subDetOk(DcsSet const& valList, int LS) const;
  bool sortList(DcsSet& valList) const;

private:
  DcsSet mHBValues;
  bool mHBsorted;
  DcsSet mHEValues;
  bool mHEsorted;
  DcsSet mHO0Values;
  bool mHO0sorted;
  DcsSet mHO12Values;
  bool mHO12sorted;
  DcsSet mHFValues;
  bool mHFsorted;

  COND_SERIALIZABLE;
};

#endif
