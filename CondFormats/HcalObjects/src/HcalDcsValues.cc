#include <algorithm>
#include <set>
#include "CondFormats/HcalObjects/interface/HcalDcsValues.h"

HcalDcsValues::HcalDcsValues() : 
  mHBsorted(false),mHEsorted(false),mHO0sorted(false), mHO12sorted(false),
  mHFsorted(false) {
}

HcalDcsValues::~HcalDcsValues() {
}

bool HcalDcsValues::addValue(HcalDcsValue const& newVal) {
  HcalDcsDetId tempId( newVal.DcsId() );
  switch (tempId.subdet()) {
  case HcalDcsBarrel :
    mHBValues.push_back(newVal);
    mHBsorted=false;
    return true;
  case HcalDcsEndcap :
    mHEValues.push_back(newVal);
    mHEsorted=false;
    return true;
  case HcalDcsOuter :
    if (tempId.ring() == 0) {
      mHO0Values.push_back(newVal);
      mHO0sorted=false;
    } else {
      mHO12Values.push_back(newVal);
      mHO12sorted=false;
    }
    return true;
  case HcalDcsForward :
    mHFValues.push_back(newVal);
    mHFsorted=false;
    return true;
  default : 
    return false;
  }
}

void HcalDcsValues::sortAll() {
  mHBsorted = sortList(mHBValues);
  mHEsorted = sortList(mHEValues);
  mHO0sorted = sortList(mHO0Values);
  mHO12sorted = sortList(mHO12Values);
  mHFsorted = sortList(mHFValues);
}

bool HcalDcsValues::exists(HcalDcsDetId const& fid) {
  switch (fid.subdet()) {
  case HcalDcsBarrel : 
    if (!mHBsorted) mHBsorted = sortList(mHBValues);
    return foundDcsId(mHBValues, fid);
  case HcalDcsEndcap : 
    if (!mHEsorted) mHEsorted = sortList(mHEValues);
    return foundDcsId(mHEValues, fid);
  case HcalDcsOuter : 
    if (fid.ring() == 0) {
      if (!mHO0sorted) mHO0sorted = sortList(mHO0Values);
      return foundDcsId(mHO0Values, fid);
    } else {
      if (!mHO12sorted) mHO12sorted = sortList(mHO12Values);
      return foundDcsId(mHO12Values, fid);
    }
  case HcalDcsForward : 
    if (!mHFsorted) mHFsorted = sortList(mHFValues);
    return foundDcsId(mHBValues, fid);
  default : return false;
  }
}

HcalDcsValues::DcsSet HcalDcsValues::getValues(HcalDcsDetId const& fid) {
  DcsSet * valList = 0;
  switch(fid.subdet()) {
  case HcalDcsBarrel : 
    valList = &mHBValues; 
    if (!mHBsorted) mHBsorted = sortList(mHBValues);
    break;
  case HcalDcsEndcap : 
    valList = &mHEValues; 
    if (!mHEsorted) mHEsorted = sortList(mHEValues);
    break;
  case HcalDcsOuter :
    if (fid.ring() == 0) {
      valList = &mHO0Values;
      if (!mHO0sorted) mHO0sorted = sortList(mHO0Values);
    } else {
      valList = &mHO12Values;
      if (!mHO12sorted) mHO12sorted = sortList(mHO12Values);
    }
    break;
  case HcalDcsForward : 
    valList = &mHFValues; 
    if (!mHFsorted) mHFsorted = sortList(mHFValues);
    break;
  default : valList = 0;
  }

  if (valList) {
    HcalDcsValue dummy(fid.rawId(), -1, 0., 0., 0.);
    DcsSet::const_iterator lb = lower_bound(valList->begin(), valList->end(), dummy);
    if ((lb != valList->end()) && (lb->DcsId() == fid.rawId())) {
      DcsSet::const_iterator ub = upper_bound(valList->begin(), valList->end(), dummy);
      return DcsSet(lb, ub) ;
    }
  }
  return DcsSet() ;
}

// std::set<uint32_t> HcalDcsValues::getAllIds() const {
//   std::set<uint32_t> chans;
//   std::pair< set<uint32_t>::iterator, bool > retVal;
//   std::set<uint32_t>::iterator currId = 0;
//   DcsSet::const_iterator currRec;
//   for (currRec = mHBValues.begin(); currRec != mHBValues.end(); ++currRec) {
//     if (!currId) {
//       retVal = chans.insert(currRec->DcsId());
//       currId = retVal.first;
//     } else
//       currId = chans.insert(currId, currRec->Dcs());
//   }
//   for (currRec = mHEValues.begin(); currRec != mHEValues.end(); ++currRec) {
//     if (!currId) {
//       retVal = chans.insert(currRec->DcsId());
//       currId = retVal.first;
//     } else
//       currId = chans.insert(currId, currRec->Dcs());
//   }
//   for (currRec = mHO0Values.begin(); currRec != mHO0Values.end(); ++currRec) {
//     if (!currId) {
//       retVal = chans.insert(currRec->DcsId());
//       currId = retVal.first;
//     } else
//       currId = chans.insert(currId, currRec->Dcs());
//   }
//   for (currRec = mHO12Values.begin(); currRec != mHO12Values.end(); ++currRec) {
//     if (!currId) {
//       retVal = chans.insert(currRec->DcsId());
//       currId = retVal.first;
//     } else
//       currId = chans.insert(currId, currRec->Dcs());
//   }
//   for (currRec = mHFValues.begin(); currRec != mHFValues.end(); ++currRec) {
//     if (!currId) {
//       retVal = chans.insert(currRec->DcsId());
//       currId = retVal.first;
//     } else
//       currId = chans.insert(currId, currRec->Dcs());
//   }
//   return chans;
// }


HcalDcsValues::DcsSet const & HcalDcsValues::getAllSubdetValues(DcsSubDet subd) const {
  switch (subd) {
  case HcalHB : return mHBValues;
  case HcalHE : return mHEValues;
  case HcalHO0 : return mHO0Values;
  case HcalHO12 : return mHO12Values;
  case HcalHF : return mHFValues;
  }
  return mHBValues;
}

bool HcalDcsValues::DcsValuesOK(DcsSubDet subd, int LS) {
  switch (subd) {
  case HcalHB : 
    if (!mHBsorted) mHBsorted = sortList(mHBValues);
    return subDetOk(mHBValues, LS);
  case HcalHE : 
    if (!mHEsorted) mHEsorted = sortList(mHEValues);
    return subDetOk(mHEValues, LS);
  case HcalHO0 : 
    if (!mHO0sorted) mHO0sorted = sortList(mHO0Values);
    return subDetOk(mHO0Values, LS);
  case HcalHO12 : 
    if (!mHO12sorted) mHO12sorted = sortList(mHO12Values);
    return subDetOk(mHO12Values, LS);
  case HcalHF : 
    if (!mHFsorted) mHFsorted = sortList(mHFValues);
    return subDetOk(mHFValues, LS);
  default : return false;
  }
  return false;
}

bool HcalDcsValues::foundDcsId(DcsSet const& valList, 
			       HcalDcsDetId const& fid) const {
  HcalDcsValue dummy(fid.rawId(), -1, 0., 0., 0.);
  DcsSet::const_iterator lb = lower_bound(valList.begin(), valList.end(), dummy);
  if ((lb != valList.end()) && (lb->DcsId() == fid.rawId()))
    return true;
  return false;
}

bool HcalDcsValues::subDetOk(DcsSet const& valList, int LS) const {
  std::set<uint32_t> badIds;
  DcsSet::const_iterator val = valList.begin();
  while ( (val != valList.end()) &&
	  ( (LS>-1) ? (val->LS()<=LS) : true) ) {
    if (!val->isValueGood()) {
      badIds.insert(val->DcsId());
    } else {
      std::set<uint32_t>::iterator fnd = badIds.find(val->DcsId());
      if (*fnd == val->DcsId()) badIds.erase(fnd);
    }
    ++val;
  }
  return (badIds.size()==0);
}

bool HcalDcsValues::sortList(DcsSet& valList) const {
  sort(valList.begin(), valList.end());
  return true;
}
