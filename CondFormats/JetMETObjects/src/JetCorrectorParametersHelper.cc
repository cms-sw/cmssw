//
// Original Author:  Alexx Perloff Feb 22, 2017
// $Id: JetCorrectorParameters.cc,v 1.20 2012/03/01 18:24:53 srappocc Exp $
//
// Helper class for JetCorrectorParameters
//
#include "CondFormats/JetMETObjects/interface/JetCorrectorParametersHelper.h"
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iterator>

//------------------------------------------------------------------------
//--- JetCorrectorParameters::JetCorrectorParametersHelper initializer ---
//--- initializes the mBinMap for quick lookup of mRecords ---------------
//------------------------------------------------------------------------
void JetCorrectorParametersHelper::initTransientMaps() {
  mIndexMap.clear();
  mMap.clear();
  mBinBoundaries.clear();
  mBinBoundaries.assign(JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY, std::vector<float>(0, 0));
}
void JetCorrectorParametersHelper::init(const JetCorrectorParameters::Definitions& mDefinitionsLocal,
                                        const std::vector<JetCorrectorParameters::Record>& mRecordsLocal) {
  SIZE = mDefinitionsLocal.nBinVar();
  if (SIZE > JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY) {
    std::stringstream sserr;
    sserr << "The number of binned variables requested (" << SIZE << ") is greater than the number allowed ("
          << JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY << ")";
    handleError("JetCorrectorParametersHelper", sserr.str());
  }

  initTransientMaps();
  size_t start = 0, end = 0;
  size_t nRec = mRecordsLocal.size();
  size_t indexMapSize = 0, tmpIndexMapSize = 0;
  for (unsigned i = 0; i < nRec; ++i) {
    for (unsigned j = 0; j < SIZE; j++) {
      if (j < SIZE - 1 && std::find(mBinBoundaries[j].begin(), mBinBoundaries[j].end(), mRecordsLocal[i].xMin(j)) ==
                              mBinBoundaries[j].end())
        mBinBoundaries[j].push_back(mRecordsLocal[i].xMin(j));
      else if (j == SIZE - 1) {
        if (i == 0)
          mBinBoundaries[j].reserve(mRecordsLocal.size());

        mBinBoundaries[j].push_back(mRecordsLocal[i].xMin(j));

        if (SIZE > 1 && (i == nRec - 1 || mRecordsLocal[i].xMin(j - 1) != mRecordsLocal[i + 1].xMin(j - 1))) {
          end = i;
          //mMap.emplace(gen_tuple<SIZE-1>([&](size_t k){return mRecordsLocal[i].xMin(k);}),std::make_pair(start,end));
          mMap.emplace(gen_tuple<JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY - 1>(
                           [&](size_t k) { return (k < SIZE - 1) ? mRecordsLocal[i].xMin(k) : -9999; }),
                       std::make_pair(start, end));
          start = i + 1;
        }
      }
    }
    indexMapSize = mIndexMap.size();
    tuple_type tmpTuple = gen_tuple<JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY>(
        [&](size_t k) { return (k < SIZE) ? mRecordsLocal[i].xMin(k) : -9999; });
    mIndexMap.emplace(tmpTuple, i);
    tmpIndexMapSize = mIndexMap.size();
    if (indexMapSize == tmpIndexMapSize) {
      size_t existing_index = mIndexMap.find(tmpTuple)->second;
      std::stringstream sserr;
      sserr << "Duplicate binning in record found (existing index,current index)=(" << existing_index << "," << i << ")"
            << std::endl
            << "\tBins(lower bounds)=" << tmpTuple;
      handleError("JetCorrectorParametersHelper", sserr.str());
    }
  }
  if (mBinBoundaries[SIZE - 1].size() != nRec) {
    std::stringstream sserr;
    sserr << "Did not find all bin boundaries for dimension " << SIZE - 1 << "!!!" << std::endl
          << "Found " << mBinBoundaries[SIZE - 1].size() << " out of " << nRec << " records";
    handleError("JetCorrectorParametersHelper", sserr.str());
  }
  indexMapSize = mIndexMap.size();
  if (indexMapSize != nRec) {
    handleError("JetCorrectorParametersHelper",
                "The mapping of bin lower bounds to indices does not contain all possible entries!!!");
  }
  // This function checks that all of the middle binned parameters (not the first or last) contain the same range of bins.
  // If not an exception will be thrown as the mapping will work only if all bins but the last are uniform.
  if (SIZE > 2)
    checkMiddleBinUniformity(mRecordsLocal);
}

void JetCorrectorParametersHelper::checkMiddleBinUniformity(
    const std::vector<JetCorrectorParameters::Record>& mRecords) const {
  unsigned N = SIZE - 2;
  size_t nRec = mRecords.size();
  std::vector<int> fN(N, -1);
  //The order of looping (records or dimensions) does not matter because you have to go through all of them once anyway
  //Loop over each binned dimension that isn't the first or the last
  for (unsigned idim = 1; idim <= N; idim++) {
    int nBoundaryPassed = 0;
    if (fN[idim - 1] == -1)
      fN[idim - 1] = mBinBoundaries[idim].size();
    //Loop over the mRecords vector
    for (unsigned iRecord = 0; iRecord < nRec; iRecord++) {
      if (mRecords[iRecord].xMin(idim) != mBinBoundaries[idim][nBoundaryPassed % fN[idim - 1]]) {
        throw cms::Exception("MissingRecord")
            << "found a missing record in binned dimension " << idim << " after record " << iRecord << std::endl
            << "\tthe bin lower bound should have been " << mBinBoundaries[idim][nBoundaryPassed % fN[idim - 1]]
            << ", but was instead " << mRecords[iRecord].xMin(idim) << std::endl
            << "\tall binned dimensions, besides the last one, must have uniform binning." << std::endl
            << mRecords[iRecord - 1] << std::endl
            << mRecords[iRecord] << std::endl;
      } else if ((iRecord == nRec - 1 || mRecords[iRecord].xMin(idim) != mRecords[iRecord + 1].xMin(idim))) {
        nBoundaryPassed++;
      } else {
        continue;
      }
    }
  }
}

//------------------------------------------------------------------------
//--- JetCorrectorParameters::JetCorrectorParametersHelper sanity checks -
//--- checks that some conditions are met before finding the bin index ---
//------------------------------------------------------------------------
void JetCorrectorParametersHelper::binIndexChecks(unsigned N, const std::vector<float>& fX) const {
  if (N != fX.size()) {
    std::stringstream sserr;
    sserr << "The number of binned variables, " << N << ", doesn't correspond to the number requested, " << fX.size();
    handleError("JetCorrectorParametersHelper", sserr.str());
  }
}
bool JetCorrectorParametersHelper::binBoundChecks(unsigned dim,
                                                  const float& value,
                                                  const float& min,
                                                  const float& max) const {
  if (value < min || value >= max)
    return false;
  else
    return true;
}
//------------------------------------------------------------------------
//--- JetCorrectorParameters::JetCorrectorParametersHelper binIndexN -----
//--- returns the index of the record defined by fX (non-linear search) --
//------------------------------------------------------------------------
int JetCorrectorParametersHelper::binIndexN(const std::vector<float>& fX,
                                            const std::vector<JetCorrectorParameters::Record>& mRecords) const {
  unsigned Nm1 = SIZE - 1;
  binIndexChecks(SIZE, fX);

  //Create a container for the indices
  std::vector<float> fN(SIZE, -1);
  std::vector<float>::const_iterator tmpIt;

  // make sure that fX are within the first and last boundaries of mBinBoundaries (other than last dimension)
  for (unsigned idim = 0; idim == 0 || idim < fX.size() - 1; idim++) {
    if (!binBoundChecks(idim, fX[idim], *mBinBoundaries[idim].begin(), mRecords[size() - 1].xMax(idim)))
      return -1;
    tmpIt = std::lower_bound(mBinBoundaries[idim].begin(), mBinBoundaries[idim].end(), fX[idim]);
    // lower_bound finds the entry with the next highest value to fX[0]
    // so unless the two values are equal, you want the next lowest bin boundary
    if (tmpIt == mBinBoundaries[idim].end())
      tmpIt = mBinBoundaries[idim].begin() + mBinBoundaries[idim].size() - 1;
    else if (*tmpIt != fX[idim])
      tmpIt -= 1;
    fN[idim] = *tmpIt;
  }

  //find the index bounds for the possible values of the last dimension
  std::pair<size_t, size_t> indexBounds;
  if (SIZE > 1) {
    tuple_type_Nm1 to_find_Nm1 = gen_tuple<JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY - 1>(
        [&](size_t i) { return (i < Nm1) ? fN[i] : -9999; });
    if (mMap.find(to_find_Nm1) != mMap.end())
      indexBounds = mMap.at(to_find_Nm1);
    else {
      std::stringstream sserr;
      sserr << "couldn't find the index boundaries for dimension " << Nm1 << std::endl
            << "looking for last bin with N-1 values of " << to_find_Nm1 << std::endl;
      handleError("JetCorrectorParametersHelper", sserr.str());
      return -1;
    }

    //Check that the requested value is within the bin boundaries for the last dimension
    if (!binBoundChecks(Nm1, fX[Nm1], mRecords[indexBounds.first].xMin(Nm1), mRecords[indexBounds.second].xMax(Nm1)))
      return -1;
    tmpIt = std::lower_bound(
        mBinBoundaries[Nm1].begin() + indexBounds.first, mBinBoundaries[Nm1].begin() + indexBounds.second, fX[Nm1]);
    if (*tmpIt != fX[Nm1] && fX[Nm1] < *(mBinBoundaries[Nm1].begin() + indexBounds.second))
      tmpIt -= 1;
    fN[Nm1] = *tmpIt;
  }

  tuple_type to_find =
      gen_tuple<JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY>([&](size_t i) { return (i < SIZE) ? fN[i] : -9999; });
  return (mIndexMap.find(to_find) != mIndexMap.end()) ? mIndexMap.at(to_find) : -1;
}
