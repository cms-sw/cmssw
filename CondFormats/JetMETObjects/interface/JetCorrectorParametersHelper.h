//
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: JetCorrectorParameters.h,v 1.15 2012/03/01 18:24:52 srappocc Exp $
//
// Generic parameters for Jet corrections
//
#ifndef JetCorrectorParametersHelper_h
#define JetCorrectorParametersHelper_h

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/Utilities.h"

#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <functional>
#include <iostream>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//---------- JetCorrectorParametersHelper class ----------------
//-- The helper is used to find the correct Record to access --- 
class JetCorrectorParametersHelper
{
  public:
    //-------- Member functions ----------
    unsigned size()                                                                                           const {return mIndexMap.size();}
    void initTransientMaps();
    void init(const JetCorrectorParameters::Definitions& mDefinitions,
              const std::vector<JetCorrectorParameters::Record>& mRecords);
    void checkMiddleBinUniformity(const std::vector<JetCorrectorParameters::Record>& mRecords)                const;
    void binIndexChecks(unsigned N, const std::vector<float>& fX)                                             const;
    bool binBoundChecks(unsigned dim, const float& value, const float& min, const float& max)                 const;
    int  binIndexN(const std::vector<float>& fX, const std::vector<JetCorrectorParameters::Record>& mRecords) const;

    using tuple_type = typename generate_tuple_type<float,JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY>::type;
    using tuple_type_Nm1 = typename generate_tuple_type<float,JetCorrectorParameters::MAX_SIZE_DIMENSIONALITY-1>::type;
  private:
    //-------- Member variables ----------
    // Stores the lower and upper bounds of the bins for each binned dimension
    std::vector<std::vector<float> >                              mBinBoundaries;
    // Maps a set of lower bounds for N binned dimensions to the index in mRecords
    std::unordered_map<tuple_type, size_t>                        mIndexMap;
    // Maps a set of lower bounds for the first N-1 dimensions to the range of lower bound indices mBinBoundaries for the N dimension
    std::unordered_map<tuple_type_Nm1, std::pair<size_t,size_t> > mMap;
    // The number of binned dimensions as given by the JetCorrectorParameters::Definitions class
    unsigned                                                      SIZE;
};

#endif