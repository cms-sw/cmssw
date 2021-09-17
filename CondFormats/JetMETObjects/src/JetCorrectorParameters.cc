//
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: JetCorrectorParameters.cc,v 1.20 2012/03/01 18:24:53 srappocc Exp $
//
// Generic parameters for Jet corrections
//
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParametersHelper.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iterator>

//------------------------------------------------------------------------
//--- JetCorrectorParameters::Definitions constructor --------------------
//--- takes specific arguments for the member variables ------------------
//------------------------------------------------------------------------
JetCorrectorParameters::Definitions::Definitions(const std::vector<std::string>& fBinVar,
                                                 const std::vector<std::string>& fParVar,
                                                 const std::string& fFormula,
                                                 bool fIsResponse) {
  for (unsigned i = 0; i < fBinVar.size(); i++)
    mBinVar.push_back(fBinVar[i]);
  for (unsigned i = 0; i < fParVar.size(); i++)
    mParVar.push_back(fParVar[i]);
  mFormula = fFormula;
  mIsResponse = fIsResponse;
  mLevel = "";
}
//------------------------------------------------------------------------
//--- JetCorrectorParameters::Definitions constructor --------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
JetCorrectorParameters::Definitions::Definitions(const std::string& fLine) {
  std::vector<std::string> tokens = getTokens(fLine);
  if (!tokens.empty()) {
    if (tokens.size() < 6) {
      std::stringstream sserr;
      sserr << "(line " << fLine << "): less than 6 expected tokens:" << tokens.size();
      handleError("JetCorrectorParameters::Definitions", sserr.str());
    }
    unsigned nvar = getUnsigned(tokens[0]);
    unsigned npar = getUnsigned(tokens[nvar + 1]);
    for (unsigned i = 0; i < nvar; i++)
      mBinVar.push_back(tokens[i + 1]);
    for (unsigned i = 0; i < npar; i++)
      mParVar.push_back(tokens[nvar + 2 + i]);
    mFormula = tokens[npar + nvar + 2];
    std::string ss = tokens[npar + nvar + 3];
    if (ss == "Response")
      mIsResponse = true;
    else if (ss == "Correction")
      mIsResponse = false;
    else if (ss == "Resolution")
      mIsResponse = false;
    else if (ss.find("PAR") == 0)
      mIsResponse = false;
    else {
      std::stringstream sserr;
      sserr << "unknown option (" << ss << ")";
      handleError("JetCorrectorParameters::Definitions", sserr.str());
    }
    mLevel = tokens[npar + nvar + 4];
  }
}
//------------------------------------------------------------------------
//--- JetCorrectorParameters::Record constructor -------------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
JetCorrectorParameters::Record::Record(const std::string& fLine, unsigned fNvar) : mMin(0), mMax(0) {
  mNvar = fNvar;
  // quckly parse the line
  std::vector<std::string> tokens = getTokens(fLine);
  if (!tokens.empty()) {
    if (tokens.size() < 3) {
      std::stringstream sserr;
      sserr << "(line " << fLine << "): "
            << "three tokens expected, " << tokens.size() << " provided.";
      handleError("JetCorrectorParameters::Record", sserr.str());
    }
    for (unsigned i = 0; i < mNvar; i++) {
      mMin.push_back(getFloat(tokens[i * 2]));
      mMax.push_back(getFloat(tokens[i * 2 + 1]));
    }
    unsigned nParam = getUnsigned(tokens[2 * mNvar]);
    if (nParam != tokens.size() - (2 * mNvar + 1)) {
      std::stringstream sserr;
      sserr << "(line " << fLine << "): " << tokens.size() - (2 * mNvar + 1) << " parameters, but nParam=" << nParam
            << ".";
      handleError("JetCorrectorParameters::Record", sserr.str());
    }
    for (unsigned i = (2 * mNvar + 1); i < tokens.size(); ++i)
      mParameters.push_back(getFloat(tokens[i]));
  }
}
std::ostream& operator<<(std::ostream& out, const JetCorrectorParameters::Record& fBin) {
  for (unsigned j = 0; j < fBin.nVar(); j++)
    out << fBin.xMin(j) << " " << fBin.xMax(j) << " ";
  out << fBin.nParameters() << " ";
  for (unsigned j = 0; j < fBin.nParameters(); j++)
    out << fBin.parameter(j) << " ";
  out << std::endl;
  return out;
}
//------------------------------------------------------------------------
//--- JetCorrectorParameters constructor ---------------------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
JetCorrectorParameters::JetCorrectorParameters(const std::string& fFile, const std::string& fSection) {
  std::ifstream input(fFile.c_str());
  std::string currentSection = "";
  std::string line;
  std::string currentDefinitions = "";
  while (std::getline(input, line)) {
    if (line.empty())
      continue;
    std::string section = getSection(line);
    std::string tmp = getDefinitions(line);
    if (!section.empty() && tmp.empty()) {
      currentSection = section;
      continue;
    }
    if (currentSection == fSection) {
      if (!tmp.empty()) {
        currentDefinitions = tmp;
        continue;
      }
      Definitions definitions(currentDefinitions);
      if (!(definitions.nBinVar() == 0 && definitions.formula().empty()))
        mDefinitions = definitions;
      Record record(line, mDefinitions.nBinVar());
      bool check(true);
      for (unsigned i = 0; i < mDefinitions.nBinVar(); ++i)
        if (record.xMin(i) == 0 && record.xMax(i) == 0)
          check = false;
      if (record.nParameters() == 0)
        check = false;
      if (check)
        mRecords.push_back(record);
    }
  }
  if (currentDefinitions.empty())
    handleError("JetCorrectorParameters", "No definitions found!!!");
  if (mRecords.empty() && currentSection.empty())
    mRecords.push_back(Record());
  if (mRecords.empty() && !currentSection.empty()) {
    std::stringstream sserr;
    sserr << "the requested section " << fSection << " doesn't exist!";
    handleError("JetCorrectorParameters", sserr.str());
  }
  std::sort(mRecords.begin(), mRecords.end());
  valid_ = true;

  if (mDefinitions.nBinVar() <= MAX_SIZE_DIMENSIONALITY) {
    init();
  } else {
    std::stringstream sserr;
    sserr << "since the binned dimensionality is greater than " << MAX_SIZE_DIMENSIONALITY
          << " the SimpleJetCorrector will default to using the legacy binIndex function!";
    handleError("JetCorrectorParameters", sserr.str());
  }
}
//------------------------------------------------------------------------
//--- initializes the correct JetCorrectorParametersHelper ---------------
//------------------------------------------------------------------------
void JetCorrectorParameters::init() {
  std::sort(mRecords.begin(), mRecords.end());
  helper = std::make_shared<JetCorrectorParametersHelper>();
  helper->init(mDefinitions, mRecords);
}
//------------------------------------------------------------------------
//--- returns the index of the record defined by fX ----------------------
//------------------------------------------------------------------------
int JetCorrectorParameters::binIndex(const std::vector<float>& fX) const {
  int result = -1;
  unsigned N = mDefinitions.nBinVar();
  if (N != fX.size()) {
    std::stringstream sserr;
    sserr << "# bin variables " << N << " doesn't correspont to requested #: " << fX.size();
    handleError("JetCorrectorParameters", sserr.str());
  }
  unsigned tmp;
  for (unsigned i = 0; i < size(); ++i) {
    tmp = 0;
    for (unsigned j = 0; j < N; j++)
      if (fX[j] >= record(i).xMin(j) && fX[j] < record(i).xMax(j))
        tmp += 1;
    if (tmp == N) {
      result = i;
      break;
    }
  }
  return result;
}
//------------------------------------------------------------------------
//--- returns the index of the record defined by fX (non-linear search) --
//------------------------------------------------------------------------
int JetCorrectorParameters::binIndexN(const std::vector<float>& fX) const {
  if (helper->size() > 0) {
    return helper->binIndexN(fX, mRecords);
  } else {
    return -1;
  }
}
//------------------------------------------------------------------------
//--- returns the neighbouring bins of fIndex in the direction of fVar ---
//------------------------------------------------------------------------
int JetCorrectorParameters::neighbourBin(unsigned fIndex, unsigned fVar, bool fNext) const {
  int result = -1;
  unsigned N = mDefinitions.nBinVar();
  if (fVar >= N) {
    std::stringstream sserr;
    sserr << "# of bin variables " << N << " doesn't correspond to requested #: " << fVar;
    handleError("JetCorrectorParameters", sserr.str());
  }
  unsigned tmp;
  for (unsigned i = 0; i < size(); ++i) {
    tmp = 0;
    for (unsigned j = 0; j < fVar; j++)
      if (fabs(record(i).xMin(j) - record(fIndex).xMin(j)) < 0.0001)
        tmp += 1;
    for (unsigned j = fVar + 1; j < N; j++)
      if (fabs(record(i).xMin(j) - record(fIndex).xMin(j)) < 0.0001)
        tmp += 1;
    if (tmp < N - 1)
      continue;
    if (tmp == N - 1) {
      if (fNext)
        if (fabs(record(i).xMin(fVar) - record(fIndex).xMax(fVar)) < 0.0001)
          tmp += 1;
      if (!fNext)
        if (fabs(record(i).xMax(fVar) - record(fIndex).xMin(fVar)) < 0.0001)
          tmp += 1;
    }
    if (tmp == N) {
      result = i;
      break;
    }
  }
  return result;
}
//------------------------------------------------------------------------
//--- returns the number of bins in the direction of fVar ----------------
//------------------------------------------------------------------------
unsigned JetCorrectorParameters::size(unsigned fVar) const {
  if (fVar >= mDefinitions.nBinVar()) {
    std::stringstream sserr;
    sserr << "requested bin variable index " << fVar << " is greater than number of variables "
          << mDefinitions.nBinVar();
    handleError("JetCorrectorParameters", sserr.str());
  }
  unsigned result = 0;
  float tmpMin(-9999), tmpMax(-9999);
  for (unsigned i = 0; i < size(); ++i)
    if (record(i).xMin(fVar) > tmpMin && record(i).xMax(fVar) > tmpMax) {
      result++;
      tmpMin = record(i).xMin(fVar);
      tmpMax = record(i).xMax(fVar);
    }
  return result;
}
//------------------------------------------------------------------------
//--- returns the vector of bin centers of fVar --------------------------
//------------------------------------------------------------------------
std::vector<float> JetCorrectorParameters::binCenters(unsigned fVar) const {
  std::vector<float> result;
  for (unsigned i = 0; i < size(); ++i)
    result.push_back(record(i).xMiddle(fVar));
  return result;
}
//------------------------------------------------------------------------
//--- prints parameters on screen ----------------------------------------
//------------------------------------------------------------------------
void JetCorrectorParameters::printScreen() const {
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "////////  PARAMETERS: //////////////////////" << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Number of binning variables:   " << definitions().nBinVar() << std::endl;
  std::cout << "Names of binning variables:    ";
  for (unsigned i = 0; i < definitions().nBinVar(); i++)
    std::cout << definitions().binVar(i) << " ";
  std::cout << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Number of parameter variables: " << definitions().nParVar() << std::endl;
  std::cout << "Names of parameter variables:  ";
  for (unsigned i = 0; i < definitions().nParVar(); i++)
    std::cout << definitions().parVar(i) << " ";
  std::cout << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Parametrization Formula:       " << definitions().formula() << std::endl;
  if (definitions().isResponse())
    std::cout << "Type (Response or Correction): "
              << "Response" << std::endl;
  else
    std::cout << "Type (Response or Correction): "
              << "Correction" << std::endl;
  std::cout << "Correction Level:              " << definitions().level() << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "------- Bin contents -----------------------" << std::endl;
  for (unsigned i = 0; i < size(); i++) {
    for (unsigned j = 0; j < definitions().nBinVar(); j++)
      std::cout << record(i).xMin(j) << " " << record(i).xMax(j) << " ";
    std::cout << record(i).nParameters() << " ";
    for (unsigned j = 0; j < record(i).nParameters(); j++)
      std::cout << record(i).parameter(j) << " ";
    std::cout << std::endl;
  }
}
//------------------------------------------------------------------------
//--- prints parameters on file ----------------------------------------
//------------------------------------------------------------------------
void JetCorrectorParameters::printFile(const std::string& fFileName) const {
  std::ofstream txtFile;
  txtFile.open(fFileName.c_str());
  txtFile.setf(std::ios::right);
  txtFile << "{" << definitions().nBinVar() << std::setw(15);
  for (unsigned i = 0; i < definitions().nBinVar(); i++)
    txtFile << definitions().binVar(i) << std::setw(15);
  txtFile << definitions().nParVar() << std::setw(15);
  for (unsigned i = 0; i < definitions().nParVar(); i++)
    txtFile << definitions().parVar(i) << std::setw(15);
  txtFile << std::setw(definitions().formula().size() + 15) << definitions().formula() << std::setw(15);
  if (definitions().isResponse())
    txtFile << "Response" << std::setw(15);
  else
    txtFile << "Correction" << std::setw(15);
  txtFile << definitions().level() << "}"
          << "\n";
  for (unsigned i = 0; i < size(); i++) {
    for (unsigned j = 0; j < definitions().nBinVar(); j++)
      txtFile << record(i).xMin(j) << std::setw(15) << record(i).xMax(j) << std::setw(15);
    txtFile << record(i).nParameters() << std::setw(15);
    for (unsigned j = 0; j < record(i).nParameters(); j++)
      txtFile << record(i).parameter(j) << std::setw(15);
    txtFile << "\n";
  }
  txtFile.close();
}

namespace {
  const std::vector<std::string> labels_ = {
      "L1Offset",
      "L2Relative",
      "L3Absolute",
      "L4EMF",
      "L5Flavor",
      "L6UE",
      "L7Parton",
      "L1JPTOffset",
      "L2L3Residual",
      "Uncertainty",
      "L1FastJet",
      "UncertaintyAbsolute",
      "UncertaintyHighPtExtra",
      "UncertaintySinglePionECAL",
      "UncertaintyFlavor",
      "UncertaintyTime",
      "UncertaintyRelativeJEREC1",
      "UncertaintyRelativeJEREC2",
      "UncertaintyRelativeJERHF",
      "UncertaintyRelativeStatEC2",
      "UncertaintyRelativeStatHF",
      "UncertaintyRelativeFSR",
      "UncertaintyPileUpDataMC",
      "UncertaintyPileUpOOT",
      "UncertaintyPileUpPtBB",
      "UncertaintyPileUpBias",
      "UncertaintyPileUpJetRate",
      "UncertaintySinglePionHCAL",
      "UncertaintyRelativePtEC1",
      "UncertaintyRelativePtEC2",
      "UncertaintyRelativePtHF",
      "UncertaintyRelativeSample",
      "UncertaintyPileUpPtEC",
      "UncertaintyPileUpPtHF",
      "L1RC",
      "L1Residual",
      "UncertaintyAux3",
      "UncertaintyAux4",
  };

  const std::vector<std::string> l5Flavors_ = {"L5Flavor_bJ",
                                               "L5Flavor_cJ",
                                               "L5Flavor_qJ",
                                               "L5Flavor_gJ",
                                               "L5Flavor_bT",
                                               "L5Flavor_cT",
                                               "L5Flavor_qT",
                                               "L5Flavor_gT"};

  const std::vector<std::string> l7Partons_ = {"L7Parton_gJ",
                                               "L7Parton_qJ",
                                               "L7Parton_cJ",
                                               "L7Parton_bJ",
                                               "L7Parton_jJ",
                                               "L7Parton_qT",
                                               "L7Parton_cT",
                                               "L7Parton_bT",
                                               "L7Parton_tT"};
}  // namespace
std::string JetCorrectorParametersCollection::findLabel(key_type k) {
  if (isL5(k))
    return findL5Flavor(k);
  else if (isL7(k))
    return findL7Parton(k);
  else
    return labels_[k];
}

std::string JetCorrectorParametersCollection::findL5Flavor(key_type k) {
  if (k == L5Flavor)
    return labels_[L5Flavor];
  else
    return l5Flavors_[k / 100 - 1];
}

std::string JetCorrectorParametersCollection::findL7Parton(key_type k) {
  if (k == L7Parton)
    return labels_[L7Parton];
  else
    return l7Partons_[k / 1000 - 1];
}

void JetCorrectorParametersCollection::getSections(std::string inputFile, std::vector<std::string>& outputs) {
  outputs.clear();
  std::ifstream input(inputFile.c_str());
  while (!input.eof()) {
    char buff[10000];
    input.getline(buff, 10000);
    std::string in(buff);
    if (in[0] == '[') {
      std::string tok = getSection(in);
      if (!tok.empty()) {
        outputs.push_back(tok);
      }
    }
  }
  std::cout << "Found these sections for file: " << std::endl;
  copy(outputs.begin(), outputs.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
}

// Add a JetCorrectorParameter object, possibly with flavor.
void JetCorrectorParametersCollection::push_back(key_type i, value_type const& j, label_type const& flav) {
  std::cout << "i    = " << i << std::endl;
  std::cout << "flav = " << flav << std::endl;
  if (isL5(i)) {
    std::cout << "This is L5, getL5Bin = " << getL5Bin(flav) << std::endl;
    correctionsL5_.push_back(pair_type(getL5Bin(flav), j));
  } else if (isL7(i)) {
    std::cout << "This is L7, getL7Bin = " << getL7Bin(flav) << std::endl;
    correctionsL7_.push_back(pair_type(getL7Bin(flav), j));
  } else if (flav.empty()) {
    corrections_.push_back(pair_type(i, j));
  } else {
    std::cout << "***** NOT ADDING " << flav << ", corresponding position in JetCorrectorParameters is not found."
              << std::endl;
  }
}

// Access the JetCorrectorParameter via the key k.
// key_type is hashed to deal with the three collections
JetCorrectorParameters const& JetCorrectorParametersCollection::operator[](key_type k) const {
  collection_type::const_iterator ibegin, iend, i;
  if (isL5(k)) {
    ibegin = correctionsL5_.begin();
    iend = correctionsL5_.end();
    i = ibegin;
  } else if (isL7(k)) {
    ibegin = correctionsL7_.begin();
    iend = correctionsL7_.end();
    i = ibegin;
  } else {
    ibegin = corrections_.begin();
    iend = corrections_.end();
    i = ibegin;
  }
  for (; i != iend; ++i) {
    if (k == i->first)
      return i->second;
  }
  throw cms::Exception("InvalidInput") << " cannot find key " << static_cast<int>(k)
                                       << " in the JEC payload, this usually means you have to change the global tag"
                                       << std::endl;
}

// Get a list of valid keys. These will contain hashed keys
// that are aware of all three collections.
void JetCorrectorParametersCollection::validKeys(std::vector<key_type>& keys) const {
  keys.clear();
  for (collection_type::const_iterator ibegin = corrections_.begin(), iend = corrections_.end(), i = ibegin; i != iend;
       ++i) {
    keys.push_back(i->first);
  }
  for (collection_type::const_iterator ibegin = correctionsL5_.begin(), iend = correctionsL5_.end(), i = ibegin;
       i != iend;
       ++i) {
    keys.push_back(i->first);
  }
  for (collection_type::const_iterator ibegin = correctionsL7_.begin(), iend = correctionsL7_.end(), i = ibegin;
       i != iend;
       ++i) {
    keys.push_back(i->first);
  }
}

// Find the L5 bin for hashing
JetCorrectorParametersCollection::key_type JetCorrectorParametersCollection::getL5Bin(std::string const& flav) {
  std::vector<std::string>::const_iterator found = find(l5Flavors_.begin(), l5Flavors_.end(), flav);
  if (found != l5Flavors_.end()) {
    return (found - l5Flavors_.begin() + 1) * 100;
  } else
    return L5Flavor;
}
// Find the L7 bin for hashing
JetCorrectorParametersCollection::key_type JetCorrectorParametersCollection::getL7Bin(std::string const& flav) {
  std::vector<std::string>::const_iterator found = find(l7Partons_.begin(), l7Partons_.end(), flav);
  if (found != l7Partons_.end()) {
    return (found - l7Partons_.begin() + 1) * 1000;
  } else
    return L7Parton;
}

// Check if this is an L5 hashed value
bool JetCorrectorParametersCollection::isL5(key_type k) { return k == L5Flavor || (k / 100 > 0 && k / 1000 == 0); }
// Check if this is an L7 hashed value
bool JetCorrectorParametersCollection::isL7(key_type k) { return k == L7Parton || (k / 1000 > 0); }

// Find the key corresponding to each label
JetCorrectorParametersCollection::key_type JetCorrectorParametersCollection::findKey(std::string const& label) const {
  // First check L5 corrections
  std::vector<std::string>::const_iterator found1 = find(l5Flavors_.begin(), l5Flavors_.end(), label);
  if (found1 != l5Flavors_.end()) {
    return getL5Bin(label);
  }

  // Next check L7 corrections
  std::vector<std::string>::const_iterator found2 = find(l7Partons_.begin(), l7Partons_.end(), label);
  if (found2 != l7Partons_.end()) {
    return getL7Bin(label);
  }

  // Finally check the default corrections
  std::vector<std::string>::const_iterator found3 = find(labels_.begin(), labels_.end(), label);
  if (found3 != labels_.end()) {
    return static_cast<key_type>(found3 - labels_.begin());
  }

  // Didn't find default corrections, throw exception
  throw cms::Exception("InvalidInput") << " Cannot find label " << label << std::endl;
}

//#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(JetCorrectorParameters);
TYPELOOKUP_DATA_REG(JetCorrectorParametersCollection);
