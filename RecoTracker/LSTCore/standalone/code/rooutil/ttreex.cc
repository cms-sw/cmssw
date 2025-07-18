//  .
// ..: P. Chang, philip@physics.ucsd.edu

#include "ttreex.h"

using namespace RooUtil;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//
// TTree++ (TTreeX) class
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////

//_________________________________________________________________________________________________
RooUtil::TTreeX::TTreeX() { ttree = 0; }

//_________________________________________________________________________________________________
RooUtil::TTreeX::TTreeX(TString treename, TString title) { ttree = new TTree(treename.Data(), title.Data()); }

//_________________________________________________________________________________________________
RooUtil::TTreeX::TTreeX(TTree* tree) { ttree = tree; }

//_________________________________________________________________________________________________
RooUtil::TTreeX::~TTreeX() {}

//_________________________________________________________________________________________________
void* RooUtil::TTreeX::getValPtr(TString brname) {
  TBranch* br = ttree->GetBranch(brname);
  unsigned int nleaves = br->GetListOfLeaves()->GetEntries();
  if (nleaves != 1)
    RooUtil::error("# of leaf for this branch=" + brname + " is not equals to 1!", __FUNCTION__);
  if (!(((TLeaf*)br->GetListOfLeaves()->At(0))->GetValuePointer()))
    ttree->GetEntry(0);
  return ((TLeaf*)br->GetListOfLeaves()->At(0))->GetValuePointer();
}

//__________________________________________________________________________________________________
void RooUtil::TTreeX::clear() {
  for (auto& pair : mapInt_t)
    pair.second = -999;
  for (auto& pair : mapBool_t)
    pair.second = 0;
  for (auto& pair : mapFloat_t)
    pair.second = -999;
  for (auto& pair : mapTString)
    pair.second = "";
  for (auto& pair : mapLV)
    pair.second.SetXYZT(0, 0, 0, 0);
  for (auto& pair : mapTBits)
    pair.second = 0;
  for (auto& pair : mapULL)
    pair.second = 0;
  for (auto& pair : mapUI)
    pair.second = 0;
  for (auto& pair : mapVecInt_t)
    pair.second.clear();
  for (auto& pair : mapVecUInt_t)
    pair.second.clear();
  for (auto& pair : mapVecBool_t)
    pair.second.clear();
  for (auto& pair : mapVecFloat_t)
    pair.second.clear();
  for (auto& pair : mapVecTString)
    pair.second.clear();
  for (auto& pair : mapVecLV)
    pair.second.clear();
  for (auto& pair : mapVecVInt)
    pair.second.clear();
  for (auto& pair : mapVecVFloat)
    pair.second.clear();
  for (auto& pair : mapIsBranchSet)
    pair.second = false;
}

//__________________________________________________________________________________________________
void RooUtil::TTreeX::save(TFile* ofile) {
  RooUtil::print(Form("TTreeX::save() saving tree to %s", ofile->GetName()));
  ofile->cd();
  this->ttree->Write();
}

//__________________________________________________________________________________________________
void TTreeX::sortVecBranchesByPt(TString p4_bn,
                                 std::vector<TString> aux_float_bns,
                                 std::vector<TString> aux_int_bns,
                                 std::vector<TString> aux_bool_bns) {
  // https://stackoverflow.com/questions/236172/how-do-i-sort-a-stdvector-by-the-values-of-a-different-stdvector
  // The first argument is the p4 branches
  // The rest of the argument holds the list of auxilary branches that needs to be sorted together.

  // Creating a "ordered" index list
  std::vector<std::pair<size_t, lviter>> order(mapVecLV[p4_bn.Data()].size());

  size_t n = 0;
  for (lviter it = mapVecLV[p4_bn.Data()].begin(); it != mapVecLV[p4_bn.Data()].end(); ++it, ++n)
    order[n] = make_pair(n, it);

  sort(order.begin(), order.end(), ordering());

  // Sort!
  mapVecLV[p4_bn.Data()] = sortFromRef<LV>(mapVecLV[p4_bn.Data()], order);

  for (auto& aux_float_bn : aux_float_bns)
    mapVecFloat_t[aux_float_bn.Data()] = sortFromRef<Float_t>(mapVecFloat_t[aux_float_bn.Data()], order);

  for (auto& aux_int_bn : aux_int_bns)
    mapVecInt_t[aux_int_bn.Data()] = sortFromRef<Int_t>(mapVecInt_t[aux_int_bn.Data()], order);

  for (auto& aux_bool_bn : aux_bool_bns)
    mapVecBool_t[aux_bool_bn.Data()] = sortFromRef<Bool_t>(mapVecBool_t[aux_bool_bn.Data()], order);
}

//_________________________________________________________________________________________________
void TTreeX::createFlatBranch(std::vector<TString> p4_bns,
                              std::vector<TString> float_bns,
                              std::vector<TString> int_bns,
                              std::vector<TString> bool_bns,
                              int multiplicity) {
  for (auto& p4_bn : p4_bns) {
    for (int i = 0; i < multiplicity; ++i)
      createBranch<LV>(TString::Format("%s_%d", p4_bn.Data(), i));
  }
  for (auto& float_bn : float_bns) {
    for (int i = 0; i < multiplicity; ++i)
      createBranch<float>(TString::Format("%s_%d", float_bn.Data(), i));
  }
  for (auto& int_bn : int_bns) {
    for (int i = 0; i < multiplicity; ++i)
      createBranch<int>(TString::Format("%s_%d", int_bn.Data(), i));
  }
  for (auto& bool_bn : bool_bns) {
    for (int i = 0; i < multiplicity; ++i)
      createBranch<bool>(TString::Format("%s_%d", bool_bn.Data(), i));
  }
}

//_________________________________________________________________________________________________
void TTreeX::setFlatBranch(std::vector<TString> p4_bns,
                           std::vector<TString> float_bns,
                           std::vector<TString> int_bns,
                           std::vector<TString> bool_bns,
                           int multiplicity) {
  for (auto& p4_bn : p4_bns) {
    const std::vector<LV>& vec = getBranch<std::vector<LV>>(p4_bn, false);
    for (int i = 0; i < multiplicity && i < (int)vec.size(); ++i)
      setBranch<LV>(TString::Format("%s_%d", p4_bn.Data(), i), vec[i]);
  }
  for (auto& float_bn : float_bns) {
    const std::vector<float>& vec = getBranch<std::vector<float>>(float_bn, false);
    for (int i = 0; i < multiplicity && i < (int)vec.size(); ++i)
      setBranch<float>(TString::Format("%s_%d", float_bn.Data(), i), vec[i]);
  }
  for (auto& int_bn : int_bns) {
    const std::vector<int>& vec = getBranch<std::vector<int>>(int_bn, false);
    for (int i = 0; i < multiplicity && i < (int)vec.size(); ++i)
      setBranch<int>(TString::Format("%s_%d", int_bn.Data(), i), vec[i]);
  }
  for (auto& bool_bn : bool_bns) {
    const std::vector<bool>& vec = getBranch<std::vector<bool>>(bool_bn, false);
    for (int i = 0; i < multiplicity && i < (int)vec.size(); ++i)
      setBranch<bool>(TString::Format("%s_%d", bool_bn.Data(), i), vec[i]);
  }
}

//_________________________________________________________________________________________________
template <>
void TTreeX::setBranch<Int_t>(TString bn, Int_t val, bool force, bool ignore) {
  if (force) {
    mapInt_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapInt_t.find(bn.Data()) != mapInt_t.end()) {
    mapInt_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<Bool_t>(TString bn, Bool_t val, bool force, bool ignore) {
  if (force) {
    mapBool_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapBool_t.find(bn.Data()) != mapBool_t.end()) {
    mapBool_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<Float_t>(TString bn, Float_t val, bool force, bool ignore) {
  if (force) {
    mapFloat_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapFloat_t.find(bn.Data()) != mapFloat_t.end()) {
    mapFloat_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<TString>(TString bn, TString val, bool force, bool ignore) {
  if (force) {
    mapTString[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapTString.find(bn.Data()) != mapTString.end()) {
    mapTString[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<LV>(TString bn, LV val, bool force, bool ignore) {
  if (force) {
    mapLV[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapLV.find(bn.Data()) != mapLV.end()) {
    mapLV[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<TBits>(TString bn, TBits val, bool force, bool ignore) {
  if (force) {
    mapTBits[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapTBits.find(bn.Data()) != mapTBits.end()) {
    mapTBits[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<unsigned long long>(TString bn, unsigned long long val, bool force, bool ignore) {
  if (force) {
    mapULL[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapULL.find(bn.Data()) != mapULL.end()) {
    mapULL[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<unsigned int>(TString bn, unsigned int val, bool force, bool ignore) {
  if (force) {
    mapUI[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapUI.find(bn.Data()) != mapUI.end()) {
    mapUI[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<std::vector<Int_t>>(TString bn, std::vector<Int_t> val, bool force, bool ignore) {
  if (force) {
    mapVecInt_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapVecInt_t.find(bn.Data()) != mapVecInt_t.end()) {
    mapVecInt_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<std::vector<UInt_t>>(TString bn, std::vector<UInt_t> val, bool force, bool ignore) {
  if (force) {
    mapVecUInt_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapVecUInt_t.find(bn.Data()) != mapVecUInt_t.end()) {
    mapVecUInt_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<std::vector<Bool_t>>(TString bn, std::vector<Bool_t> val, bool force, bool ignore) {
  if (force) {
    mapVecBool_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapVecBool_t.find(bn.Data()) != mapVecBool_t.end()) {
    mapVecBool_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<std::vector<Float_t>>(TString bn, std::vector<Float_t> val, bool force, bool ignore) {
  if (force) {
    mapVecFloat_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapVecFloat_t.find(bn.Data()) != mapVecFloat_t.end()) {
    mapVecFloat_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<std::vector<TString>>(TString bn, std::vector<TString> val, bool force, bool ignore) {
  if (force) {
    mapVecTString[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapVecTString.find(bn.Data()) != mapVecTString.end()) {
    mapVecTString[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<std::vector<LV>>(TString bn, std::vector<LV> val, bool force, bool ignore) {
  if (force) {
    mapVecLV[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapVecLV.find(bn.Data()) != mapVecLV.end()) {
    mapVecLV[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<std::vector<VInt>>(TString bn, std::vector<VInt> val, bool force, bool ignore) {
  if (force) {
    mapVecVInt[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapVecVInt.find(bn.Data()) != mapVecVInt.end()) {
    mapVecVInt[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::setBranch<std::vector<VFloat>>(TString bn, std::vector<VFloat> val, bool force, bool ignore) {
  if (force) {
    mapVecVFloat[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapVecVFloat.find(bn.Data()) != mapVecVFloat.end()) {
    mapVecVFloat[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::pushbackToBranch<Int_t>(TString bn, Int_t val) {
  if (mapVecInt_t.find(bn.Data()) != mapVecInt_t.end()) {
    mapVecInt_t[bn.Data()].push_back(val);
    mapIsBranchSet[bn.Data()] = true;
  } else {
    error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::pushbackToBranch<UInt_t>(TString bn, UInt_t val) {
  if (mapVecUInt_t.find(bn.Data()) != mapVecUInt_t.end()) {
    mapVecUInt_t[bn.Data()].push_back(val);
    mapIsBranchSet[bn.Data()] = true;
  } else {
    error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::pushbackToBranch<Bool_t>(TString bn, Bool_t val) {
  if (mapVecBool_t.find(bn.Data()) != mapVecBool_t.end()) {
    mapVecBool_t[bn.Data()].push_back(val);
    mapIsBranchSet[bn.Data()] = true;
  } else {
    error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::pushbackToBranch<Float_t>(TString bn, Float_t val) {
  if (mapVecFloat_t.find(bn.Data()) != mapVecFloat_t.end()) {
    mapVecFloat_t[bn.Data()].push_back(val);
    mapIsBranchSet[bn.Data()] = true;
  } else {
    error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::pushbackToBranch<TString>(TString bn, TString val) {
  if (mapVecTString.find(bn.Data()) != mapVecTString.end()) {
    mapVecTString[bn.Data()].push_back(val);
    mapIsBranchSet[bn.Data()] = true;
  } else {
    error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::pushbackToBranch<LV>(TString bn, LV val) {
  if (mapVecLV.find(bn.Data()) != mapVecLV.end()) {
    mapVecLV[bn.Data()].push_back(val);
    mapIsBranchSet[bn.Data()] = true;
  } else {
    error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::pushbackToBranch<VInt>(TString bn, VInt val) {
  if (mapVecVInt.find(bn.Data()) != mapVecVInt.end()) {
    mapVecVInt[bn.Data()].push_back(val);
    mapIsBranchSet[bn.Data()] = true;
  } else {
    error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::pushbackToBranch<VFloat>(TString bn, VFloat val) {
  if (mapVecVFloat.find(bn.Data()) != mapVecVFloat.end()) {
    mapVecVFloat[bn.Data()].push_back(val);
    mapIsBranchSet[bn.Data()] = true;
  } else {
    error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}
// functors
template <>
void TTreeX::setBranch<std::function<float()>>(TString bn, std::function<float()> val, bool force, bool ignore) {
  if (force) {
    mapFloatFunc_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
    return;
  }
  if (mapFloatFunc_t.find(bn.Data()) != mapFloatFunc_t.end()) {
    mapFloatFunc_t[bn.Data()] = val;
    mapIsBranchSet[bn.Data()] = true;
  } else {
    if (!ignore)
      error(TString::Format("branch doesn't exist bn = %s", bn.Data()));
  }
}

//_________________________________________________________________________________________________
template <>
const Int_t& TTreeX::getBranch<Int_t>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapInt_t[bn.Data()];
}
template <>
const Bool_t& TTreeX::getBranch<Bool_t>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapBool_t[bn.Data()];
}
template <>
const Float_t& TTreeX::getBranch<Float_t>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapFloat_t[bn.Data()];
}
template <>
const TString& TTreeX::getBranch<TString>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapTString[bn.Data()];
}
template <>
const LV& TTreeX::getBranch<LV>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapLV[bn.Data()];
}
template <>
const TBits& TTreeX::getBranch<TBits>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapTBits[bn.Data()];
}
template <>
const unsigned long long& TTreeX::getBranch<unsigned long long>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapULL[bn.Data()];
}
template <>
const unsigned int& TTreeX::getBranch<unsigned int>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapUI[bn.Data()];
}
template <>
const std::vector<Int_t>& TTreeX::getBranch<std::vector<Int_t>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapVecInt_t[bn.Data()];
}
template <>
const std::vector<UInt_t>& TTreeX::getBranch<std::vector<UInt_t>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapVecUInt_t[bn.Data()];
}
template <>
const std::vector<Bool_t>& TTreeX::getBranch<std::vector<Bool_t>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapVecBool_t[bn.Data()];
}
template <>
const std::vector<Float_t>& TTreeX::getBranch<std::vector<Float_t>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapVecFloat_t[bn.Data()];
}
template <>
const std::vector<TString>& TTreeX::getBranch<std::vector<TString>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapVecTString[bn.Data()];
}
template <>
const std::vector<LV>& TTreeX::getBranch<std::vector<LV>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapVecLV[bn.Data()];
}
template <>
const std::vector<VInt>& TTreeX::getBranch<std::vector<VInt>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapVecVInt[bn.Data()];
}
template <>
const std::vector<VFloat>& TTreeX::getBranch<std::vector<VFloat>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapVecVFloat[bn.Data()];
}
// functors
template <>
const std::function<float()>& TTreeX::getBranch<std::function<float()>>(TString bn, bool check) {
  if (check)
    if (!mapIsBranchSet[bn.Data()])
      error(TString::Format("branch hasn't been set yet bn = %s", bn.Data()));
  return mapFloatFunc_t[bn.Data()];
}

//_________________________________________________________________________________________________
template <>
const Int_t& TTreeX::getBranchLazy<Int_t>(TString bn) {
  return getBranch<Int_t>(bn, false);
}
template <>
const Bool_t& TTreeX::getBranchLazy<Bool_t>(TString bn) {
  return getBranch<Bool_t>(bn, false);
}
template <>
const Float_t& TTreeX::getBranchLazy<Float_t>(TString bn) {
  return getBranch<Float_t>(bn, false);
}
template <>
const TString& TTreeX::getBranchLazy<TString>(TString bn) {
  return getBranch<TString>(bn, false);
}
template <>
const LV& TTreeX::getBranchLazy<LV>(TString bn) {
  return getBranch<LV>(bn, false);
}
template <>
const TBits& TTreeX::getBranchLazy<TBits>(TString bn) {
  return getBranch<TBits>(bn, false);
}
template <>
const unsigned long long& TTreeX::getBranchLazy<unsigned long long>(TString bn) {
  return getBranch<unsigned long long>(bn, false);
}
template <>
const unsigned int& TTreeX::getBranchLazy<unsigned int>(TString bn) {
  return getBranch<unsigned int>(bn, false);
}
template <>
const std::vector<Int_t>& TTreeX::getBranchLazy<std::vector<Int_t>>(TString bn) {
  return getBranch<std::vector<Int_t>>(bn, false);
}
template <>
const std::vector<UInt_t>& TTreeX::getBranchLazy<std::vector<UInt_t>>(TString bn) {
  return getBranch<std::vector<UInt_t>>(bn, false);
}
template <>
const std::vector<Bool_t>& TTreeX::getBranchLazy<std::vector<Bool_t>>(TString bn) {
  return getBranch<std::vector<Bool_t>>(bn, false);
}
template <>
const std::vector<Float_t>& TTreeX::getBranchLazy<std::vector<Float_t>>(TString bn) {
  return getBranch<std::vector<Float_t>>(bn, false);
}
template <>
const std::vector<TString>& TTreeX::getBranchLazy<std::vector<TString>>(TString bn) {
  return getBranch<std::vector<TString>>(bn, false);
}
template <>
const std::vector<LV>& TTreeX::getBranchLazy<std::vector<LV>>(TString bn) {
  return getBranch<std::vector<LV>>(bn, false);
}
template <>
const std::vector<VInt>& TTreeX::getBranchLazy<std::vector<VInt>>(TString bn) {
  return getBranch<std::vector<VInt>>(bn, false);
}
template <>
const std::vector<VFloat>& TTreeX::getBranchLazy<std::vector<VFloat>>(TString bn) {
  return getBranch<std::vector<VFloat>>(bn, false);
}
// functors
template <>
const std::function<float()>& TTreeX::getBranchLazy<std::function<float()>>(TString bn) {
  return getBranch<std::function<float()>>(bn, false);
}

//_________________________________________________________________________________________________
template <>
bool TTreeX::isBranchSet<Int_t>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<Bool_t>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<Float_t>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<TString>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<LV>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<TBits>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<unsigned long long>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<unsigned int>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<std::vector<Int_t>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<std::vector<UInt_t>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<std::vector<Bool_t>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<std::vector<Float_t>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<std::vector<TString>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<std::vector<LV>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<std::vector<VInt>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
template <>
bool TTreeX::isBranchSet<std::vector<VFloat>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}
// functors
template <>
bool TTreeX::isBranchSet<std::function<float()>>(TString bn) {
  return mapIsBranchSet[bn.Data()];
}

//_________________________________________________________________________________________________
template <>
Int_t* TTreeX::getBranchAddress<Int_t>(TString bn) {
  return &mapInt_t[bn.Data()];
}
template <>
Bool_t* TTreeX::getBranchAddress<Bool_t>(TString bn) {
  return &mapBool_t[bn.Data()];
}
template <>
Float_t* TTreeX::getBranchAddress<Float_t>(TString bn) {
  return &mapFloat_t[bn.Data()];
}

//_________________________________________________________________________________________________
template <>
void TTreeX::createBranch<Int_t>(TString bn, bool writeToTree) {
  if (mapInt_t.find(bn.Data()) == mapInt_t.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapInt_t[bn.Data()]));
    else
      mapInt_t[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<Bool_t>(TString bn, bool writeToTree) {
  if (mapBool_t.find(bn.Data()) == mapBool_t.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapBool_t[bn.Data()]));
    else
      mapBool_t[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<Float_t>(TString bn, bool writeToTree) {
  if (mapFloat_t.find(bn.Data()) == mapFloat_t.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapFloat_t[bn.Data()]));
    else
      mapFloat_t[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<TString>(TString bn, bool writeToTree) {
  if (mapTString.find(bn.Data()) == mapTString.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapTString[bn.Data()]));
    else
      mapTString[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<LV>(TString bn, bool writeToTree) {
  if (mapLV.find(bn.Data()) == mapLV.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapLV[bn.Data()]));
    else
      mapLV[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<TBits>(TString bn, bool writeToTree) {
  if (mapTBits.find(bn.Data()) == mapTBits.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapTBits[bn.Data()]));
    else
      mapTBits[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<unsigned long long>(TString bn, bool writeToTree) {
  if (mapULL.find(bn.Data()) == mapULL.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapULL[bn.Data()]));
    else
      mapULL[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<unsigned int>(TString bn, bool writeToTree) {
  if (mapUI.find(bn.Data()) == mapUI.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapUI[bn.Data()]));
    else
      mapUI[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<std::vector<Int_t>>(TString bn, bool writeToTree) {
  if (mapVecInt_t.find(bn.Data()) == mapVecInt_t.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapVecInt_t[bn.Data()]));
    else
      mapVecInt_t[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<std::vector<UInt_t>>(TString bn, bool writeToTree) {
  if (mapVecUInt_t.find(bn.Data()) == mapVecUInt_t.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapVecUInt_t[bn.Data()]));
    else
      mapVecUInt_t[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<std::vector<Bool_t>>(TString bn, bool writeToTree) {
  if (mapVecBool_t.find(bn.Data()) == mapVecBool_t.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapVecBool_t[bn.Data()]));
    else
      mapVecBool_t[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<std::vector<Float_t>>(TString bn, bool writeToTree) {
  if (mapVecFloat_t.find(bn.Data()) == mapVecFloat_t.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapVecFloat_t[bn.Data()]));
    else
      mapVecFloat_t[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<std::vector<TString>>(TString bn, bool writeToTree) {
  if (mapVecTString.find(bn.Data()) == mapVecTString.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapVecTString[bn.Data()]));
    else
      mapVecTString[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<std::vector<LV>>(TString bn, bool writeToTree) {
  if (mapVecLV.find(bn.Data()) == mapVecLV.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapVecLV[bn.Data()]));
    else
      mapVecLV[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<std::vector<VInt>>(TString bn, bool writeToTree) {
  if (mapVecVInt.find(bn.Data()) == mapVecVInt.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapVecVInt[bn.Data()]));
    else
      mapVecVInt[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
template <>
void TTreeX::createBranch<std::vector<VFloat>>(TString bn, bool writeToTree) {
  if (mapVecVFloat.find(bn.Data()) == mapVecVFloat.end()) {
    if (writeToTree)
      ttree->Branch(bn, &(mapVecVFloat[bn.Data()]));
    else
      mapVecVFloat[bn.Data()];
  } else {
    error(TString::Format("branch already exists bn = %s", bn.Data()));
  }
}
// functors
template <>
void TTreeX::createBranch<std::function<float()>>(TString bn, bool writeToTree) {
  if (mapFloatFunc_t.find(bn.Data()) == mapFloatFunc_t.end())
    mapFloatFunc_t[bn.Data()];
  else
    error(TString::Format("branch already exists bn = %s", bn.Data()));
}

//_________________________________________________________________________________________________
template <>
bool TTreeX::hasBranch<Int_t>(TString bn) {
  if (mapInt_t.find(bn.Data()) == mapInt_t.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<Bool_t>(TString bn) {
  if (mapBool_t.find(bn.Data()) == mapBool_t.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<Float_t>(TString bn) {
  if (mapFloat_t.find(bn.Data()) == mapFloat_t.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<TString>(TString bn) {
  if (mapTString.find(bn.Data()) == mapTString.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<LV>(TString bn) {
  if (mapLV.find(bn.Data()) == mapLV.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<TBits>(TString bn) {
  if (mapTBits.find(bn.Data()) == mapTBits.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<unsigned long long>(TString bn) {
  if (mapULL.find(bn.Data()) == mapULL.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<unsigned int>(TString bn) {
  if (mapUI.find(bn.Data()) == mapUI.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<std::vector<Int_t>>(TString bn) {
  if (mapVecInt_t.find(bn.Data()) == mapVecInt_t.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<std::vector<UInt_t>>(TString bn) {
  if (mapVecUInt_t.find(bn.Data()) == mapVecUInt_t.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<std::vector<Bool_t>>(TString bn) {
  if (mapVecBool_t.find(bn.Data()) == mapVecBool_t.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<std::vector<Float_t>>(TString bn) {
  if (mapVecFloat_t.find(bn.Data()) == mapVecFloat_t.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<std::vector<TString>>(TString bn) {
  if (mapVecTString.find(bn.Data()) == mapVecTString.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<std::vector<LV>>(TString bn) {
  if (mapVecLV.find(bn.Data()) == mapVecLV.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<std::vector<VInt>>(TString bn) {
  if (mapVecVInt.find(bn.Data()) == mapVecVInt.end())
    return false;
  else
    return true;
}
template <>
bool TTreeX::hasBranch<std::vector<VFloat>>(TString bn) {
  if (mapVecVFloat.find(bn.Data()) == mapVecVFloat.end())
    return false;
  else
    return true;
}
// functors
template <>
bool TTreeX::hasBranch<std::function<float()>>(TString bn) {
  if (mapFloatFunc_t.find(bn.Data()) == mapFloatFunc_t.end())
    return false;
  else
    return true;
}

//_________________________________________________________________________________________________
template <>
void TTreeX::setBranch<std::map<TTREEXSTRING, std::vector<Int_t>>>(std::map<TTREEXSTRING, std::vector<Int_t>>& objidx) {
  for (auto& pair : objidx) {
    setBranch<Int_t>("n" + pair.first, pair.second.size());
    setBranch<std::vector<Int_t>>(pair.first, pair.second);
  }
}
template <>
void TTreeX::createBranch<std::map<TTREEXSTRING, std::vector<Int_t>>>(
    std::map<TTREEXSTRING, std::vector<Int_t>>& objidx) {
  for (auto& pair : objidx) {
    createBranch<Int_t>("n" + pair.first);
    createBranch<std::vector<Int_t>>(pair.first);
  }
}

//eof
