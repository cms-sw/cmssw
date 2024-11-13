#include "cutflowutil.h"

//_______________________________________________________________________________________________________
void RooUtil::CutflowUtil::createCutflowBranches(std::map<TString, std::vector<TString>>& cutlists,
                                                 RooUtil::TTreeX& tx) {
  for (auto& cutlist : cutlists) {
    for (auto& cutname : cutlist.second) {
      //            std::cout <<  " cutname: " << cutname <<  std::endl;
      if (!tx.hasBranch<bool>(cutname))
        tx.createBranch<bool>(cutname);
      if (!tx.hasBranch<float>(cutname + "_weight"))
        tx.createBranch<float>(cutname + "_weight");
    }
  }
}

//_______________________________________________________________________________________________________
void RooUtil::CutflowUtil::createCutflowBranches(CutNameListMap& cutlists, RooUtil::TTreeX& tx) {
  std::map<TString, std::vector<TString>> obj = cutlists.getStdVersion();
  createCutflowBranches(obj, tx);
}

//_______________________________________________________________________________________________________
std::tuple<std::vector<bool>, std::vector<float>> RooUtil::CutflowUtil::getCutflow(std::vector<TString> cutlist,
                                                                                   RooUtil::TTreeX& tx) {
  std::vector<float> rtnwgt;
  std::vector<bool> rtn;
  float wgtall = 1;
  bool passall = true;
  for (auto& cutname : cutlist) {
    bool pass = tx.getBranch<bool>(cutname);
    float wgt = tx.getBranch<float>(cutname + "_weight");
    wgtall *= wgt;
    passall &= pass;
    rtnwgt.push_back(wgtall);
    rtn.push_back(passall);
  }
  return std::make_tuple(rtn, rtnwgt);
}

//_______________________________________________________________________________________________________
std::pair<bool, float> RooUtil::CutflowUtil::passCuts(std::vector<TString> cutlist, RooUtil::TTreeX& tx) {
  std::vector<bool> cutflow;
  std::vector<float> cutflow_weight;
  std::tie(cutflow, cutflow_weight) = getCutflow(cutlist, tx);
  // Just need to check the last one
  bool passall = cutflow.back();
  float wgtall = cutflow_weight.back();
  return std::make_pair(passall, wgtall);
}

//_______________________________________________________________________________________________________
void RooUtil::CutflowUtil::fillCutflow(std::vector<TString> cutlist, RooUtil::TTreeX& tx, THist* h) {
  std::vector<bool> cutflow;
  std::vector<float> cutflow_weight;
  std::tie(cutflow, cutflow_weight) = getCutflow(cutlist, tx);
  for (unsigned int i = 0; i < cutflow.size(); ++i)
    if (cutflow[i] > 0)
      h->Fill(i, cutflow[i] * cutflow_weight[i]);
    else
      return;
}

//_______________________________________________________________________________________________________
void RooUtil::CutflowUtil::fillRawCutflow(std::vector<TString> cutlist, RooUtil::TTreeX& tx, THist* h) {
  std::vector<bool> cutflow;
  std::vector<float> cutflow_weight;
  std::tie(cutflow, cutflow_weight) = getCutflow(cutlist, tx);
  for (unsigned int i = 0; i < cutflow.size(); ++i)
    if (cutflow[i] > 0)
      h->Fill(i);
    else
      return;
}

//_______________________________________________________________________________________________________
std::tuple<std::map<CUTFLOWMAPSTRING, THist*>, std::map<CUTFLOWMAPSTRING, THist*>>
RooUtil::CutflowUtil::createCutflowHistograms(RooUtil::CutflowUtil::CutNameListMap& cutlists, TString syst) {
  std::map<TString, std::vector<TString>> obj = cutlists.getStdVersion();
  return createCutflowHistograms(obj, syst);
}

//_______________________________________________________________________________________________________
std::tuple<std::map<CUTFLOWMAPSTRING, THist*>, std::map<CUTFLOWMAPSTRING, THist*>>
RooUtil::CutflowUtil::createCutflowHistograms(std::map<TString, std::vector<TString>>& cutlists, TString syst) {
  std::map<CUTFLOWMAPSTRING, THist*> cutflows;
  std::map<CUTFLOWMAPSTRING, THist*> rawcutflows;
  for (auto& cutlist : cutlists) {
    cutflows[(cutlist.first + syst).Data()] =
        new THist(cutlist.first + syst + "_cutflow", "", cutlist.second.size(), 0, cutlist.second.size());
    rawcutflows[(cutlist.first + syst).Data()] =
        new THist(cutlist.first + syst + "_rawcutflow", "", cutlist.second.size(), 0, cutlist.second.size());
    cutflows[(cutlist.first + syst).Data()]->Sumw2();
    rawcutflows[(cutlist.first + syst).Data()]->Sumw2();
    cutflows[(cutlist.first + syst).Data()]->SetDirectory(0);
    rawcutflows[(cutlist.first + syst).Data()]->SetDirectory(0);
    for (unsigned int i = 0; i < cutlist.second.size(); ++i) {
      cutflows[(cutlist.first + syst).Data()]->GetXaxis()->SetBinLabel(i + 1, cutlist.second[i]);
      rawcutflows[(cutlist.first + syst).Data()]->GetXaxis()->SetBinLabel(i + 1, cutlist.second[i]);
    }
  }
  return std::make_tuple(cutflows, rawcutflows);
}

////_______________________________________________________________________________________________________
//void RooUtil::CutflowUtil::fillCutflowHistograms(RooUtil::CutflowUtil::CutNameListMap& cutlists, RooUtil::TTreeX& tx, std::map<TString, THist*>& cutflows, std::map<TString, THist*>& rawcutflows)
//{
//    std::map<TString, std::vector<TString>> obj = cutlists.getStdVersion();
//    fillCutflowHistograms(obj, tx, cutflows, rawcutflows);
//}
//
////_______________________________________________________________________________________________________
//void RooUtil::CutflowUtil::fillCutflowHistograms(std::map<TString, std::vector<TString>>& cutlists, RooUtil::TTreeX& tx, std::map<TString, THist*>& cutflows, std::map<TString, THist*>& rawcutflows)
//{
//    for (auto& cutlist : cutlists)
//    {
//        RooUtil::CutflowUtil::fillCutflow(cutlist.second, tx, cutflows[cutlist.first]);
//        RooUtil::CutflowUtil::fillRawCutflow(cutlist.second, tx, rawcutflows[cutlist.first]);
//    }
//}

//_______________________________________________________________________________________________________
void RooUtil::CutflowUtil::saveCutflowHistograms(std::map<CUTFLOWMAPSTRING, THist*>& cutflows,
                                                 std::map<CUTFLOWMAPSTRING, THist*>& rawcutflows) {
  for (auto& cutflow : cutflows)
    cutflow.second->Write();
  for (auto& rawcutflow : rawcutflows)
    rawcutflow.second->Write();
}
