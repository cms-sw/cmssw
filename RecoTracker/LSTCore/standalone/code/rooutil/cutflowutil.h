#ifndef cutflowutil_h
#define cutflowutil_h

#include "ttreex.h"
#include "printutil.h"
#include <tuple>
#include <vector>
#include <map>
#include "TH1.h"
#include "TString.h"
#include <iostream>
#include <algorithm>
#include <sys/ioctl.h>
#include <functional>

//#define USE_TTREEX
#define USE_CUTLAMBDA
#define TREEMAPSTRING std::string
#define CUTFLOWMAPSTRING TString
#define DATA c_str
#define THist TH1D

namespace RooUtil {
  namespace CutflowUtil {

    class CutNameList {
    public:
      std::vector<TString> cutlist;
      CutNameList() {}
      CutNameList(const CutNameList& cutnamelist) { cutlist = cutnamelist.cutlist; }
      void clear() { cutlist.clear(); }
      void addCutName(TString cutname) { cutlist.push_back(cutname); }
      void print() {
        for (auto& str : cutlist)
          std::cout << str << std::endl;
      }
    };

    class CutNameListMap {
    public:
      std::map<TString, CutNameList> cutlists;
      std::vector<TString> cutlist;
      CutNameList& operator[](TString name) { return cutlists[name]; }
      void clear() { cutlists.clear(); }
      void print() {
        for (auto& cl : cutlists) {
          std::cout << "CutNameList - " << cl.first << std::endl;
          cl.second.print();
        }
      }
      std::map<TString, std::vector<TString>> getStdVersion() {
        std::map<TString, std::vector<TString>> obj_cutlists;
        for (auto& cl : cutlists)
          obj_cutlists[cl.first] = cl.second.cutlist;
        return obj_cutlists;
      }
    };

    void createCutflowBranches(CutNameListMap& cutlists, RooUtil::TTreeX& tx);
    void createCutflowBranches(std::map<TString, std::vector<TString>>& cutlists, RooUtil::TTreeX& tx);
    std::tuple<std::vector<bool>, std::vector<float>> getCutflow(std::vector<TString> cutlist, RooUtil::TTreeX& tx);
    std::pair<bool, float> passCuts(std::vector<TString> cutlist, RooUtil::TTreeX& tx);
    void fillCutflow(std::vector<TString> cutlist, RooUtil::TTreeX& tx, THist* h);
    void fillRawCutflow(std::vector<TString> cutlist, RooUtil::TTreeX& tx, THist* h);
    std::tuple<std::map<CUTFLOWMAPSTRING, THist*>, std::map<CUTFLOWMAPSTRING, THist*>> createCutflowHistograms(
        CutNameListMap& cutlists, TString syst = "");
    std::tuple<std::map<CUTFLOWMAPSTRING, THist*>, std::map<CUTFLOWMAPSTRING, THist*>> createCutflowHistograms(
        std::map<TString, std::vector<TString>>& cutlists, TString syst = "");
    void saveCutflowHistograms(std::map<CUTFLOWMAPSTRING, THist*>& cutflows,
                               std::map<CUTFLOWMAPSTRING, THist*>& rawcutflows);
    //        void fillCutflowHistograms(CutNameListMap& cutlists, RooUtil::TTreeX& tx, std::map<TString, THist*>& cutflows, std::map<TString, THist*>& rawcutflows);
    //        void fillCutflowHistograms(std::map<TString, std::vector<TString>>& cutlists, RooUtil::TTreeX& tx, std::map<TString, THist*>& cutflows, std::map<TString, THist*>& rawcutflows);

  }  // namespace CutflowUtil

  class CutTree {
  public:
    TString name;
    CutTree* parent;
    std::vector<CutTree*> parents;
    std::vector<CutTree*> children;
    std::vector<TString> systcutnames;
    std::vector<CutTree*> systcuts;
    std::map<TString, CutTree*> systs;
    int pass;
    float weight;
    std::vector<int> systpasses;
    std::vector<float> systweights;
    bool pass_this_cut;
    float weight_this_cut;
    std::function<bool()> pass_this_cut_func;
    std::function<float()> weight_this_cut_func;
//            std::vector<TString> hists1d;
//            std::vector<std::tuple<TString, TString>> hists2d;
#ifdef USE_CUTLAMBDA
    std::map<TString, std::vector<std::tuple<THist*, std::function<float()>>>> hists1d;
    std::map<TString,
             std::vector<std::tuple<THist*, std::function<std::vector<float>()>, std::function<std::vector<float>()>>>>
        hists1dvec;
    std::map<TString, std::vector<std::tuple<TH2F*, std::function<float()>, std::function<float()>>>> hists2d;
    std::map<TString,
             std::vector<std::tuple<TH2F*,
                                    std::function<std::vector<float>()>,
                                    std::function<std::vector<float>()>,
                                    std::function<std::vector<float>()>>>>
        hists2dvec;
#else
    std::map<TString, std::vector<std::tuple<THist*, TString>>> hists1d;
    std::map<TString, std::vector<std::tuple<TH2F*, TString, TString>>> hists2d;
#endif
    std::vector<std::tuple<int, int, unsigned long long>> eventlist;
    CutTree(TString n) : name(n), parent(0), pass(false), weight(0) {}
    ~CutTree() {
      for (auto& child : children) {
        if (child)
          delete child;
      }
    }
    void printCuts(int indent = 0, std::vector<int> multichild = std::vector<int>()) {
      struct winsize w;
      ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
      int colsize = std::min(w.ws_col - 100, 600);
      if (indent == 0) {
        TString header = "Cut name";
        int extra = colsize - header.Length();
        for (int i = 0; i < extra; ++i)
          header += " ";
        header += "|pass|weight|systs";
        print(header);
        TString delimiter = "";
        for (int i = 0; i < w.ws_col - 10; ++i)
          delimiter += "=";
        print(delimiter);
      }
      TString msg = "";
      for (int i = 0; i < indent; ++i) {
        if (std::find(multichild.begin(), multichild.end(), i + 1) != multichild.end()) {
          if (indent == i + 1)
            msg += " +";
          else
            msg += " |";
        } else
          msg += "  ";
      }
      msg += name;
      int extrapad = colsize - msg.Length() > 0 ? colsize - msg.Length() : 0;
      for (int i = 0; i < extrapad; ++i)
        msg += " ";
      //msg += TString::Format("| %d | %.5f|", pass, weight);
      msg += TString::Format("| %d | %f|", pass, weight);
      for (auto& key : systs) {
        msg += key.first + " ";
      }
      print(msg);
      if (children.size() > 1)
        multichild.push_back(indent + 1);
      for (auto& child : children) {
        (*child).printCuts(indent + 1, multichild);
      }
    }
    void printEventList() {
      print(TString::Format("Print event list for the cut = %s", name.Data()));
      for (auto& eventid : eventlist) {
        int run = std::get<0>(eventid);
        int lumi = std::get<1>(eventid);
        unsigned long long evt = std::get<2>(eventid);
        TString msg = TString::Format("%d:%d:%llu", run, lumi, evt);
        std::cout << msg << std::endl;
      }
    }
    void writeEventList(TString ofilename) {
      std::ofstream outFile(ofilename);
      for (auto& tuple : eventlist) {
        int run = std::get<0>(tuple);
        int lumi = std::get<1>(tuple);
        unsigned long long evt = std::get<2>(tuple);
        outFile << run << ":" << lumi << ":" << evt << std::endl;
      }
      outFile.close();
    }
    void addCut(TString n) {
      CutTree* obj = new CutTree(n);
      obj->parent = this;
      obj->parents.push_back(this);
      children.push_back(obj);
    }
    void addSyst(TString syst) {
      // If already added ignore
      if (systs.find(syst) != systs.end())
        return;
      // Syst CutTree object knows the parents, and children, however, the children does not know the syst-counter-part parent, nor the parent knows the syste-counter-part children.
      CutTree* obj = new CutTree(this->name + syst);
      systs[syst] = obj;
      systcutnames.push_back(syst);
      systcuts.push_back(obj);
      systpasses.push_back(1);
      systweights.push_back(1);
      obj->children = this->children;
      obj->parents = this->parents;
      obj->parent = this->parent;
    }
#ifdef USE_CUTLAMBDA
    void addHist1D(THist* h, std::function<float()> var, TString syst) {
      if (syst.IsNull())
        hists1d["Nominal"].push_back(std::make_tuple(h, var));
      else
        hists1d[syst].push_back(std::make_tuple(h, var));
    }
    void addHist1DVec(THist* h,
                      std::function<std::vector<float>()> var,
                      std::function<std::vector<float>()> wgt,
                      TString syst) {
      if (syst.IsNull())
        hists1dvec["Nominal"].push_back(std::make_tuple(h, var, wgt));
      else
        hists1dvec[syst].push_back(std::make_tuple(h, var, wgt));
    }
    void addHist2D(TH2F* h, std::function<float()> varx, std::function<float()> vary, TString syst) {
      if (syst.IsNull())
        hists2d["Nominal"].push_back(std::make_tuple(h, varx, vary));
      else
        hists2d[syst].push_back(std::make_tuple(h, varx, vary));
    }
    void addHist2DVec(TH2F* h,
                      std::function<std::vector<float>()> varx,
                      std::function<std::vector<float>()> vary,
                      std::function<std::vector<float>()> elemwgt,
                      TString syst) {
      if (syst.IsNull())
        hists2dvec["Nominal"].push_back(std::make_tuple(h, varx, vary, elemwgt));
      else
        hists2dvec[syst].push_back(std::make_tuple(h, varx, vary, elemwgt));
    }
#else
    void addHist1D(THist* h, TString var, TString syst) {
      if (syst.IsNull())
        hists1d["Nominal"].push_back(std::make_tuple(h, var));
      else
        hists1d[syst].push_back(std::make_tuple(h, var));
    }
    void addHist2D(TH2F* h, TString varx, TString vary, TString syst) {
      if (syst.IsNull())
        hists2d["Nominal"].push_back(std::make_tuple(h, varx, vary));
      else
        hists2d[syst].push_back(std::make_tuple(h, varx, vary));
    }
#endif
    CutTree* getCutPointer(TString n) {
      // If the name match then return itself
      if (name.EqualTo(n)) {
        return this;
      } else {
        // Otherwise, loop over the children an if a children has the correct one return the found CutTree
        for (auto& child : children) {
          CutTree* c = child->getCutPointer(n);
          if (c)
            return c;
        }
        return 0;
      }
    }
    // Wrapper to return the object instead of pointer
    CutTree& getCut(TString n) {
      CutTree* c = getCutPointer(n);
      if (c) {
        return *c;
      } else {
        RooUtil::error(TString::Format("Asked for %s cut, but did not find the cut", n.Data()));
        return *this;
      }
    }
    std::vector<TString> getCutList(TString n, std::vector<TString> cut_list = std::vector<TString>()) {
      // Main idea: start from the end node provided by the first argument "n", and work your way up to the root node.
      //
      // The algorithm will first determine whether I am starting from a specific cut requested by the user or within in recursion.
      // If the cut_list.size() == 0, the function is called by the user (since no list is aggregated so far)
      // In that case, first find the pointer to the object we want and set it to "c"
      // If cut_list.size() is non-zero then take this as the cut that I am starting and I go up the chain to aggregate all the cuts prior to the requested cut
      CutTree* c = 0;
      if (cut_list.size() == 0) {
        c = &getCut(n);
        cut_list.push_back(c->name);
      } else {
        c = this;
        cut_list.push_back(n);
      }
      if (c->parent) {
        return (c->parent)->getCutList((c->parent)->name, cut_list);
      } else {
        std::reverse(cut_list.begin(), cut_list.end());
        return cut_list;
      }
    }
    std::vector<TString> getEndCuts(std::vector<TString> endcuts = std::vector<TString>()) {
      if (children.size() == 0) {
        endcuts.push_back(name);
        return endcuts;
      }
      for (auto& child : children)
        endcuts = child->getEndCuts(endcuts);
      return endcuts;
    }
    std::vector<TString> getCutListBelow(TString n, std::vector<TString> cut_list = std::vector<TString>()) {
      // Main idea: start from the node provided by the first argument "n", and work your way down to the ends.
      CutTree* c = 0;
      if (cut_list.size() == 0) {
        c = &getCut(n);
        cut_list.push_back(c->name);
      } else {
        c = this;
        cut_list.push_back(n);
      }
      if (children.size() > 0) {
        for (auto& child : c->children) {
          cut_list = child->getCutListBelow(child->name, cut_list);
        }
        return cut_list;
      } else {
        return cut_list;
      }
    }
    void clear() {
      pass = false;
      weight = 0;
      for (auto& child : children)
        child->clear();
    }
    void addSyst(TString syst,
                 std::vector<TString> patterns,
                 std::vector<TString> vetopatterns = std::vector<TString>()) {
      for (auto& pattern : patterns)
        if (name.Contains(pattern)) {
          bool veto = false;
          for (auto& vetopattern : vetopatterns) {
            if (name.Contains(vetopattern))
              veto = true;
          }
          if (not veto)
            addSyst(syst);
        }
      for (auto& child : children)
        child->addSyst(syst, patterns, vetopatterns);
    }
    void clear_passbits() {
      pass = 0;
      weight = 0;
      for (auto& child : children)
        child->clear_passbits();
    }
    void evaluate(RooUtil::TTreeX& tx,
                  TString cutsystname = "",
                  bool doeventlist = false,
                  bool aggregated_pass = true,
                  float aggregated_weight = 1) {
#ifdef USE_TTREEX
      evaluate_use_ttreex(tx, cutsystname, doeventlist, aggregated_pass, aggregated_weight);
#else
#ifdef USE_CUTLAMBDA
      evaluate_use_lambda(tx, cutsystname, doeventlist, aggregated_pass, aggregated_weight);
#else
      evaluate_use_internal_variable(tx, cutsystname, doeventlist, aggregated_pass, aggregated_weight);
#endif
#endif
    }
    void evaluate_use_lambda(RooUtil::TTreeX& tx,
                             TString cutsystname = "",
                             bool doeventlist = false,
                             bool aggregated_pass = true,
                             float aggregated_weight = 1) {
      if (!parent) {
        clear_passbits();
        pass = 1;
        weight = 1;
      } else {
        if (cutsystname.IsNull()) {
          if (pass_this_cut_func) {
            pass = pass_this_cut_func() && aggregated_pass;
            weight = weight_this_cut_func() * aggregated_weight;
            if (!pass)
              return;
          } else {
            TString msg = "cowardly passing the event because cut and weight func not set! cut name = " + name;
            warning(msg);
            pass = aggregated_pass;
            weight = aggregated_weight;
          }
        } else {
          if (systs.find(cutsystname) == systs.end()) {
            if (pass_this_cut_func) {
              pass = pass_this_cut_func() && aggregated_pass;
              weight = weight_this_cut_func() * aggregated_weight;
              if (!pass)
                return;
            } else {
              TString msg = "cowardly passing the event because cut and weight func not set! cut name = " + name;
              warning(msg);
              pass = aggregated_pass;
              weight = aggregated_weight;
            }
          } else {
            if (systs[cutsystname]->pass_this_cut_func) {
              pass = systs[cutsystname]->pass_this_cut_func() && aggregated_pass;
              weight = systs[cutsystname]->weight_this_cut_func() * aggregated_weight;
              if (!pass)
                return;
            } else {
              TString msg = "cowardly passing the event because cut and weight func not set! cut name = " + name +
                            " syst name = " + cutsystname;
              warning(msg);
              pass = aggregated_pass;
              weight = aggregated_weight;
            }
          }
        }
      }
      if (doeventlist and pass and cutsystname.IsNull()) {
        if (tx.hasBranch<int>("run") && tx.hasBranch<int>("lumi") && tx.hasBranch<unsigned long long>("evt")) {
          eventlist.push_back(std::make_tuple(
              tx.getBranch<int>("run"), tx.getBranch<int>("lumi"), tx.getBranch<unsigned long long>("evt")));
        }
      }
      for (auto& child : children)
        child->evaluate_use_lambda(tx, cutsystname, doeventlist, pass, weight);
    }
    void evaluate_use_internal_variable(RooUtil::TTreeX& tx,
                                        TString cutsystname = "",
                                        bool doeventlist = false,
                                        bool aggregated_pass = true,
                                        float aggregated_weight = 1) {
      if (!parent) {
        clear_passbits();
        pass = 1;
        weight = 1;
      } else {
        if (cutsystname.IsNull()) {
          pass = pass_this_cut && aggregated_pass;
          weight = weight_this_cut * aggregated_weight;
          if (!pass)
            return;
        } else {
          if (systs.find(cutsystname) == systs.end()) {
            pass = pass_this_cut && aggregated_pass;
            weight = weight_this_cut * aggregated_weight;
            if (!pass)
              return;
          } else {
            pass = systs[cutsystname]->pass_this_cut && aggregated_pass;
            weight = systs[cutsystname]->weight_this_cut * aggregated_weight;
            if (!pass)
              return;
          }
        }
      }
      if (doeventlist and pass and cutsystname.IsNull()) {
        if (tx.hasBranch<int>("run") && tx.hasBranch<int>("lumi") && tx.hasBranch<unsigned long long>("evt")) {
          eventlist.push_back(std::make_tuple(
              tx.getBranch<int>("run"), tx.getBranch<int>("lumi"), tx.getBranch<unsigned long long>("evt")));
        }
      }
      for (auto& child : children)
        child->evaluate_use_internal_variable(tx, cutsystname, doeventlist, pass, weight);
    }
    void evaluate_use_ttreex(RooUtil::TTreeX& tx,
                             TString cutsystname = "",
                             bool doeventlist = false,
                             bool aggregated_pass = true,
                             float aggregated_weight = 1) {
      if (!parent) {
        pass = tx.getBranch<bool>(name);
        weight = tx.getBranch<float>(name + "_weight");
      } else {
        if (cutsystname.IsNull()) {
          if (!tx.hasBranch<bool>(name))
            return;
          pass = tx.getBranch<bool>(name) && aggregated_pass;
          weight = tx.getBranch<float>(name + "_weight") * aggregated_weight;
        } else {
          if (systs.find(cutsystname) == systs.end()) {
            if (!tx.hasBranch<bool>(name))
              return;
            pass = tx.getBranch<bool>(name) && aggregated_pass;
            weight = tx.getBranch<float>(name + "_weight") * aggregated_weight;
          } else {
            if (!tx.hasBranch<bool>(name + cutsystname))
              return;
            pass = tx.getBranch<bool>(name + cutsystname) && aggregated_pass;
            weight = tx.getBranch<float>(name + cutsystname + "_weight") * aggregated_weight;
          }
        }
      }
      if (doeventlist and pass and cutsystname.IsNull()) {
        if (tx.hasBranch<int>("run") && tx.hasBranch<int>("lumi") && tx.hasBranch<unsigned long long>("evt")) {
          eventlist.push_back(std::make_tuple(
              tx.getBranch<int>("run"), tx.getBranch<int>("lumi"), tx.getBranch<unsigned long long>("evt")));
        }
      }
      for (auto& child : children)
        child->evaluate_use_ttreex(tx, cutsystname, doeventlist, pass, weight);
    }
    void sortEventList() {
      std::sort(
          eventlist.begin(),
          eventlist.end(),
          [](const std::tuple<int, int, unsigned long long>& a, const std::tuple<int, int, unsigned long long>& b) {
            if (std::get<0>(a) != std::get<0>(b))
              return std::get<0>(a) < std::get<0>(b);
            else if (std::get<1>(a) != std::get<1>(b))
              return std::get<1>(a) < std::get<1>(b);
            else if (std::get<2>(a) != std::get<2>(b))
              return std::get<2>(a) < std::get<2>(b);
            else
              return true;
          });
    }
    void clearEventList() { eventlist.clear(); }
    void addEventList(int run, int lumi, unsigned long long evt) {
      eventlist.push_back(std::make_tuple(run, lumi, evt));
    }
#ifdef USE_CUTLAMBDA
    void fillHistograms(TString syst, float extrawgt) {
      // If the cut didn't pass then stop
      if (!pass)
        return;

      if (hists1d.size() != 0 or hists2d.size() != 0 or hists2dvec.size() != 0 or hists1dvec.size() != 0) {
        TString systkey = syst.IsNull() ? "Nominal" : syst;
        for (auto& tuple : hists1d[systkey]) {
          THist* h = std::get<0>(tuple);
          std::function<float()> vardef = std::get<1>(tuple);
          h->Fill(vardef(), weight * extrawgt);
        }
        for (auto& tuple : hists2d[systkey]) {
          TH2F* h = std::get<0>(tuple);
          std::function<float()> varxdef = std::get<1>(tuple);
          std::function<float()> varydef = std::get<2>(tuple);
          h->Fill(varxdef(), varydef(), weight * extrawgt);
        }
        for (auto& tuple : hists1dvec[systkey]) {
          THist* h = std::get<0>(tuple);
          std::function<std::vector<float>()> vardef = std::get<1>(tuple);
          std::function<std::vector<float>()> wgtdef = std::get<2>(tuple);
          std::vector<float> varx = vardef();
          std::vector<float> elemwgts;
          if (wgtdef)
            elemwgts = wgtdef();
          for (unsigned int i = 0; i < varx.size(); ++i) {
            if (wgtdef)
              h->Fill(varx[i], weight * extrawgt * elemwgts[i]);
            else
              h->Fill(varx[i], weight * extrawgt);
          }
        }
        for (auto& tuple : hists2dvec[systkey]) {
          TH2F* h = std::get<0>(tuple);
          std::function<std::vector<float>()> varxdef = std::get<1>(tuple);
          std::function<std::vector<float>()> varydef = std::get<2>(tuple);
          std::function<std::vector<float>()> wgtdef = std::get<3>(tuple);
          std::vector<float> varx = varxdef();
          std::vector<float> vary = varydef();
          if (varx.size() != vary.size()) {
            TString msg =
                "the vector input to be looped over do not have same length for x and y! check the variable definition "
                "for histogram ";
            msg += h->GetName();
            warning(msg);
          }
          std::vector<float> elemwgts;
          if (wgtdef)
            elemwgts = wgtdef();
          for (unsigned int i = 0; i < varx.size(); ++i) {
            if (wgtdef)
              h->Fill(varx[i], vary[i], weight * extrawgt * elemwgts[i]);
            else
              h->Fill(varx[i], vary[i], weight * extrawgt);
          }
        }
      }
      for (auto& child : children)
        child->fillHistograms(syst, extrawgt);
    }
#else
    void fillHistograms(RooUtil::TTreeX& tx, TString syst, float extrawgt) {
      // If the cut didn't pass then stop
      if (!pass)
        return;

      if (hists1d.size() != 0 or hists2d.size() != 0) {
        TString systkey = syst.IsNull() ? "Nominal" : syst;
        for (auto& tuple : hists1d[systkey]) {
          THist* h = std::get<0>(tuple);
          TString varname = std::get<1>(tuple);
          h->Fill(tx.getBranch<float>(varname), weight * extrawgt);
        }
        for (auto& tuple : hists2d[systkey]) {
          TH2F* h = std::get<0>(tuple);
          TString varname = std::get<1>(tuple);
          TString varnamey = std::get<2>(tuple);
          h->Fill(tx.getBranch<float>(varname), tx.getBranch<float>(varnamey), weight * extrawgt);
        }
      }
      for (auto& child : children)
        child->fillHistograms(tx, syst, extrawgt);

      //                if (!parent)
      //                {
      //                    pass = pass_this_cut;
      //                    weight = weight_this_cut;
      //                }
      //                else
      //                {
      //                    if (cutsystname.IsNull())
      //                    {
      //                        pass = pass_this_cut && aggregated_pass;
      //                        weight = weight_this_cut * aggregated_weight;
      //                    }
      //                    else
      //                    {
      //                        if (systs.find(cutsystname) == systs.end())
      //                        {
      //                            pass = pass_this_cut && aggregated_pass;
      //                            weight = weight_this_cut * aggregated_weight;
      //                        }
      //                        else
      //                        {
      //                            pass = systs[cutsystname]->pass_this_cut && aggregated_pass;
      //                            weight = systs[cutsystname]->weight_this_cut * aggregated_weight;
      //                        }
      //                    }
      //                }
    }
#endif
  };
}  // namespace RooUtil

#endif
