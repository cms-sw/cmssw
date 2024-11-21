#ifndef anautil_h
#define anautil_h

#include "cutflowutil.h"
#include "ttreex.h"
#include "printutil.h"
#include <utility>
#include <vector>
#include <map>
#include <tuple>
#include "TH1.h"
#include "TString.h"
#include <iostream>
#include <algorithm>

bool PASS();
float UNITY();

namespace RooUtil {
  class Histograms {
  public:
#ifdef USE_CUTLAMBDA
    std::map<TString, std::tuple<std::vector<float>, std::function<float()>>> th1fs_varbin;
    std::map<TString, std::tuple<unsigned int, float, float, std::function<float()>>> th1fs;
    std::map<TString,
             std::tuple<std::vector<float>, std::function<std::vector<float>()>, std::function<std::vector<float>()>>>
        th1vecfs_varbin;
    std::map<
        TString,
        std::tuple<unsigned int, float, float, std::function<std::vector<float>()>, std::function<std::vector<float>()>>>
        th1vecfs;
    std::map<
        std::pair<TString, TString>,
        std::tuple<unsigned int, float, float, unsigned int, float, float, std::function<float()>, std::function<float()>>>
        th2fs;
    std::map<std::pair<TString, TString>,
             std::tuple<unsigned int,
                        float,
                        float,
                        unsigned int,
                        float,
                        float,
                        std::function<std::vector<float>()>,
                        std::function<std::vector<float>()>,
                        std::function<std::vector<float>()>>>
        th2vecfs;
    std::map<std::pair<TString, TString>,
             std::tuple<std::vector<float>,
                        unsigned int,
                        float,
                        float,
                        std::function<std::vector<float>()>,
                        std::function<std::vector<float>()>,
                        std::function<std::vector<float>()>>>
        th2vecfs_xvarbin;
#else
    std::map<TString, std::vector<float>> th1fs_varbin;
    std::map<TString, std::tuple<unsigned int, float, float>> th1fs;
    std::map<std::pair<TString, TString>, std::tuple<unsigned int, float, float, unsigned int, float, float>> th2fs;
#endif
    Histograms();
    ~Histograms();
#ifdef USE_CUTLAMBDA
    void addHistogram(TString, unsigned int, float, float, std::function<float()>);
    void addVecHistogram(TString,
                         unsigned int,
                         float,
                         float,
                         std::function<std::vector<float>()>,
                         std::function<std::vector<float>()> wgt = nullptr);
    void addHistogram(TString, std::vector<float>, std::function<float()>);
    void addVecHistogram(TString,
                         std::vector<float>,
                         std::function<std::vector<float>()>,
                         std::function<std::vector<float>()> wgt = nullptr);
    void add2DHistogram(TString,
                        unsigned int,
                        float,
                        float,
                        TString,
                        unsigned int,
                        float,
                        float,
                        std::function<float()>,
                        std::function<float()>);
    void add2DVecHistogram(TString,
                           unsigned int,
                           float,
                           float,
                           TString,
                           unsigned int,
                           float,
                           float,
                           std::function<std::vector<float>()>,
                           std::function<std::vector<float>()>,
                           std::function<std::vector<float>()> wgt = nullptr);
    void add2DVecHistogram(TString,
                           std::vector<float>,
                           TString,
                           unsigned int,
                           float,
                           float,
                           std::function<std::vector<float>()>,
                           std::function<std::vector<float>()>,
                           std::function<std::vector<float>()> wgt = nullptr);
#else
    void addHistogram(TString, unsigned int, float, float, std::function<float()> vardef);
    void addHistogram(TString, std::vector<float>, std::function<float()> vardef);
    void add2DHistogram(TString, unsigned int, float, float, TString, unsigned int, float, float);
#endif
  };

  class Cutflow {
  public:
    CutTree cuttree;
    CutTree* last_active_cut;  // when getCut is called this is set
    std::map<TREEMAPSTRING, CutTree*> cuttreemap;
    std::map<CUTFLOWMAPSTRING, THist*> cutflow_histograms;
    std::map<CUTFLOWMAPSTRING, THist*> rawcutflow_histograms;
    std::map<std::tuple<TREEMAPSTRING, TREEMAPSTRING, TREEMAPSTRING>, THist*>
        booked_histograms;  // key is <cutname, syst, varname>
    std::map<std::tuple<TREEMAPSTRING, TREEMAPSTRING, TREEMAPSTRING, TREEMAPSTRING>, TH2F*>
        booked_2dhistograms;  // key is <cutname, syst, varname, varnamey>
    std::vector<std::tuple<TREEMAPSTRING, TREEMAPSTRING, TREEMAPSTRING>>
        booked_histograms_nominal_keys;  // key is <cutname, syst="", varname>
    std::vector<std::tuple<TREEMAPSTRING, TREEMAPSTRING, TREEMAPSTRING, TREEMAPSTRING>>
        booked_2dhistograms_nominal_keys;  // key is <cutname, syst="", varname, varnamey>
    std::vector<std::tuple<THist*, std::vector<int*>, std::vector<float*>, std::function<float()>>>
        cutflow_histograms_v2;
    std::vector<std::tuple<THist*, std::vector<int*>>> rawcutflow_histograms_v2;
    std::vector<TString> cutflow_nofill_cut_list;
    TFile* ofile;
    TTree* t;
    TTreeX* tx;
    std::vector<TString> cutsysts;
    std::vector<TString> systs;
    std::map<TString, std::function<float()>> systs_funcs;
    std::map<TString, std::vector<TString>> cutlists;
    std::map<TString, std::vector<CutTree*>> cuttreelists;
    bool iseventlistbooked;
    int seterrorcount;
    bool doskipsysthist;
    bool dosavettreex;
    bool cutflow_booked;
    Cutflow();
    Cutflow(TFile* o);
    ~Cutflow();
    void setTFile(TFile* o) { ofile = o; }
    void addToCutTreeMap(TString n);
    void setLastActiveCut(TString n);
#ifdef USE_CUTLAMBDA
    void addCut(TString n, std::function<bool()> pass, std::function<float()> weight);
    void addCutToLastActiveCut(TString n, std::function<bool()> pass, std::function<float()> weight);
#else
    void addCut(TString n);
    void addCutToLastActiveCut(TString n);
#endif
    void copyAndEditCuts(TString, std::map<TString, TString>);
    void printCuts();
    CutTree& getCut(TString n);
    void removeCut(TString n);
    // void filterCuts(TString n);
    void filterCuts(std::vector<TString> ns);
    void setCutLists(std::vector<TString> regions);
    void addCutToSkipCutflowList(TString n);
    void bookCutflowTree();
    void bookCutflowHistograms();
    void bookCutflowHistograms_v1();
    void bookCutflowHistograms_v2();
    void bookCutflowsForRegions(std::vector<TString> regions);
    void bookCutflows();
    void setSkipSystematicHistograms(bool = true);
    void setSaveTTreeX(bool = true);
    void saveOutput();
    void saveCutflows();
    void saveHistograms();
    void saveTTreeX();
#ifdef USE_CUTLAMBDA
    void setCut(TString cutname, std::function<bool()> pass, std::function<float()> weight);
    void setCutSyst(TString cutname, TString syst, std::function<bool()> pass, std::function<float()> weight);
#else
    void setCut(TString cutname, bool pass, float weight);
    void setCutSyst(TString cutname, TString syst, bool pass, float weight);
#endif
    void addCutSyst(TString syst,
                    std::vector<TString> pattern,
                    std::vector<TString> vetopattern = std::vector<TString>());
#ifdef USE_CUTLAMBDA
    void addWgtSyst(TString syst, std::function<float()>);
#else
    void addWgtSyst(TString syst);
#endif
    void setWgtSyst(TString syst, float weight);  // TODO make TTreeX using lambda...?
    void createWgtSystBranches();
    void setVariable(TString varname, float);
    void setEventID(int, int, unsigned long long);
    void bookEventLists();
    void fill();
    void fillCutflows(TString syst = "", bool iswgtsyst = true);
    void fillCutflow(std::vector<TString>& cutlist, THist* h, THist* hraw, float wgtsyst = 1);
    void fillCutflows_v1(TString syst = "", bool iswgtsyst = true);
    void fillCutflow_v2(std::vector<CutTree*>& cutlist, THist* h, THist* hraw, float wgtsyst = 1);
    void fillCutflows_v2(TString syst = "", bool iswgtsyst = true);
    void fillHistograms(TString syst = "", bool iswgtsyst = true);
#ifdef USE_CUTLAMBDA
    void bookHistogram(TString,
                       std::pair<TString, std::tuple<unsigned, float, float, std::function<float()>>>,
                       TString = "");
    void bookVecHistogram(
        TString,
        std::pair<
            TString,
            std::tuple<unsigned, float, float, std::function<std::vector<float>()>, std::function<std::vector<float>()>>>,
        TString = "");
    void bookHistogram(TString,
                       std::pair<TString, std::tuple<std::vector<float>, std::function<float()>>>,
                       TString = "");
    void bookVecHistogram(
        TString,
        std::pair<
            TString,
            std::tuple<std::vector<float>, std::function<std::vector<float>()>, std::function<std::vector<float>()>>>,
        TString = "");
    void book2DHistogram(
        TString,
        std::pair<
            std::pair<TString, TString>,
            std::tuple<unsigned, float, float, unsigned, float, float, std::function<float()>, std::function<float()>>>,
        TString = "");
    void book2DVecHistogram(TString,
                            std::pair<std::pair<TString, TString>,
                                      std::tuple<unsigned,
                                                 float,
                                                 float,
                                                 unsigned,
                                                 float,
                                                 float,
                                                 std::function<std::vector<float>()>,
                                                 std::function<std::vector<float>()>,
                                                 std::function<std::vector<float>()>>>,
                            TString = "");
    void book2DVecHistogram(TString,
                            std::pair<std::pair<TString, TString>,
                                      std::tuple<std::vector<float>,
                                                 unsigned,
                                                 float,
                                                 float,
                                                 std::function<std::vector<float>()>,
                                                 std::function<std::vector<float>()>,
                                                 std::function<std::vector<float>()>>>,
                            TString = "");
#else
    void bookHistogram(TString, std::pair<TString, std::tuple<unsigned, float, float>>, TString = "");
    void bookHistogram(TString, std::pair<TString, std::vector<float>>, TString = "");
    void book2DHistogram(
        TString,
        std::pair<std::pair<TString, TString>, std::tuple<unsigned, float, float, unsigned, float, float>>,
        TString = "");
#endif
    void bookHistograms(Histograms& histograms);
    void bookHistograms(Histograms& histograms, std::vector<TString> cutlist);
    void bookHistogramsForCut(Histograms& histograms, TString);
    void bookHistogramsForCutAndBelow(Histograms& histograms, TString);
    void bookHistogramsForCutAndAbove(Histograms& histograms, TString);
    void bookHistogramsForEndCuts(Histograms& histograms);
    void printSetFunctionError(TString msg);
    void setHistsAxesExtendable();
  };
}  // namespace RooUtil

#endif
