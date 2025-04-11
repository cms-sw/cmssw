#pragma once
#include "histManagerBase.h"
#include "TrackAlgo.h"
#include "object.h"

class match_obj_histManager
   : public histManagerBase
{
     public:
       match_obj_histManager(const string& obj, const float& in_drcut):
       histManagerBase(obj)
       ,drcut(in_drcut) {

        hists["deltar"] = createhist(Form("%s_delta_r", base_name.c_str()), "delta_r;delta_r;yield", 50, 0., drcut);
        hists["ratio"] = createhist(Form("%s_ratio", base_name.c_str()), ";Raw_pt/Raw'_pt;yield", 50, 0.95, 1.05);
       }


     const float get_drcut() const
     {
         return drcut;
     };

     void compareMatching()
     {

         map<string, TH1F*> hists_1 = { {"matched_pt", hists["matched_pt_r"]},
                                       {"unmatched_pt", hists["unmatched_pt_r"]}
         };

         map<string, TH1F*> hists_2 = { {"matched_pt", hists["matched_pt_rp"]},
                                       {"unmatched_pt", hists["unmatched_pt_rp"]}
         };

        compareDist(hists_1, "raw",
                     hists_2, "rawp",
                     get_base_name()+"_");
        Plot_single({"deltar", "ratio"});
     };

     protected:
       float drcut;
};

class match_trackobj_histManager
      : public match_obj_histManager
{
        public:
          match_trackobj_histManager(const string& obj, const float& in_drcut):
          match_obj_histManager(obj, in_drcut)
          {

              bool lowpt = string(base_name).find("lowpt") != std::string::npos;
              for (const auto match_type: {"matched", "unmatched"})
              {
                for (const auto var_type: {"pt"})
                {
                  for (const auto raw_type: {"r", "rp"})
                  {
                     auto key = Form("%s_%s_%s", match_type, var_type, raw_type);
                     hists[key] = createhist(Form("%s_%s", base_name.c_str(), key), Form("%s;pt;yield", key), numBins, (lowpt) ? customBins_lowpt : customBins_highpt);
                  }
                }
              }
           };
};

class match_jetobj_histManager
      : public match_obj_histManager
{         
        public:
          match_jetobj_histManager(const string& obj, const float& in_drcut):
          match_obj_histManager(obj, in_drcut)
          {
          
              for (const auto match_type: {"matched", "unmatched"})
              {
                for (const auto var_type: {"pt"})
                {
                  for (const auto raw_type: {"r", "rp"})
                  {
                     auto key = Form("%s_%s_%s", match_type, var_type, raw_type);
                     hists[key] = createhist(Form("%s_%s", base_name.c_str(), key), Form("%s;pt;yield", key), numBins_jets, customBins_jets);
                  }
                } 
              }
          };

};

