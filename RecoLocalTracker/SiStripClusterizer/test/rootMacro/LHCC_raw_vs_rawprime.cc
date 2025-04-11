////// Saswati Nandan, Inida/INFN,Pisa /////
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <functional>
#include <cassert>

#include "TFile.h"
#include "TDirectoryFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TGraphErrors.h"
#include "TStopwatch.h"

#include "EvthistManager.h"
#include "Match_obj_histManager.h"
#include "object.h"

using namespace std;

constexpr float cut_dzSig = 3.0;  
constexpr float cut_dxySig = 3.0; 
constexpr float cut_ptRes = 0.10; 
constexpr float cut_chi2 = 2;//0.18;  
constexpr float cut_nhits = 11;

auto deltaR(float e1, float e2, float p1, float p2) {

        auto dp = std::abs(p1 -p2);
        if (dp > float(M_PI))
           dp -= float(2 * M_PI);
        return TMath::Sqrt(pow((e1 - e2), 2) + pow(dp, 2));
}


struct TreeReader{

   TTree* tree = NULL;

   long long nentries = 0;
   UInt_t event = 0;
   int run = 0;
   int lumi = 0;
   int nTrk = 0;

   static constexpr int nMax = 40000;
   static constexpr int nMax_jet = 800;

   float trkPt[nMax] = {0};
   float trkEta[nMax] = {0};
   float trkPhi[nMax] = {0};
   float trkDxy1[nMax] = {0};
   float trkDxyError1[nMax] = {0};
   float trkDz1[nMax] = {0};
   float trkDzError1[nMax] = {0};

   Int_t trkAlgo[nMax] = {0};
   Int_t trkNHit[nMax] = {0};
   Int_t trkNdof[nMax] = {0};
   Int_t trkNlayer[nMax] = {0};
   float trkChi2[nMax] = {0};
   float trkPtError[nMax] = {0};
   float inner_xy[nMax] = {0};
   float inner_z[nMax] = {0};

   int nJet = 0;

   float jetPt[nMax_jet] = {0};
   float jetEta[nMax_jet] = {0};
   float jetPhi[nMax_jet] = {0};
   float jetMass[nMax_jet] = {0};

   TreeReader(TTree* in_tree):
   tree(in_tree)
   {
     nentries = tree->GetEntries();
     tree->SetBranchAddress("event", &event);
     tree->SetBranchAddress("run",  &run);
     tree->SetBranchAddress("lumi", &lumi);
     tree->SetBranchAddress("nTracks", &nTrk);
     tree->SetBranchAddress("nJets",  &nJet);

     tree->SetBranchAddress("trkPt",  trkPt);
     tree->SetBranchAddress("trkEta", trkEta);
     tree->SetBranchAddress("trkPhi", trkPhi);
     tree->SetBranchAddress("trkDxy1",trkDxy1);
     tree->SetBranchAddress("trkDxyError1", trkDxyError1);
     tree->SetBranchAddress("trkDz1",  trkDz1);
     tree->SetBranchAddress("trkDzError1", trkDzError1);
     tree->SetBranchAddress("trkAlgo",   trkAlgo);
     tree->SetBranchAddress("trkNHit", trkNHit);
     tree->SetBranchAddress("trkNdof", trkNdof);
     tree->SetBranchAddress("trkNlayer",trkNlayer);
     tree->SetBranchAddress("trkChi2",  trkChi2);
     tree->SetBranchAddress("trkPtError", trkPtError);
     tree->SetBranchAddress("inner_xy", inner_xy);
     tree->SetBranchAddress("inner_z", inner_z);

     tree->SetBranchAddress("jetPt", jetPt);
     tree->SetBranchAddress("jetEta", jetEta);
     tree->SetBranchAddress("jetPhi",jetPhi);
     tree->SetBranchAddress("jetMass", jetMass);

 };
   ~TreeReader()
   {
    delete tree;
   };

};

void event_loop( map< int, map< int, map<int, bool> > >& evtMatchedMap,
                   const TreeReader& treereader,
                   EvthistManager& evthist,
                   map<int, vector<Track> >& r_good_lowpt_trk,
		   map<int, vector<Track> >& r_good_highpt_trk,
                   map<int, vector<Jet> >& r_goodjet
		  ){
       
	std::cout << "analyzing " << std::endl;

	for (int idx = 0; idx < treereader.nentries; ++idx) {

		if(idx%1000 == 0) cout << "Scanning raw tracks: " << idx << "/" << treereader.nentries << endl;

		treereader.tree->GetEntry(idx);

		if ( !evtMatchedMap[treereader.run][treereader.lumi][treereader.event]) continue;

		for (int trkIdx = 0; trkIdx < treereader.nTrk; ++trkIdx)
                {

  	          evthist.fill("trk_dzDdzerr", std::abs(treereader.trkDz1[trkIdx]/treereader.trkDzError1[trkIdx]));
  	          evthist.fill("trk_chi2", treereader.trkChi2[trkIdx]);
                  evthist.fill("trk_nhits", treereader.trkNHit[trkIdx]);
                  evthist.fill("trk_pterrDpt", std::abs(treereader.trkPtError[trkIdx]/treereader.trkPt[trkIdx]));

                  evthist.fill("trk_cutflow", trk_cuts::nocut);
                  evthist.fill("trk_cutflow_z"+to_string(treereader.trkAlgo[trkIdx]), abs(treereader.inner_z[trkIdx]), trk_cuts::nocut);
                  evthist.fill("trk_cutflow_xy"+to_string(treereader.trkAlgo[trkIdx]), abs(treereader.inner_xy[trkIdx]), trk_cuts::nocut);
                  if(treereader.trkChi2[trkIdx] > cut_chi2) continue;
                  evthist.fill("trk_cutflow", trk_cuts::chi2);
                  evthist.fill("trk_cutflow_z"+to_string(treereader.trkAlgo[trkIdx]), abs(treereader.inner_z[trkIdx]), trk_cuts::chi2);
                  evthist.fill("trk_cutflow_xy"+to_string(treereader.trkAlgo[trkIdx]), abs(treereader.inner_xy[trkIdx]), trk_cuts::chi2);
                  if(std::abs(treereader.trkPtError[trkIdx]/treereader.trkPt[trkIdx]) >= cut_ptRes) continue;
                  evthist.fill("trk_cutflow", trk_cuts::ptRes);
                  evthist.fill("trk_cutflow_z"+to_string(treereader.trkAlgo[trkIdx]), abs(treereader.inner_z[trkIdx]), trk_cuts::ptRes);
                  evthist.fill("trk_cutflow_xy"+to_string(treereader.trkAlgo[trkIdx]), abs(treereader.inner_xy[trkIdx]), trk_cuts::ptRes);
                  if((int) treereader.trkNHit[trkIdx] < cut_nhits) continue;
                  evthist.fill("trk_cutflow", trk_cuts::nhits);
                  evthist.fill("trk_cutflow_z"+to_string(treereader.trkAlgo[trkIdx]), abs(treereader.inner_z[trkIdx]), trk_cuts::nhits);
                  evthist.fill("trk_cutflow_xy"+to_string(treereader.trkAlgo[trkIdx]), abs(treereader.inner_xy[trkIdx]), trk_cuts::nhits);
                  evthist.fill("trk_pt", treereader.trkPt[trkIdx]);
		  evthist.fill("trk_eta", treereader.trkEta[trkIdx]);
		  evthist.fill("trk_dxyDdxyerr", treereader.trkDxy1[trkIdx]/treereader.trkDxyError1[trkIdx]);
                  
                  evthist.fill("trk_eta_phi", treereader.trkEta[trkIdx], treereader.trkPhi[trkIdx]);
                  evthist.fill("trk_inner_xy", treereader.inner_xy[trkIdx]);
                  evthist.fill("trk_inner_z", treereader.inner_z[trkIdx]);

		  if (treereader.trkPt[trkIdx] < 1.0)
                     r_good_lowpt_trk[treereader.event].emplace_back(trkIdx, treereader.trkPt[trkIdx],
                       treereader.trkEta[trkIdx], treereader.trkPhi[trkIdx],
                       treereader.trkDxy1[trkIdx], treereader.trkDxyError1[trkIdx],
                       treereader.trkDz1[trkIdx], treereader.trkDzError1[trkIdx],
                       treereader.trkAlgo[trkIdx], treereader.trkNHit[trkIdx],
                       treereader.trkNdof[trkIdx], treereader.trkNlayer[trkIdx],
                       treereader.trkChi2[trkIdx], treereader.trkPtError[trkIdx]);
		  else
	             r_good_highpt_trk[treereader.event].emplace_back(trkIdx, treereader.trkPt[trkIdx],
                       treereader.trkEta[trkIdx], treereader.trkPhi[trkIdx],
                       treereader.trkDxy1[trkIdx], treereader.trkDxyError1[trkIdx],
                       treereader.trkDz1[trkIdx], treereader.trkDzError1[trkIdx],
                       treereader.trkAlgo[trkIdx], treereader.trkNHit[trkIdx],
                       treereader.trkNdof[trkIdx], treereader.trkNlayer[trkIdx],
                       treereader.trkChi2[trkIdx], treereader.trkPtError[trkIdx]);
                }

                 ///// jet ////

                 for (int jetIdx = 0; jetIdx < treereader.nJet; ++jetIdx)
                 {

                   evthist.fill("jet_pt",  treereader.jetPt[jetIdx]);
                   evthist.fill("jet_eta", treereader.jetEta[jetIdx]);

                   evthist.fill("jet_cutflow", jet_cuts::nocut);

                   if(treereader.jetPt[jetIdx] < 20 ) continue;
                   evthist.fill("jet_cutflow", jet_cuts::pt);

                   if(abs(treereader.jetEta[jetIdx]) > 2.4) continue;
                   evthist.fill("jet_cutflow", jet_cuts::eta);

                   r_goodjet[treereader.event].emplace_back(jetIdx,
                      treereader.jetPt[jetIdx], treereader.jetEta[jetIdx],
                      treereader.jetPhi[jetIdx], treereader.jetMass[jetIdx]);
                 }
           }
}

struct match_property{

float drmin;
float r_pt;
float rp_pt;
int   r_idx;
int   rp_idx;

  match_property():
          drmin(-1)
          ,r_pt(-1)
          ,rp_pt(-1)
          ,r_idx(-1)
          ,rp_idx(-1)
          {};

  match_property(float in_drmin, float in_r_pt, float in_rp_pt,
                 int in_r_idx,  int in_rp_idx
                ):
                drmin(in_drmin)
               ,r_pt(in_r_pt)
               ,rp_pt(in_rp_pt)
               ,r_idx(in_r_idx)
               ,rp_idx(in_rp_idx)
               {};
};

template<class T, class M>

void do_matching(const map<int, vector<T> > & r_objs, const map<int, vector<T> >& rp_objs,
	        M & obj_hists
		) 
        {

	  cout << "doing matching " << endl;

	  std::map<int, std::vector<int> >matched_trk_p;

	  int not_matched_obj_r = 0;
	  int not_matched_obj_rp = 0;

          int total_obj_r(0);
          int total_obj_rp(0);

	  for(auto const & [e_r, objs_r]: r_objs)
	  {
             map<int, match_property> matched_objs; // key:index of obj_rp, value: pt values of matched objs, drmin, idx of objs
             map<int, match_property> unmatched_objs_r;
             bool rawp_event_present(true);
	     for(auto const & obj_r: objs_r)
             {
               ++total_obj_r;
               float drmin = 9999;
               match_property tmp = match_property();
               if (rp_objs.find(e_r) == rp_objs.end()) {
                   cout << "no object found in rawp event with event # " << e_r << endl;
                   rawp_event_present = false;
                   break;
               }
               auto objs_rp = rp_objs.at(e_r);
	       for(auto const & obj_rp: objs_rp)
	       {
	           auto dr = deltaR(obj_r.eta, obj_rp.eta, obj_r.phi, obj_rp.phi);
		   if (dr < obj_hists.get_drcut() && dr < drmin) {
                      drmin = dr;
		      tmp = match_property(drmin, obj_r.pt, obj_rp.pt, obj_r.idx, obj_rp.idx);
		   }
	       } // end of objs_rp loop
               if (tmp.rp_idx != -1)
               {
                   if(matched_objs.count(tmp.rp_idx) != 0)
                   {
                     if (drmin < matched_objs[tmp.rp_idx].drmin)
                     {
                        unmatched_objs_r[matched_objs[tmp.rp_idx].r_idx] = match_property(matched_objs[tmp.rp_idx].drmin,-1,-1, -1, -1);
                        matched_objs[tmp.rp_idx] = tmp;
                     }
                   }
                   else
                      matched_objs[tmp.rp_idx] = tmp;
               }
               else
                  unmatched_objs_r[obj_r.idx] = match_property();
             } // end of objs_r loop

             for(auto const & obj_r: objs_r)
             {
               if(unmatched_objs_r.count(obj_r.idx))
               {
                  not_matched_obj_r++;
                  obj_hists.fill("unmatched_pt_r", obj_r.pt);
               }
             }
             if (!rawp_event_present) continue;
             for(auto const & obj_rp: rp_objs.at(e_r))
             {
               total_obj_rp++;
               if(matched_objs.count(obj_rp.idx) == 0)
               {
                  not_matched_obj_rp++;
                  obj_hists.fill("unmatched_pt_rp", obj_rp.pt);
               }
             }

             for(auto const& [matched_idx, matched_obj]: matched_objs)
             {
               obj_hists.fill("deltar", matched_obj.drmin);
               obj_hists.fill("matched_pt_r", matched_obj.r_pt);
               obj_hists.fill("matched_pt_r", matched_obj.rp_pt);
               obj_hists.fill("ratio", matched_obj.r_pt/matched_obj.rp_pt);
             }

          } // end of r_objs loop

	  cout << setprecision(2);
          cout << Form("total %s in raw: ", obj_hists.get_base_name().c_str()) << total_obj_r << endl;
          cout << Form("total %s in rawp: ", obj_hists.get_base_name().c_str()) << total_obj_rp << endl;
          cout << Form("total unmatched %s in raw: ", obj_hists.get_base_name().c_str()) << not_matched_obj_r << endl;
          cout << Form("total unmatched %s in rawp: ", obj_hists.get_base_name().c_str()) << not_matched_obj_rp << endl;	
          cout << Form("not matched %s: in raw ", obj_hists.get_base_name().c_str()) << (100.*not_matched_obj_r/total_obj_r) << "%" << endl;
          cout << Form("not matched %s: in rawp ", obj_hists.get_base_name().c_str()) << (not_matched_obj_rp*100./total_obj_rp) << "%" << endl;

}

int main(int argc, char const *argv[]) { //LHCC_raw_vs_rawprime() {

	std::string expTag = "test";
	map< int, map< int, map<int, bool> > > evtMatchedMap;
	map< int, map< int, map<int, bool> > > evtMap;
 
	TFile* f1                = TFile::Open(argv[1], "read");
        TreeReader treereader_rp ((TTree*) f1->Get("flatNtuple/tree"));
        
	TFile* f2               = TFile::Open(argv[2], "read");
        TreeReader treereader_r ((TTree*) f2->Get("flatNtuple/tree"));

	TFile* f = new TFile("object_study.root", "recreate"); 

	for (int r_idx = 0; r_idx < treereader_r.nentries; ++r_idx) {
                treereader_r.tree->GetEntry(r_idx);
		evtMatchedMap[treereader_r.run][treereader_r.lumi][treereader_r.event] = 0;
	}

	for (int rp_idx = 0; rp_idx < treereader_rp.nentries; ++rp_idx) {

		treereader_rp.tree->GetEntry(rp_idx);
		
		if (  evtMatchedMap.find(treereader_rp.run) == evtMatchedMap.end() ) continue;
                if (  evtMatchedMap[treereader_rp.run].find(treereader_rp.lumi) == evtMatchedMap[treereader_rp.run].end() ) continue;
		if (  evtMatchedMap[treereader_rp.run][treereader_rp.lumi].find(treereader_rp.event) == evtMatchedMap[treereader_rp.run][treereader_rp.lumi].end() ) continue;

		evtMatchedMap[treereader_rp.run][treereader_rp.lumi][treereader_rp.event] = 1;

	}

	cout << "creating hists for raw " << endl;

	EvthistManager evthist_r("raw");
        
	//// raw ///

	map<int, vector<Track> > r_good_lowpt_trk, r_good_highpt_trk;
        map<int, vector<Jet> > r_goodjet;

	cout << "calling eventloop for raw" << endl;

	event_loop(evtMatchedMap, treereader_r, evthist_r,
		   r_good_lowpt_trk, r_good_highpt_trk, r_goodjet
		  );

	/////// rawprime ////

	cout << "creating hists for rawp" << endl;

	EvthistManager evthist_rp("rawp");
	map<int, vector<Track> > rp_good_lowpt_trk, rp_good_highpt_trk;
       	map<int, vector<Jet> > rp_goodjet;	

	cout << "calling eventloop for rawp" << endl;

	event_loop(evtMatchedMap, treereader_rp, evthist_rp,
                   rp_good_lowpt_trk, rp_good_highpt_trk, rp_goodjet
                  );	
	
	f->cd();

	evthist_r.write();
	evthist_rp.write();
	evthist_r.compareDist(evthist_rp);

	cout << "calling matching" << endl;

        {
           match_trackobj_histManager trk_lowpt_hists("tracks_lowpt", 0.05);
           do_matching(r_good_lowpt_trk, rp_good_lowpt_trk,
                    trk_lowpt_hists
           );
           trk_lowpt_hists.write();
           trk_lowpt_hists.compareMatching();
        }

	{
           match_trackobj_histManager trk_highpt_hists("tracks_highpt", 0.05);
           do_matching(r_good_highpt_trk, rp_good_highpt_trk,
                    trk_highpt_hists
           );
           trk_highpt_hists.write();
           trk_highpt_hists.compareMatching();
        }
       
        {
           match_jetobj_histManager jet_hists("jets", 0.4);
           do_matching(r_goodjet, rp_goodjet,
                    jet_hists
           );
           jet_hists.write();
           jet_hists.compareMatching();
        }

        cout << "matching done" << endl;
        f->Close();	

	return 0;
}
