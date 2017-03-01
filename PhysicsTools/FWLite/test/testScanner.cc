/// This is only to make sure that our FWLite tools also compile with gcc
/// that usually spots errors in a much more readable way

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/Scanner.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <TROOT.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <iostream>

int main (int argc, char* argv[]) 
{
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " cmssw_reco_file.root" << std::endl; return 2; }

    FWLiteEnabler::enable();
    gROOT->SetStyle ("Plain");
    gStyle->SetPalette(1);
    gStyle->SetHistMinimumZero(1);


    TFile *f = TFile::Open(argv[1]);

    fwlite::Event ev(f);
    fwlite::Scanner<std::vector<reco::Track> > sc(&ev, "generalTracks");

    TCanvas *c1 = new TCanvas("c1", "c1");
    sc.scan("pt:eta","quality('highPurity')",2);

    sc.setMaxEvents(200); 
    double c = sc.count("quality('highPurity')");
    std::cout << "Found " << c << " highPurity tracks." << std::endl;
    double ce = sc.countEvents();
    std::cout << "Found " << ce << " events." << std::endl;


    TH1 *heta = sc.draw("eta");
    heta->Sumw2(); heta->Scale(1.0/sc.countEvents());
    c1->Print("eta.compiled.png");
    heta->SetLineColor(kBlue);
    TH1 *hp = sc.draw("eta", "quality('highPurity')", "SAME");
    hp->Sumw2(); hp->Scale(1.0/sc.countEvents());
    c1->Print("eta_and_hp.compiled.png");

    TH1 *hb = sc.draw("pt", "abs(eta)<1 && pt < 5", "", "hbarrel");
    TH1 *he = sc.draw("pt", "abs(eta)>1 && pt < 5", "", "hendcaps");
    hb->Draw();
    c1->Print("barrel_1.compiled.png");
    gROOT->FindObject("hbarrel")->Draw();
    c1->Print("barrel_2.compiled.png");
    he->Draw();
    c1->Print("endcaps_1.compiled.png");
    gROOT->FindObject("hendcaps")->Draw();
    c1->Print("endcaps_2.compiled.png");

    hb = sc.draw("pt", "abs(eta)<1 && pt < 5", "NORM");
    hb->SetLineColor(2);
    c1->Print("normalized_pt.compiled.png");
    he = sc.draw("pt", "abs(eta)>1 && pt < 5", "NORM SAME");
    he->SetLineColor(4);
    c1->Print("normalized_pts.compiled.png");

    sc.draw("eta", 5, -2.5, 2.5);
    c1->Print("eta_binned.compiled.png");

    double etabins[4] = { -2.5, -1, 1, 2.5 };
    sc.draw("eta", 3, etabins, "quality('loose')");
    c1->Print("eta_specialbins.compiled.png");

    TProfile *p = sc.drawProf("eta", 5, -2.5, 2.5, "pt",  "quality('loose')");
    p->SetLineColor(kRed);
    c1->Print("eta_prof_pt.compiled.png");

    sc.drawProf("eta", "pt", "quality('highPurity')", "SAME");
    c1->Print("eta_prof_pt_same.compiled.png");

    sc.draw2D("pt", 5, 0, 5, "eta", 4, -2.5, 2.5, "quality('highPurity')", "COLZ");
    c1->Print("pt_eta_2d_manual.compiled.png");
    sc.draw2D("pt", "eta", "quality('highPurity') && pt <= 5", "COLZ");
    c1->Print("pt_eta_2d_auto.compiled.png");

    TGraph *g = sc.drawGraph("pt","eta", "quality('highPurity') && pt <= 5");
    g->SetMarkerStyle(8);
    c1->Print("pt_eta_2d_graph.compiled.png");

    // Now make dN/deta only for events that have two tracks with |eta|<1, pt > 500 MeV
    fwlite::ObjectCountSelector<std::vector<reco::Track> > *ntracks;
    ntracks = new fwlite::ObjectCountSelector<std::vector<reco::Track> >("generalTracks","","", "pt > 0.5 && abs(eta)<1", 2);
    sc.addEventSelector(ntracks);
    TH1 *heta2 = sc.draw("eta",5,-2.5,2.5);
    heta2->Sumw2(); heta2->Scale(1.0/sc.countEvents()); heta2->SetLineColor(4);
    ntracks->setMin(0);
    heta = sc.draw("eta","","SAME");
    heta->Sumw2(); heta->Scale(1.0/sc.countEvents());
    c1->Print("eta_twotracks.compiled.png");

    sc.setMaxEvents(20); 
    RooDataSet *ds = sc.fillDataSet("pt:eta:@hits=hitPattern.numberOfValidHits", "@highPurity=quality('highPurity'):@highPt=pt>2", "pt > 0.5");
    ds->Print("v");
    delete ds;

    return 0;
}


