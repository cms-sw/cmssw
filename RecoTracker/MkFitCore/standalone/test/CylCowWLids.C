// To be used in compiled mode

#include "../CylCowWLids.cc"

#include "TCanvas.h"
#include "TLine.h"

TrackerInfo g_tracker_info;

//------------------------------------------------------------------------------

void print_etas(LayerInfo &li, float dz) {
  float r, z;
  if (li.is_barrel()) {
    r = li.r_mean();
    z = li.zmax();
  } else {
    r = li.rout();
    z = li.z_mean();
  }

  printf("%2d %6.4f %6.4f %6.4f", li.layer_id(), getEta(r, z - dz), getEta(r, z), getEta(r, z + dz));

  if (!li.is_barrel()) {
    r = li.rin();

    printf("  -  %6.4f %6.4f %6.4f", getEta(r, z - dz), getEta(r, z), getEta(r, z + dz));
  }

  printf("\n");
}

//------------------------------------------------------------------------------

void CylCowWLids() {
  Create_TrackerInfo(g_tracker_info, true);

  float zM = 120;
  float rM = 100;

  float cScale = 6;
  TCanvas *c = new TCanvas("cvs", "", cScale * zM, cScale * rM);
  TPad *p = new TPad("pad", "", 0, 0, 1, 1);
  p->Draw();
  p->Update();
  p->cd();

  p->DrawFrame(0, 0, zM, rM);

  printf("Eta coordinates of edges for z0 (-3, 0, +3) cm\n");
  printf("----------------------------------------------\n");

  for (auto i : g_tracker_info.m_barrel) {
    const LayerInfo &li = g_tracker_info.layer(i);

    TLine *l = new TLine(0, li.r_mean(), li.zmax(), li.r_mean());
    l->SetLineColor(kBlue);
    l->SetLineWidth(2);
    l->Draw();

    print_etas(li, 3);
  }

  for (auto i : g_tracker_info.m_ecap_pos) {
    LayerInfo &li = g_tracker_info.layer(i);

    TLine *l = new TLine(li.z_mean(), li.rin(), li.z_mean(), li.rout());
    l->SetLineColor(kMagenta + 3);
    l->SetLineWidth(2);
    l->Draw();

    print_etas(li, 3);
  }

  p->Modified();
  p->Update();
}
