#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"

using modBins = std::vector<std::pair<int, int>>;
using rocBins = std::vector<std::tuple<int, int, int>>;

// find detector coordinates for filling
/*--------------------------------------------------------------------*/
DetCoordinates Phase1PixelROCMaps::findDetCoordinates(const uint32_t& t_detid)
/*--------------------------------------------------------------------*/
{
  DetCoordinates coord;

  auto myDetId = DetId(t_detid);
  int subid = DetId(t_detid).subdetId();

  if (subid == PixelSubdetector::PixelBarrel) {
    coord.m_layer = m_trackerTopo.pxbLayer(myDetId);
    coord.m_s_ladder = this->signed_ladder(myDetId, true);
    coord.m_s_module = this->signed_module(myDetId, true);

    bool isFlipped = this->isBPixOuterLadder(myDetId, false);
    if ((coord.m_layer > 1 && coord.m_s_module < 0))
      isFlipped = !isFlipped;

    coord.m_isFlipped = isFlipped;

  }  // if it's barrel
  else if (subid == PixelSubdetector::PixelEndcap) {
    coord.m_ring = this->ring(myDetId, true);
    coord.m_s_blade = this->signed_blade(myDetId, true);
    coord.m_s_disk = this->signed_disk(myDetId, true);
    coord.m_panel = m_trackerTopo.pxfPanel(t_detid);
    coord.m_isFlipped = (coord.m_s_disk > 0) ? (coord.m_panel == 1) : (coord.m_panel == 2);
  }  // it it's endcap
  else {
    throw cms::Exception("LogicError") << "Unknown Pixel SubDet ID " << std::endl;
  }

  if (std::strcmp(m_option, kVerbose) == 0) {
    coord.printCoordinates();
  }

  return coord;
}

// overloaded method: mask entire module
/*--------------------------------------------------------------------*/
modBins Phase1PixelROCMaps::maskedBarrelRocsToBins(DetCoordinates coord)
/*--------------------------------------------------------------------*/
{
  modBins rocsToMask;
  int nlad = nlad_list[coord.m_layer - 1];

  int start_x = coord.m_s_module > 0 ? ((coord.m_s_module + 4) * 8) + 1 : ((4 - (std::abs(coord.m_s_module))) * 8) + 1;
  int start_y =
      coord.m_s_ladder > 0 ? ((coord.m_s_ladder + nlad) * 2) + 1 : ((nlad - (std::abs(coord.m_s_ladder))) * 2) + 1;

  int end_x = start_x + 7;
  int end_y = start_y + 1;

  for (int bin_x = 1; bin_x <= 72; bin_x++) {
    for (int bin_y = 1; bin_y <= (nlad * 4 + 2); bin_y++) {
      if (bin_x >= start_x && bin_x <= end_x && bin_y >= start_y && bin_y <= end_y) {
        rocsToMask.push_back(std::make_pair(bin_x, bin_y));
      }
    }
  }
  return rocsToMask;
}

// overloaded method: mask single ROCs
/*--------------------------------------------------------------------*/
rocBins Phase1PixelROCMaps::maskedBarrelRocsToBins(DetCoordinates coord, std::bitset<16> myRocs)
/*--------------------------------------------------------------------*/
{
  rocBins rocsToMask;
  int nlad = nlad_list[coord.m_layer - 1];

  int start_x = coord.m_s_module > 0 ? ((coord.m_s_module + 4) * 8) + 1 : ((4 - (std::abs(coord.m_s_module))) * 8) + 1;
  int start_y =
      coord.m_s_ladder > 0 ? ((coord.m_s_ladder + nlad) * 2) + 1 : ((nlad - (std::abs(coord.m_s_ladder))) * 2) + 1;

  int roc0_x = ((coord.m_layer == 1) || (coord.m_layer > 1 && coord.m_s_module > 0)) ? start_x + 7 : start_x;
  int roc0_y = start_y - 1;

  size_t idx = 0;
  while (idx < myRocs.size()) {
    if (myRocs.test(idx)) {
      //////////////////////////////////////////////////////////////////////////////////////
      //		                        |					  //
      // In BPix Layer1 & module > 0 in L2,3,4  |   In BPix Layer 2,3,4 module < 0        //
      //                                        |					  //
      // ROCs are ordered in the following      |   ROCs are ordered in the following     //
      // fashion for unflipped modules          |   fashion for unflipped modules         //
      //				        |  				          //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      // | 8 |9  |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      //					|  					  //
      // if the module is flipped the ordering  |   if the module is flipped the ordering //
      // is reveresed                           |   is reversed                           //
      //					|                                         //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      // | 8 | 9 |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      //////////////////////////////////////////////////////////////////////////////////////

      int roc_x(0), roc_y(0);

      if ((coord.m_layer == 1) || (coord.m_layer > 1 && coord.m_s_module > 0)) {
        if (!coord.m_isFlipped) {
          roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
          roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
        } else {
          roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
          roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
        }
      } else {
        if (!coord.m_isFlipped) {
          roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
          roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
        } else {
          roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
          roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
        }
      }
      rocsToMask.push_back(std::make_tuple(roc_x, roc_y, idx));
    }
    ++idx;
  }
  return rocsToMask;
}

// overloaded method: mask entire module
/*--------------------------------------------------------------------*/
modBins Phase1PixelROCMaps::maskedForwardRocsToBins(DetCoordinates coord)
/*--------------------------------------------------------------------*/
{
  modBins rocsToMask;
  int nybins = nybins_list[coord.m_ring - 1];

  int start_x = coord.m_s_disk > 0 ? ((coord.m_s_disk + 3) * 8) + 1 : ((3 - (std::abs(coord.m_s_disk))) * 8) + 1;
  int start_y = coord.m_s_blade > 0 ? (nybins / 2) + (coord.m_s_blade * 4) - (coord.m_panel * 2) + 3
                                    : ((nybins / 2) - (std::abs(coord.m_s_blade) * 4) - coord.m_panel * 2) + 3;

  int end_x = start_x + 7;
  int end_y = start_y + 1;

  for (int bin_x = 1; bin_x <= 56; bin_x++) {
    for (int bin_y = 1; bin_y <= nybins; bin_y++) {
      if (bin_x >= start_x && bin_x <= end_x && bin_y >= start_y && bin_y <= end_y) {
        rocsToMask.push_back(std::make_pair(bin_x, bin_y));
      }
    }
  }
  return rocsToMask;
}

// overloaded method: mask single ROCs
/*--------------------------------------------------------------------*/
rocBins Phase1PixelROCMaps::maskedForwardRocsToBins(DetCoordinates coord, std::bitset<16> myRocs)
/*--------------------------------------------------------------------*/
{
  rocBins rocsToMask;
  int nybins = nybins_list[coord.m_ring - 1];

  int start_x = coord.m_s_disk > 0 ? ((coord.m_s_disk + 3) * 8) + 1 : ((3 - (std::abs(coord.m_s_disk))) * 8) + 1;
  int start_y = coord.m_s_blade > 0 ? (nybins / 2) + (coord.m_s_blade * 4) - (coord.m_panel * 2) + 3
                                    : ((nybins / 2) - (std::abs(coord.m_s_blade) * 4) - coord.m_panel * 2) + 3;

  int roc0_x = coord.m_s_disk > 0 ? start_x + 7 : start_x;
  int roc0_y = start_y - 1;

  size_t idx = 0;
  while (idx < myRocs.size()) {
    if (myRocs.test(idx)) {
      int roc_x(0), roc_y(0);

      //////////////////////////////////////////////////////////////////////////////////////
      //		                        |					  //
      // In FPix + (Disk 1,2,3)                 |   In FPix - (Disk -1,-2,-3)	          //
      //                                        |					  //
      // ROCs are ordered in the following      |   ROCs are ordered in the following     //
      // fashion for unflipped modules          |   fashion for unflipped modules         //
      //					|  				          //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      // | 8 |9  |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      //					|  					  //
      // if the module is flipped the ordering  |   if the module is flipped the ordering //
      // is reveresed                           |   is reversed                           //
      //					|                                         //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      // | 8 | 9 |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
      // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
      //////////////////////////////////////////////////////////////////////////////////////

      if (coord.m_s_disk > 0) {
        if (!coord.m_isFlipped) {
          roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
          roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
        } else {
          roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
          roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
        }
      } else {
        if (!coord.m_isFlipped) {
          roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
          roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
        } else {
          roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
          roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
        }
      }

      rocsToMask.push_back(std::make_tuple(roc_x, roc_y, idx));
    }
    ++idx;
  }
  return rocsToMask;
}

/*--------------------------------------------------------------------*/
void Phase1PixelROCMaps::fillWholeModule(const uint32_t& detid, double value)
/*--------------------------------------------------------------------*/
{
  auto coord = findDetCoordinates(detid);
  auto rocsToMark = coord.isBarrel() ? this->maskedBarrelRocsToBins(coord) : this->maskedForwardRocsToBins(coord);

  if (coord.isBarrel()) {
    for (const auto& bin : rocsToMark) {
      double x = h_bpix_maps[coord.m_layer - 1]->GetXaxis()->GetBinCenter(bin.first);
      double y = h_bpix_maps[coord.m_layer - 1]->GetYaxis()->GetBinCenter(bin.second);
      h_bpix_maps[coord.m_layer - 1]->Fill(x, y, value);
    }
  } else {
    for (const auto& bin : rocsToMark) {
      double x = h_fpix_maps[coord.m_ring - 1]->GetXaxis()->GetBinCenter(bin.first);
      double y = h_fpix_maps[coord.m_ring - 1]->GetYaxis()->GetBinCenter(bin.second);
      h_fpix_maps[coord.m_ring - 1]->Fill(x, y, value);
    }
  }
  return;
}

/*--------------------------------------------------------------------*/
void Phase1PixelROCMaps::fillSelectedRocs(const uint32_t& detid, const std::bitset<16>& theROCs, double value)
/*--------------------------------------------------------------------*/
{
  auto coord = findDetCoordinates(detid);
  auto rocsToMark =
      coord.isBarrel() ? this->maskedBarrelRocsToBins(coord, theROCs) : this->maskedForwardRocsToBins(coord, theROCs);

  if (coord.isBarrel()) {
    for (const auto& bin : rocsToMark) {
      double x = h_bpix_maps[coord.m_layer - 1]->GetXaxis()->GetBinCenter(std::get<0>(bin));
      double y = h_bpix_maps[coord.m_layer - 1]->GetYaxis()->GetBinCenter(std::get<1>(bin));
      h_bpix_maps[coord.m_layer - 1]->Fill(x, y, value);
    }
  } else {
    for (const auto& bin : rocsToMark) {
      double x = h_fpix_maps[coord.m_ring - 1]->GetXaxis()->GetBinCenter(std::get<0>(bin));
      double y = h_fpix_maps[coord.m_ring - 1]->GetYaxis()->GetBinCenter(std::get<1>(bin));
      h_fpix_maps[coord.m_ring - 1]->Fill(x, y, value);
    }
  }

  return;
}

/*--------------------------------------------------------------------*/
void PixelROCMapHelper::draw_line(
    double x1, double x2, double y1, double y2, int width = 2, int style = 1, int color = 1)
/*--------------------------------------------------------------------*/
{
  TLine* l = new TLine(x1, y1, x2, y2);
  l->SetBit(kCanDelete);
  l->SetLineWidth(width);
  l->SetLineStyle(style);
  l->SetLineColor(color);
  l->Draw();
}

/*--------------------------------------------------------------------*/
void PixelROCMapHelper::dress_plot(TPad*& canv,
                                   TH2* h,
                                   int lay,
                                   int ring = 0,
                                   int phase = 0,
                                   bool standard_palette = true,
                                   bool half_shift = true,
                                   bool mark_zero = true)
/*--------------------------------------------------------------------*/
{
  std::string s_title;
  const auto zAxisTitle = fmt::sprintf("%s", h->GetZaxis()->GetTitle());

  if (lay > 0) {
    canv->cd(lay);
    canv->cd(lay)->SetTopMargin(0.05);
    canv->cd(lay)->SetBottomMargin(0.07);
    canv->cd(lay)->SetLeftMargin(0.1);
    if (!zAxisTitle.empty()) {
      h->GetZaxis()->SetTitleOffset(1.3);
      h->GetZaxis()->CenterTitle(true);
      canv->cd(lay)->SetRightMargin(0.14);
    } else {
      canv->cd(lay)->SetRightMargin(0.11);
    }
    s_title = "Barrel Pixel Layer " + std::to_string(lay);
  } else {
    canv->cd(ring);
    canv->cd(ring)->SetTopMargin(0.05);
    canv->cd(ring)->SetBottomMargin(0.07);
    canv->cd(ring)->SetLeftMargin(0.1);
    if (!zAxisTitle.empty()) {
      h->GetZaxis()->SetTitleOffset(1.3);
      h->GetZaxis()->CenterTitle(true);
      canv->cd(ring)->SetRightMargin(0.14);
    } else {
      canv->cd(ring)->SetRightMargin(0.11);
    }
    if (ring > 4) {
      ring = ring - 4;
    }
    s_title = "Forward Pixel Ring " + std::to_string(ring);
  }

  if (standard_palette) {
    gStyle->SetPalette(1);
  } else {
    /*
    const Int_t NRGBs = 5;
    const Int_t NCont = 255;

    Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
    Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
    Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
    Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);
    */

    // this is the fine gradient palette (blue to red)
    double max = h->GetMaximum();
    double min = h->GetMinimum();
    double val_white = 0.;
    double per_white = (max != min) ? ((val_white - min) / (max - min)) : 0.5;

    const int Number = 3;
    double Red[Number] = {0., 1., 1.};
    double Green[Number] = {0., 1., 0.};
    double Blue[Number] = {1., 1., 0.};
    double Stops[Number] = {0., per_white, 1.};
    int nb = 256;
    h->SetContour(nb);
    TColor::CreateGradientColorTable(Number, Stops, Red, Green, Blue, nb);
    // if max == min impose the range to be the same as it was a real diff
    if (max == min)
      h->GetZaxis()->SetRangeUser(-1., 1.);
  }

  h->SetMarkerSize(0.7);
  h->Draw("colz1");

  auto ltx = TLatex();
  ltx.SetTextFont(62);
  ltx.SetTextColor(1);
  ltx.SetTextSize(0.06);
  ltx.SetTextAlign(31);
  ltx.DrawLatexNDC(1 - gPad->GetRightMargin(), 1 - gPad->GetTopMargin() + 0.01, (s_title).c_str());

  // Draw Lines around modules
  if (lay > 0) {
    std::vector<std::vector<int>> nladder = {{10, 16, 22}, {6, 14, 22, 32}};
    int nlad = nladder[phase][lay - 1];
    for (int xsign = -1; xsign <= 1; xsign += 2)
      for (int ysign = -1; ysign <= 1; ysign += 2) {
        float xlow = xsign * (half_shift * 0.5);
        float xhigh = xsign * (half_shift * 0.5 + 4);
        float ylow = ysign * (half_shift * 0.5 + (phase == 0) * 0.5);
        float yhigh = ysign * (half_shift * 0.5 - (phase == 0) * 0.5 + nlad);
        // Outside box
        PixelROCMapHelper::draw_line(xlow, xhigh, ylow, ylow, 1);    // bottom
        PixelROCMapHelper::draw_line(xlow, xhigh, yhigh, yhigh, 1);  // top
        PixelROCMapHelper::draw_line(xlow, xlow, ylow, yhigh, 1);    // left
        PixelROCMapHelper::draw_line(xhigh, xhigh, ylow, yhigh, 1);  // right
        // Inner Horizontal lines
        for (int lad = 1; lad < nlad; ++lad) {
          float y = ysign * (lad + half_shift * 0.5);
          PixelROCMapHelper::draw_line(xlow, xhigh, y, y, 1);
        }
        for (int lad = 1; lad <= nlad; ++lad)
          if (!(phase == 0 && (lad == 1 || lad == nlad))) {
            float y = ysign * (lad + half_shift * 0.5 - 0.5);
            PixelROCMapHelper::draw_line(xlow, xhigh, y, y, 1, 3);
          }
        // Inner Vertical lines
        for (int mod = 1; mod < 4; ++mod) {
          float x = xsign * (mod + half_shift * 0.5);
          PixelROCMapHelper::draw_line(x, x, ylow, yhigh, 1);
        }
        // Make a BOX around ROC 0
        // Phase 0 - ladder +1 is always non-flipped
        // Phase 1 - ladder +1 is always     flipped
        if (mark_zero) {
          for (int mod = 1; mod <= 4; ++mod)
            for (int lad = 1; lad <= nlad; ++lad) {
              bool flipped = ysign == 1 ? lad % 2 == 0 : lad % 2 == 1;
              if (phase == 1)
                flipped = !flipped;
              int roc0_orientation = flipped ? -1 : 1;
              if (xsign == -1)
                roc0_orientation *= -1;
              if (ysign == -1)
                roc0_orientation *= -1;
              float x1 = xsign * (mod + half_shift * 0.5);
              float x2 = xsign * (mod + half_shift * 0.5 - 1. / 8);
              float y1 = ysign * (lad + half_shift * 0.5 - 0.5);
              float y2 = ysign * (lad + half_shift * 0.5 - 0.5 + roc0_orientation * 1. / 2);
              if (!(phase == 0 && (lad == 1 || lad == nlad) && xsign == -1)) {
                if (lay == 1 && xsign <= -1) {
                  float x1 = xsign * ((mod - 1) + half_shift * 0.5);
                  float x2 = xsign * ((mod - 1) + half_shift * 0.5 + 1. / 8);
                  float y1 = ysign * (lad + half_shift * 0.5 - 0.5 + roc0_orientation);
                  float y2 = ysign * (lad + half_shift * 0.5 - 0.5 + roc0_orientation * 3. / 2);
                  PixelROCMapHelper::draw_line(x1, x2, y1, y1, 1);
                  PixelROCMapHelper::draw_line(x2, x2, y1, y2, 1);
                } else {
                  PixelROCMapHelper::draw_line(x1, x2, y1, y1, 1);
                  //PixelROCMapHelper::draw_line(x1, x2, y2, y2, 1);
                  //PixelROCMapHelper::draw_line(x1, x1, y1, y2, 1);
                  PixelROCMapHelper::draw_line(x2, x2, y1, y2, 1);
                }
              }
            }
        }
      }
  } else {
    // FPIX
    for (int dsk = 1, ndsk = 2 + (phase == 1); dsk <= ndsk; ++dsk) {
      for (int xsign = -1; xsign <= 1; xsign += 2)
        for (int ysign = -1; ysign <= 1; ysign += 2) {
          if (phase == 0) {
            int first_roc = 3, nbin = 16;
            for (int bld = 1, nbld = 12; bld <= nbld; ++bld) {
              // Horizontal lines
              for (int plq = 1, nplq = 7; plq <= nplq; ++plq) {
                float xlow =
                    xsign * (half_shift * 0.5 + dsk - 1 + (first_roc - 3 + 2 * plq + (plq == 1)) / (float)nbin);
                float xhigh =
                    xsign * (half_shift * 0.5 + dsk - 1 + (first_roc - 3 + 2 * (plq + 1) - (plq == 7)) / (float)nbin);
                float ylow = ysign * (half_shift * 0.5 + (bld - 0.5) - (2 + plq / 2) * 0.1);
                float yhigh = ysign * (half_shift * 0.5 + (bld - 0.5) + (2 + plq / 2) * 0.1);
                PixelROCMapHelper::draw_line(xlow, xhigh, ylow, ylow, 1);    // bottom
                PixelROCMapHelper::draw_line(xlow, xhigh, yhigh, yhigh, 1);  // top
              }
              // Vertical lines
              for (int plq = 1, nplq = 7 + 1; plq <= nplq; ++plq) {
                float x = xsign * (half_shift * 0.5 + dsk - 1 +
                                   (first_roc - 3 + 2 * plq + (plq == 1) - (plq == 8)) / (float)nbin);
                float ylow = ysign * (half_shift * 0.5 + (bld - 0.5) - (2 + (plq - (plq == 8)) / 2) * 0.1);
                float yhigh = ysign * (half_shift * 0.5 + (bld - 0.5) + (2 + (plq - (plq == 8)) / 2) * 0.1);
                PixelROCMapHelper::draw_line(x, x, ylow, yhigh, 1);
              }
              // Panel 2 has dashed mid-plane
              for (int plq = 2, nplq = 6; plq <= nplq; ++plq)
                if (plq % 2 == 0) {
                  float x = xsign * (half_shift * 0.5 + dsk - 1 +
                                     (first_roc - 3 + 2 * plq + (plq == 1) - (plq == 8) + 1) / (float)nbin);
                  float ylow = ysign * (half_shift * 0.5 + (bld - 0.5) - (2 + (plq - (plq == 8)) / 2) * 0.1);
                  float yhigh = ysign * (half_shift * 0.5 + (bld - 0.5) + (2 + (plq - (plq == 8)) / 2) * 0.1);
                  PixelROCMapHelper::draw_line(x, x, ylow, yhigh, 1, 2);
                }
              // Make a BOX around ROC 0
              for (int plq = 1, nplq = 7; plq <= nplq; ++plq) {
                float x1 = xsign * (half_shift * 0.5 + dsk - 1 + (first_roc - 3 + 2 * plq + (plq == 1)) / (float)nbin);
                float x2 =
                    xsign * (half_shift * 0.5 + dsk - 1 + (first_roc - 3 + 2 * plq + (plq == 1) + 1) / (float)nbin);
                int sign = xsign * ysign * ((plq % 2) ? 1 : -1);
                float y1 = ysign * (half_shift * 0.5 + (bld - 0.5) + sign * (2 + plq / 2) * 0.1);
                float y2 = ysign * (half_shift * 0.5 + (bld - 0.5) + sign * (plq / 2) * 0.1);
                //PixelROCMapHelper::draw_line(x1, x2, y1, y1, 1);
                PixelROCMapHelper::draw_line(x1, x2, y2, y2, 1);
                //PixelROCMapHelper::draw_line(x1, x1, y1, y2, 1);
                PixelROCMapHelper::draw_line(x2, x2, y1, y2, 1);
              }
            }
          } else if (phase == 1) {
            if (ring == 0) {  // both
              for (int ring = 1; ring <= 2; ++ring)
                for (int bld = 1, nbld = 5 + ring * 6; bld <= nbld; ++bld) {
                  float scale = (ring == 1) ? 1.5 : 1;
                  Color_t p1_color = 1, p2_color = 1;
                  // Horizontal lines
                  // Panel 2 has dashed mid-plane
                  float x1 = xsign * (half_shift * 0.5 + dsk - 1 + (ring - 1) * 0.5);
                  float x2 = xsign * (half_shift * 0.5 + dsk - 1 + ring * 0.5);
                  int sign = ysign;
                  float y1 = ysign * (half_shift * 0.5 - 0.5 + scale * bld + sign * 0.5);
                  //float yp1_mid = ysign * (half_shift*0.5 - 0.5 + scale*bld + sign*0.25);
                  float y2 = ysign * (half_shift * 0.5 - 0.5 + scale * bld);
                  float yp2_mid = ysign * (half_shift * 0.5 - 0.5 + scale * bld - sign * 0.25);
                  float y3 = ysign * (half_shift * 0.5 - 0.5 + scale * bld - sign * 0.5);
                  PixelROCMapHelper::draw_line(x1, x2, y1, y1, 1, 1, p1_color);
                  //PixelROCMapHelper::draw_line(x1, x2, yp1_mid, yp1_mid, 1, 3);
                  PixelROCMapHelper::draw_line(x1, x2, y2, y2, 1, 1, p1_color);
                  PixelROCMapHelper::draw_line(x1, x2, yp2_mid, yp2_mid, 1, 2);
                  PixelROCMapHelper::draw_line(x1, x2, y3, y3, 1, 1, p2_color);
                  // Vertical lines
                  float x = xsign * (half_shift * 0.5 + dsk - 1 + (ring - 1) * 0.5);
                  PixelROCMapHelper::draw_line(x, x, y1, y2, 1, 1, p1_color);
                  PixelROCMapHelper::draw_line(x, x, y2, y3, 1, 1, p2_color);
                  if (ring == 2) {
                    //PixelROCMapHelper::draw_line(x,  x,  y2,  y3, 1, 1, p1_color);
                    x = xsign * (half_shift * 0.5 + dsk);
                    PixelROCMapHelper::draw_line(x, x, y1, y2, 1, 1, p1_color);
                    PixelROCMapHelper::draw_line(x, x, y2, y3, 1, 1, p2_color);
                  }
                  // Make a BOX around ROC 0
                  x1 = xsign * (half_shift * 0.5 + dsk - 1 + ring * 0.5 - 1 / 16.);
                  x2 = xsign * (half_shift * 0.5 + dsk - 1 + ring * 0.5);
                  float y1_p1 = ysign * (half_shift * 0.5 - 0.5 + scale * bld + sign * 0.25);
                  float y2_p1 = ysign * (half_shift * 0.5 - 0.5 + scale * bld + sign * 0.25 + xsign * ysign * 0.25);
                  PixelROCMapHelper::draw_line(x1, x2, y1_p1, y1_p1, 1);
                  //PixelROCMapHelper::draw_line(x1, x2, y2_p1, y2_p1, 1);
                  PixelROCMapHelper::draw_line(x1, x1, y1_p1, y2_p1, 1);
                  //PixelROCMapHelper::draw_line(x2, x2, y1_p1, y2_p1, 1);
                  float y1_p2 = ysign * (half_shift * 0.5 - 0.5 + scale * bld - sign * 0.25);
                  float y2_p2 = ysign * (half_shift * 0.5 - 0.5 + scale * bld - sign * 0.25 - xsign * ysign * 0.25);
                  PixelROCMapHelper::draw_line(x1, x2, y1_p2, y1_p2, 1);
                  //PixelROCMapHelper::draw_line(x1, x2, y2_p2, y2_p2, 1);
                  PixelROCMapHelper::draw_line(x1, x1, y1_p2, y2_p2, 1);
                  //PixelROCMapHelper::draw_line(x2, x2, y1_p2, y2_p2, 1);
                }
            } else {  // only one ring, 1 or 2
              for (int bld = 1, nbld = 5 + ring * 6; bld <= nbld; ++bld) {
                Color_t p1_color = 1, p2_color = 1;
                // Horizontal lines
                // Panel 2 has dashed mid-plane
                float x1 = xsign * (half_shift * 0.5 + dsk - 1);
                float x2 = xsign * (half_shift * 0.5 + dsk);
                int sign = ysign;
                float y1 = ysign * (half_shift * 0.5 - 0.5 + bld + sign * 0.5);
                //float yp1_mid = ysign * (half_shift*0.5 - 0.5 + bld + sign*0.25);
                float y2 = ysign * (half_shift * 0.5 - 0.5 + bld);
                float yp2_mid = ysign * (half_shift * 0.5 - 0.5 + bld - sign * 0.25);
                float y3 = ysign * (half_shift * 0.5 - 0.5 + bld - sign * 0.5);
                PixelROCMapHelper::draw_line(x1, x2, y1, y1, 1, 1, p1_color);
                //PixelROCMapHelper::draw_line(x1, x2, yp1_mid, yp1_mid, 1, 3);
                PixelROCMapHelper::draw_line(x1, x2, y2, y2, 1, 1, p1_color);
                PixelROCMapHelper::draw_line(x1, x2, yp2_mid, yp2_mid, 1, 2);
                PixelROCMapHelper::draw_line(x1, x2, y3, y3, 1, 1, p2_color);
                // Vertical lines
                float x = xsign * (half_shift * 0.5 + dsk - 1);
                PixelROCMapHelper::draw_line(x, x, y1, y2, 1, 1, p1_color);
                PixelROCMapHelper::draw_line(x, x, y2, y3, 1, 1, p2_color);
                if (ring == 2) {
                  //PixelROCMapHelper::draw_line(x,  x,  y2,  y3, 1, 1, p1_color);
                  x = xsign * (half_shift * 0.5 + dsk);
                  PixelROCMapHelper::draw_line(x, x, y1, y2, 1, 1, p1_color);
                  PixelROCMapHelper::draw_line(x, x, y2, y3, 1, 1, p2_color);
                }
                // Make a BOX around ROC 0
                x1 = xsign * (half_shift * 0.5 + dsk - 1 / 8.);
                x2 = xsign * (half_shift * 0.5 + dsk);
                float y1_p1 = ysign * (half_shift * 0.5 - 0.5 + bld + sign * 0.25);
                float y2_p1 = ysign * (half_shift * 0.5 - 0.5 + bld + sign * 0.25 + xsign * ysign * 0.25);
                PixelROCMapHelper::draw_line(x1, x2, y1_p1, y1_p1, 1);
                //PixelROCMapHelper::draw_line(x1, x2, y2_p1, y2_p1, 1);
                PixelROCMapHelper::draw_line(x1, x1, y1_p1, y2_p1, 1);
                //PixelROCMapHelper::draw_line(x2, x2, y1_p1, y2_p1, 1);
                float y1_p2 = ysign * (half_shift * 0.5 - 0.5 + bld - sign * 0.25);
                float y2_p2 = ysign * (half_shift * 0.5 - 0.5 + bld - sign * 0.25 - xsign * ysign * 0.25);
                PixelROCMapHelper::draw_line(x1, x2, y1_p2, y1_p2, 1);
                //PixelROCMapHelper::draw_line(x1, x2, y2_p2, y2_p2, 1);
                PixelROCMapHelper::draw_line(x1, x1, y1_p2, y2_p2, 1);
                //PixelROCMapHelper::draw_line(x2, x2, y1_p2, y2_p2, 1);
              }
            }
          }
        }
    }
    // Special shifted "rebin" for Phase 0
    // Y axis should always have at least half-roc granularity because
    // there are half-ROC size shifts implemented in the coordinates
    // To remove this and show full ROC granularity
    // We merge bin contents in each pair of bins corresponding to one ROC
    // TODO: make sure this works for Profiles
    if (phase == 0 && h->GetNbinsY() == 250 && h->GetNbinsX() == 80) {
      int nentries = h->GetEntries();
      for (int binx = 1; binx <= 80; ++binx) {
        double sum = 0;
        for (int biny = 1; biny <= 250; ++biny) {
          bool odd_nrocy = (binx - 1 < 40) != (((binx - 1) / 4) % 2);
          if (biny % 2 == odd_nrocy)
            sum += h->GetBinContent(binx, biny);
          else {
            sum += h->GetBinContent(binx, biny);
            if (sum) {
              h->SetBinContent(binx, biny, sum);
              h->SetBinContent(binx, biny - 1, sum);
            }
            sum = 0;
          }
        }
      }
      h->SetEntries(nentries);
    }
  }
}

/*--------------------------------------------------------------------*/
void Phase1PixelROCMaps::drawBarrelMaps(TCanvas& canvas, const std::string& text)
/*--------------------------------------------------------------------*/
{
  canvas.cd();
  canvas.Modified();

  auto topPad = new TPad("pad1", "upper pad", 0.005, 0.96, 0.995, 0.995);
  topPad->Draw();
  topPad->cd();
  auto ltx = TLatex();
  ltx.SetTextFont(62);

  std::size_t found = text.find("Delta");
  if (found != std::string::npos) {
    ltx.SetTextSize(0.7);
  } else {
    ltx.SetTextSize(1.);
  }
  ltx.DrawLatexNDC(0.02, 0.3, text.c_str());

  canvas.cd();
  auto bottomPad = new TPad("pad2", "lower pad", 0.005, 0.005, 0.995, 0.955);
  bottomPad->Draw();
  bottomPad->cd();
  bottomPad->Divide(2, 2);
  for (unsigned int lay = 1; lay <= n_layers; lay++) {
    PixelROCMapHelper::dress_plot(bottomPad, h_bpix_maps[lay - 1].get(), lay, 0, 1, found == std::string::npos);
  }
}

/*--------------------------------------------------------------------*/
void Phase1PixelROCMaps::drawForwardMaps(TCanvas& canvas, const std::string& text)
/*--------------------------------------------------------------------*/
{
  canvas.cd();
  canvas.Modified();

  auto topPad = new TPad("pad1", "upper pad", 0.005, 0.94, 0.995, 0.995);
  topPad->Draw();
  topPad->cd();
  auto ltx = TLatex();
  ltx.SetTextFont(62);

  std::size_t found = text.find("Delta");
  if (found != std::string::npos) {
    ltx.SetTextSize(0.7);
  } else {
    ltx.SetTextSize(1.);
  }
  ltx.DrawLatexNDC(0.02, 0.3, text.c_str());

  canvas.cd();
  auto bottomPad = new TPad("pad2", "lower pad", 0.005, 0.005, 0.995, 0.935);
  bottomPad->Draw();
  bottomPad->cd();
  bottomPad->Divide(2, 1);
  for (unsigned int ring = 1; ring <= n_rings; ring++) {
    PixelROCMapHelper::dress_plot(bottomPad, h_fpix_maps[ring - 1].get(), 0, ring, 1, found == std::string::npos);
  }
}

/*--------------------------------------------------------------------*/
void Phase1PixelROCMaps::drawMaps(TCanvas& canvas, const std::string& text)
/*--------------------------------------------------------------------*/
{
  canvas.cd();
  canvas.Modified();

  auto topPad = new TPad("pad1", "upper pad", 0.005, 0.97, 0.995, 0.995);
  topPad->Draw();
  topPad->cd();
  auto ltx = TLatex();
  ltx.SetTextFont(62);

  std::size_t found = text.find("Delta");
  if (found != std::string::npos) {
    ltx.SetTextSize(0.7);
  } else {
    ltx.SetTextSize(1.);
  }
  ltx.DrawLatexNDC(0.02, 0.2, text.c_str());

  canvas.cd();
  auto bottomPad = new TPad("pad2", "lower pad", 0.005, 0.005, 0.995, 0.97);
  bottomPad->Draw();
  bottomPad->cd();
  bottomPad->Divide(2, 3);

  // dress the plots
  for (unsigned int lay = 1; lay <= n_layers; lay++) {
    PixelROCMapHelper::dress_plot(bottomPad, h_bpix_maps[lay - 1].get(), lay, 0, 1, found == std::string::npos);
  }

  bottomPad->Update();
  bottomPad->Modified();
  bottomPad->cd();

  for (unsigned int ring = 1; ring <= n_rings; ring++) {
    PixelROCMapHelper::dress_plot(
        bottomPad, h_fpix_maps[ring - 1].get(), 0, n_layers + ring, 1, found == std::string::npos);
  }
}
