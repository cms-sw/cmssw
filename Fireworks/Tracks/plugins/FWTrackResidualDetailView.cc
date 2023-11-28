
#include "TVector3.h"
#include "TH2.h"
#include "TLine.h"
#include "TLatex.h"
#include "TPaveText.h"
#include "TCanvas.h"
#include "TEveWindow.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Fireworks/Core/interface/FWDetailView.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Tracks/plugins/FWTrackResidualDetailView.h"

using reco::HitPattern;
using reco::Track;
using reco::TrackBase;
using reco::TrackResiduals;

const char* FWTrackResidualDetailView::m_det_tracker_str[6] = {"PXB", "PXF", "TIB", "TID", "TOB", "TEC"};

FWTrackResidualDetailView::FWTrackResidualDetailView()
    : m_ndet(0),
      m_nhits(0),
      m_resXFill(3007),
      m_resXCol(kGreen - 9),
      m_resYFill(3006),
      m_resYCol(kWhite),
      m_stereoFill(3004),
      m_stereoCol(kCyan - 9),
      m_invalidFill(3001),
      m_invalidCol(kRed) {}

FWTrackResidualDetailView::~FWTrackResidualDetailView() {}

void FWTrackResidualDetailView::prepareData(const FWModelId& id, const reco::Track* track) {
  auto const& residuals = track->residuals();

  const FWGeometry* geom = id.item()->getGeom();
  assert(geom != nullptr);

  const HitPattern& hitpat = track->hitPattern();
  m_nhits = hitpat.numberOfAllHits(reco::HitPattern::TRACK_HITS);
  hittype.reserve(m_nhits);
  stereo.reserve(m_nhits);
  subsubstruct.reserve(m_nhits);
  substruct.reserve(m_nhits);
  m_detector.reserve(m_nhits);
  res[0].reserve(m_nhits);
  res[1].reserve(m_nhits);

  for (int i = 0; i < m_nhits; ++i) {
    //printf("there are %d hits in the pattern, %d in the vector, this is %u\n",
    //        m_nhits, track->recHitsEnd() - track->recHitsBegin(), (*(track->recHitsBegin() + i))->geographicalId().rawId());
    uint32_t pattern = hitpat.getHitPattern(reco::HitPattern::TRACK_HITS, i);
    hittype.push_back(HitPattern::getHitType(pattern));
    stereo.push_back(HitPattern::getSide(pattern));
    subsubstruct.push_back(HitPattern::getSubSubStructure(pattern));
    substruct.push_back(HitPattern::getSubStructure(pattern));
    m_detector.push_back(HitPattern::getSubDetector(pattern));
    if ((*(track->recHitsBegin() + i))->isValid()) {
      res[0].push_back(
          getSignedResidual(geom, (*(track->recHitsBegin() + i))->geographicalId().rawId(), residuals.pullX(i)));
    } else {
      res[0].push_back(0);
    }
    res[1].push_back(residuals.pullY(i));
    // printf("%s, %i\n",m_det_tracker_str[substruct[i]-1],subsubstruct[i]);
  }

  m_det[0] = 0;
  for (int j = 0; j < m_nhits - 1;) {
    int k = j + 1;
    for (; k < m_nhits; k++) {
      if (substruct[j] == substruct[k] && subsubstruct[j] == subsubstruct[k]) {
        if (k == (m_nhits - 1))
          j = k;
      } else {
        m_ndet++;
        j = k;
        m_det[m_ndet] = j;
        break;
      }
    }
  }
  m_ndet++;
  m_det[m_ndet] = m_nhits;
  // printDebug();
}

void FWTrackResidualDetailView::build(const FWModelId& id, const reco::Track* track) {
  if (!track->extra().isAvailable()) {
    fwLog(fwlog::kError) << " no track extra information is available.\n";
    m_viewCanvas->cd();
    TLatex* latex = new TLatex();
    latex->SetTextSize(0.1);
    latex->DrawLatex(0.1, 0.5, "No track extra information");
    return;
  }
  prepareData(id, track);

  // draw histogram
  m_viewCanvas->cd();
  m_viewCanvas->SetHighLightColor(-1);
  TH2F* h_res = new TH2F("h_resx", "h_resx", 10, -5.5, 5.5, m_ndet, 0, m_ndet);
  TPad* padX = new TPad("pad1", "pad1", 0.2, 0., 0.8, 0.99);
  padX->SetBorderMode(0);
  padX->SetLeftMargin(0.2);
  padX->Draw();
  padX->cd();
  padX->SetFrameLineWidth(0);
  padX->Modified();
  h_res->SetDirectory(nullptr);
  h_res->SetStats(kFALSE);
  h_res->SetTitle("");
  h_res->SetXTitle("residual");
  h_res->GetXaxis()->SetTickLength(0);
  h_res->GetYaxis()->SetTickLength(0);
  h_res->GetXaxis()->SetNdivisions(20);
  h_res->GetYaxis()->SetLabelSize(0.06);
  h_res->Draw();
  padX->SetGridy();

  float larray[9] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5};
  float larray2[8];
  for (int l = 0; l < 8; l++) {
    float diff2 = (larray[l + 1] - larray[l]) / 2;
    larray2[l] = larray[l] + diff2;
    //  printf("(%.1f,%.1f),",larray[i],larray[i+1]);
  }

  int resi[2][64];
  for (int l = 0; l < m_nhits; l++) {
    for (int k = 0; k < 8; k++) {
      if (fabs(res[0][l]) == larray2[k])
        resi[0][l] = k;
      if (fabs(res[1][l]) == larray2[k])
        resi[1][l] = k;
    }
  }

  TLine* lines[17];
  for (int l = 0; l < 17; l++) {
    int ix = l % 9;
    int sign = 1;
    sign = (l > 8) ? -1 : 1;
    lines[l] = new TLine(sign * larray[ix], 0, sign * larray[ix], m_ndet);
    if (l != 9)
      lines[l]->SetLineStyle(3);
    padX->cd();
    lines[l]->Draw();
  }

  float width = 0.25;
  int filltype;
  Color_t color;
  Double_t box[4];
  padX->cd();

  for (int h = 0; h < 2; h++) {
    float height1 = 0;
    for (int j = 0; j < m_ndet; j++) {
      // take only X res and Y pixel residals
      if (strcmp(m_det_tracker_str[substruct[m_det[j]] - 1], "PXB") && h)
        continue;

      char det_str2[256];
      snprintf(det_str2, 255, "%s/%i", m_det_tracker_str[substruct[m_det[j]] - 1], subsubstruct[m_det[j]]);
      h_res->GetYaxis()->SetBinLabel(j + 1, det_str2);

      int diff = m_det[j + 1] - m_det[j];
      int k = 0;
      width = 1.0 / diff;

      for (int l = m_det[j]; l < (m_det[j] + diff); l++) {
        //      g->SetPoint(l,resx[l],j+0.5);
        //	printf("%i, %f %f %f\n",l,resx[l],sign*larray[resxi[l]],sign*larray[resxi[l]+1]);
        int sign = (res[h][l] < 0) ? -1 : 1;
        box[0] = (hittype[l] == 0) ? sign * larray[resi[h][l]] : -5.5;
        box[2] = (hittype[l] == 0) ? sign * larray[resi[h][l] + 1] : 5.5;
        box[1] = height1 + width * k;
        box[3] = height1 + width * (k + 1);

        if (stereo[l] == 1) {
          color = m_stereoCol;
          filltype = m_stereoFill;
        } else if (hittype[l] != 0) {
          color = m_invalidCol;
          filltype = m_invalidFill;
        } else {
          filltype = h ? m_resYFill : m_resXFill;
          color = h ? m_resYCol : m_resXCol;
        }

        drawCanvasBox(box, color, filltype, h < 1);
        k++;
      }
      height1 += 1;
    }
  }

  //  title
  const char* res_str = "residuals in Si detector local x-y coord.";
  TPaveText* pt = new TPaveText(0.0, 0.91, 1, 0.99, "blNDC");
  pt->SetBorderSize(0);
  pt->SetFillColor(kWhite);
  pt->AddText(res_str);
  pt->Draw();

  m_viewCanvas->cd();
  m_viewCanvas->SetEditable(kFALSE);

  setTextInfo(id, track);
}

double FWTrackResidualDetailView::getSignedResidual(const FWGeometry* geom, unsigned int id, double resX) {
  double local1[3] = {0, 0, 0};
  double local2[3] = {resX, 0, 0};
  double global1[3], global2[3];
  const TGeoMatrix* m = geom->getMatrix(id);
  assert(m != nullptr);
  m->LocalToMaster(local1, global1);
  m->LocalToMaster(local2, global2);
  TVector3 g1 = global1;
  TVector3 g2 = global2;
  if (g2.DeltaPhi(g1) > 0)
    return resX;
  else
    return -resX;
}

void FWTrackResidualDetailView::printDebug() {
  for (int i = 0; i < m_ndet; i++) {
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    std::cout << "idx " << i << " det[idx] " << m_det[i] << std::endl;
    std::cout << "m_det idx " << m_det[i] << std::endl;
    std::cout << "m_det_tracker_str idx " << substruct[m_det[i]] - 1 << std::endl;
    printf("m_det[idx] %i m_det_tracker_str %s substruct %i\n",
           m_det[i],
           m_det_tracker_str[substruct[m_det[i]] - 1],
           subsubstruct[m_det[i]]);
  }
}

void FWTrackResidualDetailView::setTextInfo(const FWModelId& /*id*/, const reco::Track* /*track*/) {
  m_infoCanvas->cd();

  char mytext[256];
  Double_t fontsize = 0.07;
  TLatex* latex = new TLatex();
  latex->SetTextSize(fontsize);
  latex->Draw();
  Double_t x0 = 0.02;
  Double_t y = 0.95;

  // summary
  int nvalid = 0;
  int npix = 0;
  int nstrip = 0;
  for (int i = 0; i < m_nhits; i++) {
    if (hittype[i] == 0)
      nvalid++;
    if (substruct[i] < 3)
      npix++;
    else
      nstrip++;
  }

  latex->SetTextSize(fontsize);
  Double_t boxH = 0.25 * fontsize;

  double yStep = 0.04;

  latex->DrawLatex(x0, y, "Residual:");
  y -= yStep;
  latex->DrawLatex(
      x0,
      y,
      "sgn(#hat{X}#bullet#hat{#phi}) #times #frac{X_{hit} - X_{traj}}{#sqrt{#sigma^{2}_{hit} + #sigma^{2}_{traj}}}");
  y -= 2.5 * yStep;
  snprintf(mytext, 255, "layers hit: %i", m_ndet);
  latex->DrawLatex(x0, y, mytext);
  y -= yStep;
  snprintf(mytext, 255, "valid Si hits: %i", nvalid);
  latex->DrawLatex(x0, y, mytext);
  y -= yStep;
  snprintf(mytext, 255, "total Si hits: %i", m_nhits);
  latex->DrawLatex(x0, y, mytext);
  y -= yStep;
  snprintf(mytext, 255, "valid Si pixel hits: %i", npix);
  latex->DrawLatex(x0, y, mytext);
  y -= yStep;
  snprintf(mytext, 255, "valid Si strip hits: %i", nstrip);
  latex->DrawLatex(x0, y, mytext);

  Double_t pos[4];
  pos[0] = 0.4;
  pos[2] = 0.55;

  y -= yStep * 2;
  latex->DrawLatex(x0, y, "X hit");
  pos[1] = y;
  pos[3] = pos[1] + boxH;
  drawCanvasBox(pos, m_resXCol, m_resXFill);

  y -= yStep;
  latex->DrawLatex(x0, y, "Y hit");
  pos[1] = y;
  pos[3] = pos[1] + boxH;
  drawCanvasBox(pos, m_resYCol, m_resYFill, false);

  y -= yStep;
  latex->DrawLatex(x0, y, "stereo hit");
  pos[1] = y;
  pos[3] = pos[1] + boxH;
  drawCanvasBox(pos, m_stereoCol, m_stereoFill);

  y -= yStep;
  latex->DrawLatex(x0, y, "invalid hit");
  pos[1] = y;
  pos[3] = pos[1] + boxH;
  drawCanvasBox(pos, m_invalidCol, m_invalidFill);
}

REGISTER_FWDETAILVIEW(FWTrackResidualDetailView, Residuals);
