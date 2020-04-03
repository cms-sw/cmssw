/*
 * =====================================================================================
 *
 *       Filename:  EmuEventDisplay.cc
 *
 *    Description:  draw event display based on encoded histogram
 *
 *        Version:  1.0
 *        Created:  12/12/2009 08:57:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "CSCRenderPlugin_EmuEventDisplay.h"

/**
 * @brief  Constructor
 */
EmuEventDisplay::EmuEventDisplay() {

    histo_zr = new TH2F("h1", "Events in #Zeta-R projection", 22, -11.0, 11.0, 14, -7.0, 7.0);
    histo_zr->GetXaxis()->SetTitle("#Zeta * 1000");
    histo_zr->GetXaxis()->SetTitleOffset(1.2);
    histo_zr->GetXaxis()->CenterTitle(true);
    histo_zr->GetXaxis()->SetLabelSize(0.02);
    histo_zr->GetXaxis()->SetTicks("+-");
    histo_zr->GetYaxis()->SetTitle("R * 1000");
    histo_zr->GetYaxis()->SetTitleOffset(-1.2);
    histo_zr->GetYaxis()->SetTicks("+-");
    histo_zr->GetYaxis()->CenterTitle(true);
    histo_zr->GetYaxis()->SetLabelSize(0.02);
    histo_zr->SetStats(kFALSE);

    histo_zphi = new TH2F("h2", "Events in #Zeta-#phi projection", 22, -11.0, 11.0, (int) (PI * 2 + 0.2), -0.2, PI * 2);
    histo_zphi->GetXaxis()->SetTitle("#Zeta * 1000");
    histo_zphi->GetXaxis()->SetTitleOffset(1.2);
    histo_zphi->GetXaxis()->CenterTitle(true);
    histo_zphi->GetXaxis()->SetLabelSize(0.02);
    histo_zphi->GetXaxis()->SetTicks("+-");
    histo_zphi->GetYaxis()->SetTitle("#phi");
    histo_zphi->GetYaxis()->SetTitleOffset(-1.2);
    histo_zphi->GetYaxis()->SetTicks("+-");
    histo_zphi->GetYaxis()->CenterTitle(true);
    histo_zphi->GetYaxis()->SetLabelSize(0.02);
    histo_zphi->SetStats(kFALSE);

    histo_xy = new TH2F("h3", "Events in X-Y projection", 19, -9.33, 9.33, 14, -7.0, 7.0);
    histo_xy->GetXaxis()->SetTitle("X * 1000");
    histo_xy->GetXaxis()->SetTitleOffset(1.2);
    histo_xy->GetXaxis()->CenterTitle(true);
    histo_xy->GetXaxis()->SetLabelSize(0.02);
    histo_xy->GetXaxis()->SetTicks("+-");
    histo_xy->GetYaxis()->SetTitle("Y * 1000");
    histo_xy->GetYaxis()->SetTitleOffset(-1.2);
    histo_xy->GetYaxis()->SetTicks("+-");
    histo_xy->GetYaxis()->CenterTitle(true);
    histo_xy->GetYaxis()->SetLabelSize(0.02);
    histo_xy->SetStats(kFALSE);

    EmuChamberPart chPart;
    while (chPart.next(detector)) {

        double rd = (detector.NumberOfStrips(chPart.station, chPart.ring) * detector.stripDPhiRad(chPart.station, chPart.ring)) / 2;

        double z1 = detector.Z_mm(chPart.side, chPart.station, chPart.ring, chPart.chamber, 1);
        double z2 = detector.Z_mm(chPart.side, chPart.station, chPart.ring, chPart.chamber, N_LAYERS);

        { // Z-R plane elements

            double r1 = detector.R_mm(chPart.side, chPart.station, chPart.ring, chPart.part, 1, 1, 1);
            double r2 = detector.R_mm(chPart.side, chPart.station, chPart.ring, chPart.part, N_LAYERS, detector.NumberOfHalfstrips(chPart.station, chPart.ring, chPart.part), detector.NumberOfWiregroups(chPart.station, chPart.ring));

            // Switch lower chambers to minus projection
            if (chPart.chamber > (detector.NumberOfChambers(chPart.station, chPart.ring) / 2)) {
                r1 = -r1;
                r2 = -r2;
            }

            EmuChamber* ch = new EmuChamber();
            ch->setX(0, z1)->setX(1, z1)->setX(2, z2)->setX(3, z2);
            ch->setY(0, r1)->setY(1, r2)->setY(2, r2)->setY(3, r1);

            zrChambers.push_back(ch);

        }

        { // Z-Phi plane elements

            double r = detector.PhiRadChamberCenter(chPart.station, chPart.ring, chPart.chamber);

            EmuChamber* ch = new EmuChamber(true, false);
            ch->setX(0, z1)->setX(1, z1)->setX(2, z2)->setX(3, z2);
            ch->setY(0, r + rd)->setY(1, r - rd)->setY(2, r - rd)->setY(3, r + rd);

            zpChambers.push_back(ch);

        }

        { // X-Y plane elements

            EmuChamber* ch = new EmuChamber();

            double x[4], y[4];
            detector.chamberBoundsXY(chPart.side, chPart.station, chPart.ring, chPart.chamber, chPart.part, x, y);
            ch->setX(0, x[0])->setX(1, x[1])->setX(2, x[2])->setX(3, x[3]);
            ch->setY(0, y[0])->setY(1, y[1])->setY(2, y[2])->setY(3, y[3]);

            xyChambers.push_back(ch);

        }

    }

    {
        TLine *line = new TLine(-11.0, 0, 11.0, 0);
        line->SetLineColor(kRed - 10);
        line->SetLineStyle(kDashed);
        line->SetLineWidth(1);
        zrLines.push_back(line);
    }

    {
        TLine *line = new TLine(0, -7.0, 0, 7.0);
        line->SetLineColor(kRed - 10);
        line->SetLineStyle(kDashed);
        line->SetLineWidth(1);
        zrLines.push_back(line);
    }

}

/**
 * @brief  Destructor
 */
EmuEventDisplay::~EmuEventDisplay() {

    delete histo_zr;
    delete histo_zphi;
    delete histo_xy;

    deleteItems(zrChambers);
    deleteItems(zpChambers);
    deleteItems(xyChambers);

    deleteItems(zrHits);
    deleteItems(zpHits);
    deleteItems(xyHits);

    deleteItems(zrLines);

}

/**
 * @brief  Draw ZPhi event display map
 * @param  data Data histogram
 * @return
 */
void EmuEventDisplay::drawEventDisplay_ZR(TH2* data) {

    deleteItems(zrHits);

    histo_zr->Draw();
    drawItems(zrChambers);
    drawItems(zrLines);

    EmuChamberPart chPart;
    while (chPart.next(detector)) {

        EmuRawHits hits;
        if (readHistogramHits(data, chPart, hits)) {

            std::vector<EmuLayerHit> layerHits;
            if (hits.getLayerHits(chPart, layerHits)) {
                for (unsigned int i = 0; i < layerHits.size(); i++) {

                    EmuLayerHit layerHit = layerHits[i];
                    double z = detector.Z_mm(chPart.side, chPart.station, chPart.ring, chPart.chamber, layerHit.layer + 1);
                    double r = detector.R_mm(chPart.side, chPart.station, chPart.ring, layerHit.part, layerHit.layer + 1, layerHit.hs, layerHit.wg);

                    // Lower chamber numbers go down
                    if (chPart.chamber > (detector.NumberOfChambers(chPart.station, chPart.ring) / 2)) {
                        r = -r;
                    }

                    if (zrChambers.at(chPart.partIndex)->hitInBounds(z, r)) {
                        EmuHit* hit = new EmuHit(z, r);
                        zrHits.push_back(hit);
                    }

                }
            }

        }

    }

    drawItems(zrHits);

}

void EmuEventDisplay::drawEventDisplay_ZPhi(TH2* data) {

    deleteItems(zpHits);

    histo_zphi->Draw();
    drawItems(zpChambers);

    EmuChamberPart chPart;
    while (chPart.next(detector)) {

        double r = detector.PhiRadChamberCenter(chPart.station, chPart.ring, chPart.chamber);

        for (int strip = 1; strip <= detector.NumberOfStrips(chPart.station, chPart.ring); strip++) {
            int layerBitset = (int) data->GetBinContent(chPart.chIndex, strip);
            if (layerBitset > 0) {
                for (int layer = 0; layer < N_LAYERS; layer++) {
                    if (layerBitset & (1 << layer)) {
                        double z = detector.Z_mm(chPart.side, chPart.station, chPart.ring, chPart.chamber, layer + 1);
                        double pd = detector.LocalPhiRadStripToChamberCenter(chPart.side, chPart.station, chPart.ring, layer + 1, strip);

                        if (zpChambers.at(chPart.partIndex)->hitInBounds(z, r + pd)) {
                            EmuHit* hit = new EmuHit(z, r + pd, true, false);
                            zpHits.push_back(hit);
                        }
                    }
                }
            }
        }

    }

    drawItems(zpHits);

}

void EmuEventDisplay::drawEventDisplay_XY(TH2* data) {

    deleteItems(xyHits);

    histo_xy->Draw();
    drawItems(xyChambers);

    EmuChamberPart chPart;
    while (chPart.next(detector)) {

        EmuRawHits hits;
        if (readHistogramHits(data, chPart, hits)) {

            std::vector<EmuLayerHit> layerHits;
            if (hits.getLayerHits(chPart, layerHits)) {
                for (unsigned int i = 0; i < layerHits.size(); i++) {

                    EmuLayerHit layerHit = layerHits[i];
                    double cx = detector.X_mm(chPart.side, chPart.station, chPart.ring, layerHit.part, chPart.chamber, layerHit.layer + 1, layerHit.hs, layerHit.wg);
                    double cy = detector.Y_mm(chPart.side, chPart.station, chPart.ring, layerHit.part, chPart.chamber, layerHit.layer + 1, layerHit.hs, layerHit.wg);
                    if (xyChambers.at(chPart.partIndex)->hitInBounds(cx, cy)) {
                        EmuHit* hit = new EmuHit(cx, cy);
                        xyHits.push_back(hit);
                    }
                }
            }
        }

    }

    drawItems(xyHits);

}

bool EmuEventDisplay::readHistogramHits(TH2* data, EmuChamberPart& chPart, EmuRawHits& hits) {
    unsigned int HALF_STRIP_START = 160;
    unsigned int IPART_SECOND = 2;
    int IPART_SECOND_LENGTH = 16;
    bool is_wgs = false, is_hss = false;

    // Reading wiregroups
    for (int wg = 1; wg <= detector.NumberOfWiregroups(chPart.station, chPart.ring); wg++) {
        int wgBitset = (int) data->GetBinContent(chPart.chIndex, wg);
        if (wgBitset > 0) {
            for (int layer = 0; layer < N_LAYERS; layer++) {
                if (wgBitset & (1 << layer)) {
                    hits.addWgHit(layer, wg);
                    is_wgs = true;
                }
            }
        }
    }

    // Move strips forward if it is a second "a" part
    unsigned int prevPartStrips = 0;
    if (chPart.ipart == IPART_SECOND) {
        prevPartStrips = detector.NumberOfHalfstrips(chPart.station, chPart.ring, chPart.part);
    }

    // Reading halfstrips
    for (int hs = 1; hs <= detector.NumberOfHalfstrips(chPart.station, chPart.ring, chPart.part); hs++) {
        int hsBitset = (int) data->GetBinContent(chPart.chIndex, HALF_STRIP_START + prevPartStrips + hs);
        if (hsBitset > 0) {
            for (int layer = 0; layer < N_LAYERS; layer++) {
                if (hsBitset & (1 << layer)) {

                    hits.addHsHit(layer, hs);

                    // Exceptional case for second "a" part
                    // adding hits for hs = hs + 16 and hs = hs + 32
                    if (chPart.ipart == IPART_SECOND && hs <= IPART_SECOND_LENGTH) {
                        hits.addHsHit(layer, hs + 16);
                        hits.addHsHit(layer, hs + 32);
                    }

                    is_hss = true;
                }
            }
        }
    }

    return (is_wgs && is_hss);

}
