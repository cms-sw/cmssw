std::vector<double> FindExtremes(TFile *file, int arm, int station, int plane);

void DrawPlaneHitmapWithBorder(TFile *file, int arm, int station, int plane);

TH2D FindAreaForEfficiencyMapping(TFile *file, int arm, int station);

TH2D ProduceEfficiencyMap(TFile *file, int arm, int station);

TH2D ProduceEfficiencyMap(TFile *file, int arm, int station) {

  bool wasBatch = true;
  if (!gROOT->IsBatch()) {
    wasBatch = false;
    gROOT->SetBatch(true);
  }
  TH2D *h2Area = new TH2D(FindAreaForEfficiencyMapping(file, arm, station));
  if (wasBatch) {
    gROOT->SetBatch(true);
  } else {
    gROOT->SetBatch(false);
  }

  TH2D *h2EfficiencyMap =
      (TH2D *)file->GetDirectory(Form("Arm%i_st%i_rp3", arm, station))
          ->Get(Form("h2TrackEfficiencyMap_arm%i_st%i_rp3", arm, station));
  TH2D *h2TrackHitDistribution =
      (TH2D *)file->GetDirectory(Form("Arm%i_st%i_rp3", arm, station))
          ->Get(Form("h2TrackHitDistribution_arm%i_st%i_rp3", arm, station));
  TH2D *h2EfficiencyMap_original = new TH2D(*h2EfficiencyMap);
  int nPixelsWithLowStat = 0;

  // Set to 0 all the cells where less than 3 planes overlap
  for (int xbin = 1; xbin < h2Area->GetNbinsX(); xbin++) {
    for (int ybin = 1; ybin < h2Area->GetNbinsY(); ybin++) {
      if (h2Area->GetBinContent(xbin, ybin) <= 3) {
        h2EfficiencyMap->SetBinContent(xbin, ybin, 0.);
      } else {
        if (h2TrackHitDistribution->GetBinContent(xbin, ybin) < 10) {
          nPixelsWithLowStat++;
          h2EfficiencyMap->SetBinContent(xbin, ybin, 2.);
        }
      }
    }
  }

  // AVERAGING PART
  // Set the efficiency of pixels with low statistics to the average of the ones
  // around them
  int cycles = 0;
  int nPixelsAveraged = 0;
  bool sameCycle = false;
  int searchRadius = 2;
  while (nPixelsWithLowStat > 0) {
    cout << nPixelsWithLowStat << " pixel efficiencies to compute" << endl;
    cycles++;
    int nPixelsWithLowStat_old = nPixelsWithLowStat;
    for (int xbin = 1; xbin < h2Area->GetNbinsX(); xbin++) {
      for (int ybin = 1; ybin < h2Area->GetNbinsY(); ybin++) {
        // only within the sensor, for each pixel with low statistics
        if (h2EfficiencyMap->GetBinContent(xbin, ybin) == 2) {
          double avg = h2EfficiencyMap_original->GetBinContent(xbin, ybin);
          int pixelsAroundPassingCondition = 1;
          // double avg = 0;
          // int pixelsAroundPassingCondition = 0;

          // check the adjacent pixels with enough statistics or whose
          // efficiency has been already averaged
          for (int rowShift = -searchRadius; rowShift < searchRadius+1; rowShift++) {
            for (int colShift = -searchRadius; colShift < searchRadius+1; colShift++) {
              if (rowShift == 0 && colShift == 0)
                continue;
              double pixelAroundEfficiency = h2EfficiencyMap_original->GetBinContent(
                  xbin + colShift, ybin + rowShift);
              // require the pixel around not to have low statistics and to be
              // within the area with at least 4 planes overlapping.
              // If its efficiency has already been averaged, it is used
              // if (h2Area->GetBinContent(xbin + colShift, ybin + rowShift) > 3
              // &&
              //     h2TrackHitDistribution->GetBinContent(xbin + colShift,
              //                                           ybin + rowShift) > 3)
              //                                           {
              //   avg += pixelAroundEfficiency;
              //   pixelsAroundPassingCondition++;
              //   continue;
              // }
              if (h2Area->GetBinContent(xbin + colShift, ybin + rowShift) > 3 &&
                  h2EfficiencyMap->GetBinContent(xbin + colShift,
                                                 ybin + rowShift) != 2) {
                avg += pixelAroundEfficiency;
                pixelsAroundPassingCondition++;
              }
            }
          }
          // if there is at least 2, compute the average and give it to the
          // pixel under analysis
          if (pixelsAroundPassingCondition > 1) {
            h2EfficiencyMap->SetBinContent(xbin, ybin,
                                           avg / pixelsAroundPassingCondition);
            nPixelsAveraged++;
            nPixelsWithLowStat--;
          }
          if (sameCycle && pixelsAroundPassingCondition > 0) {
            h2EfficiencyMap->SetBinContent(xbin, ybin,
                                           avg / pixelsAroundPassingCondition);
            nPixelsAveraged++;
            nPixelsWithLowStat--;
          }
        } // ...for each pixel with low statistics
      }
    }

    sameCycle = false;
    if (nPixelsWithLowStat == nPixelsWithLowStat_old) {
      sameCycle = true;
      cout << "Repeating..." << endl;
    }
  } // ...while there are pixels with low statistics
  cout << endl;
  cout << "Averaging completed in " << cycles << " cycles\n"
       << nPixelsAveraged << " cell efficiencies were computed by averaging"
       << endl;
  delete h2Area;
  return *h2EfficiencyMap;
}

// Produces a map of the number or planes overlapping in that cell
TH2D FindAreaForEfficiencyMapping(TFile *file, int arm, int station) {
  using namespace std;
  std::map<int, vector<double>> planeExtremes;
  TCanvas *c = new TCanvas("c", "c");
  TH2D *h2TrackHitDistribution;
  if (station == 2) {
    h2TrackHitDistribution =
        (TH2D *)file->GetDirectory(Form("Arm%i_st%i_rp3", arm, station))
            ->Get(Form("h2TrackHitDistribution_arm%i_st%i_rp3", arm, station));
  } else {
    h2TrackHitDistribution =
        (TH2D *)file->GetDirectory(Form("Arm%i_st%i_rp3", arm, station))
            ->Get(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp3", arm,
                       station));
  }
  h2TrackHitDistribution->Draw("colz");
  TH2D *h2Area = new TH2D(*h2TrackHitDistribution);
  h2Area->Reset();

  for (int plane = 0; plane < 6; plane++) {
    planeExtremes[plane] = FindExtremes(file, arm, station, plane);
    TF1 *lowerEdge = new TF1("lowerEdge", "pol0", planeExtremes[plane].at(0),
                             planeExtremes[plane].at(2));
    lowerEdge->SetParameters(planeExtremes[plane].at(1), 0.);
    TF1 *upperEdge = new TF1("upperEdge", "pol0", planeExtremes[plane].at(0),
                             planeExtremes[plane].at(2));
    upperEdge->SetParameters(planeExtremes[plane].at(3), 0.);

    TLine *leftEdge =
        new TLine(planeExtremes[plane].at(0), planeExtremes[plane].at(1),
                  planeExtremes[plane].at(0), planeExtremes[plane].at(3));
    TLine *rightEdge =
        new TLine(planeExtremes[plane].at(2), planeExtremes[plane].at(1),
                  planeExtremes[plane].at(2), planeExtremes[plane].at(3));

    lowerEdge->SetLineWidth(3);
    lowerEdge->SetLineColor(kBlack);
    upperEdge->SetLineWidth(3);
    upperEdge->SetLineColor(kBlack);
    leftEdge->SetLineWidth(3);
    leftEdge->SetLineColor(kBlack);
    rightEdge->SetLineWidth(3);
    rightEdge->SetLineColor(kBlack);
    lowerEdge->DrawCopy("same");
    upperEdge->DrawCopy("same");
    leftEdge->Draw("same");
    rightEdge->Draw("same");
  }
  for (int xbin = 1; xbin < h2Area->GetNbinsX(); xbin++) {
    for (int ybin = 1; ybin < h2Area->GetNbinsY(); ybin++) {
      int isWithinPlane = 0;
      for (int plane = 0; plane < 6; plane++) {
        double pixelLowXEdge = h2Area->GetXaxis()->GetBinLowEdge(xbin);
        double pixelHighXEdge = h2Area->GetXaxis()->GetBinUpEdge(xbin);
        double pixelLowYEdge = h2Area->GetYaxis()->GetBinLowEdge(ybin);
        double pixelHighYEdge = h2Area->GetYaxis()->GetBinUpEdge(ybin);
        // Counts how many times the pixel cell is within the sensor edges of
        // plane
        if (pixelLowXEdge >= planeExtremes[plane].at(0) &&
            pixelHighXEdge <= planeExtremes[plane].at(2) &&
            pixelLowYEdge >= planeExtremes[plane].at(1) &&
            pixelHighYEdge <= planeExtremes[plane].at(3)) {
          isWithinPlane++;
        }
      }
      // Saves it in a map
      if (station == 0) {
        double rotatedMapXbinCenter =
            h2TrackHitDistribution->GetXaxis()->GetBinCenter(xbin);
        double rotatedMapYbinCenter =
            h2TrackHitDistribution->GetYaxis()->GetBinCenter(ybin);
        double xbinCenter =
            rotatedMapXbinCenter * TMath::Cos((8. / 180.) * TMath::Pi()) -
            rotatedMapYbinCenter * TMath::Sin((8. / 180.) * TMath::Pi());
        double ybinCenter =
            rotatedMapXbinCenter * TMath::Sin((8. / 180.) * TMath::Pi()) +
            rotatedMapYbinCenter * TMath::Cos((8. / 180.) * TMath::Pi());
        int nonRotatedXbin =
            h2TrackHitDistribution->GetXaxis()->FindBin(xbinCenter);
        int nonRotatedYbin =
            h2TrackHitDistribution->GetYaxis()->FindBin(ybinCenter);

        h2Area->SetBinContent(nonRotatedXbin, nonRotatedYbin, isWithinPlane);
      } else {
        h2Area->SetBinContent(xbin, ybin, isWithinPlane);
      }
    }
  }
  // removes empty bins created by the rotation. They are given by the fact that
  // the overlap is computed on maps that are parallel to the axes
  if (station == 0) {
    for (int xbin = 1; xbin < h2Area->GetNbinsX(); xbin++) {
      for (int ybin = 1; ybin < h2Area->GetNbinsY(); ybin++) {
        if (h2Area->GetBinContent(xbin, ybin) == 0) {
          double avg = (h2Area->GetBinContent(xbin - 1, ybin) +
                        h2Area->GetBinContent(xbin - 1, ybin) +
                        h2Area->GetBinContent(xbin, ybin + 1) +
                        h2Area->GetBinContent(xbin, ybin + 1) +
                        h2Area->GetBinContent(xbin - 1, ybin - 1) +
                        h2Area->GetBinContent(xbin - 1, ybin + 1) +
                        h2Area->GetBinContent(xbin + 1, ybin - 1) +
                        h2Area->GetBinContent(xbin + 1, ybin + 1)) /
                       8.;
          if (avg > 2) {
            h2Area->SetBinContent(xbin, ybin, ceil(avg));
          } else {
            if (avg > 0.8) {
              h2Area->SetBinContent(xbin, ybin, round(avg));
            } else {
              h2Area->SetBinContent(xbin, ybin, floor(avg));
            }
          }
        }
      }
    }
  }
  TCanvas *c2 = new TCanvas("c2", "c2");
  h2Area->SetContour(7);
  h2Area->Draw("colz");
  return *h2Area;
}

// Does what it says :-)
void DrawPlaneHitmapWithBorder(TFile *file, int arm, int station, int plane) {
  using namespace std;
  vector<double> extremes = FindExtremes(file, arm, station, plane);
  TH2D *h2Hitmap;
  if (station == 2) {
    h2Hitmap = (TH2D *)file
                   ->GetDirectory(Form("Arm%i_st%i_rp3/Arm%i_st%i_rp3_pl%i",
                                       arm, station, arm, station, plane))
                   ->Get(Form("h2ModuleHitMap_arm%i_st%i_rp3_pl%i", arm,
                              station, plane));
  } else {
    h2Hitmap = (TH2D *)file
                   ->GetDirectory(Form("Arm%i_st%i_rp3/Arm%i_st%i_rp3_pl%i",
                                       arm, station, arm, station, plane))
                   ->Get(Form("h2ModuleHitMap_rotated_arm%i_st%i_rp3_pl%i", arm,
                              station, plane));
  }
  TCanvas *c = new TCanvas("c", "c");
  c->SetLogz();
  h2Hitmap->Draw("colz");

  TF1 *lowerEdge = new TF1("lowerEdge", "pol0", extremes.at(0), extremes.at(2));
  lowerEdge->SetParameters(extremes.at(1), 0.);
  TF1 *upperEdge = new TF1("upperEdge", "pol0", extremes.at(0), extremes.at(2));
  upperEdge->SetParameters(extremes.at(3), 0.);

  TLine *leftEdge =
      new TLine(extremes.at(0), extremes.at(1), extremes.at(0), extremes.at(3));
  TLine *rightEdge =
      new TLine(extremes.at(2), extremes.at(1), extremes.at(2), extremes.at(3));

  lowerEdge->SetLineWidth(3);
  lowerEdge->SetLineColor(kBlack);
  upperEdge->SetLineWidth(3);
  upperEdge->SetLineColor(kBlack);
  leftEdge->SetLineWidth(3);
  leftEdge->SetLineColor(kBlack);
  rightEdge->SetLineWidth(3);
  rightEdge->SetLineColor(kBlack);
  lowerEdge->Draw("same");
  upperEdge->Draw("same");
  leftEdge->Draw("same");
  rightEdge->Draw("same");

  return;
}

// Compute the minimum and maximum filled bins on both axes to establish the
// sensor position. The sensor needs to almost parallel to the axes, so
// stations 0 need rotation
std::vector<double> FindExtremes(TFile *file, int arm, int station, int plane) {
  using namespace std;
  std::vector<double> extremes;
  if ((arm != 0 && arm != 1) || (station != 0 && station != 2) || (plane < 0) ||
      (plane > 5)) {
    cout << "ERROR in FindExtremes" << endl;
    return extremes;
  }
  TH2D *h2Hitmap;
  if (station == 2) {
    h2Hitmap = (TH2D *)file
                   ->GetDirectory(Form("Arm%i_st%i_rp3/Arm%i_st%i_rp3_pl%i",
                                       arm, station, arm, station, plane))
                   ->Get(Form("h2ModuleHitMap_arm%i_st%i_rp3_pl%i", arm,
                              station, plane));
  } else {
    h2Hitmap = (TH2D *)file
                   ->GetDirectory(Form("Arm%i_st%i_rp3/Arm%i_st%i_rp3_pl%i",
                                       arm, station, arm, station, plane))
                   ->Get(Form("h2ModuleHitMap_rotated_arm%i_st%i_rp3_pl%i", arm,
                              station, plane));
  }
  int yBinLow = 999999;
  int yBinHigh = 0;
  int xBinLow = 999999;
  int xBinHigh = 0;
  int sensorVerticalWidthBin = 0;
  int sensorHorizontalWidthBin = 0;
  // find vertical limits
  for (int xbin = 0; xbin <= h2Hitmap->GetNbinsX(); xbin++) {
    int yBinFirst = 0;
    bool firstFound = false;
    int yBinLast = 0;
    for (int ybin = 0; ybin <= h2Hitmap->GetNbinsY(); ybin++) {
      if (h2Hitmap->GetBinContent(xbin, ybin) > 0) {
        if (!firstFound) {
          firstFound = true;
          yBinFirst = ybin;
        }
        yBinLast = ybin;
      }
    }
    if (yBinFirst != 0 && yBinFirst < yBinLow)
      yBinLow = yBinFirst;
    if (yBinLast > yBinHigh)
      yBinHigh = yBinLast;
  }
  // find horizontal limits
  for (int ybin = 0; ybin <= h2Hitmap->GetNbinsY(); ybin++) {
    int xBinFirst = 0;
    bool firstFound = false;
    int xBinLast = 0;
    for (int xbin = 0; xbin <= h2Hitmap->GetNbinsX(); xbin++) {
      if (h2Hitmap->GetBinContent(xbin, ybin) > 0) {
        if (!firstFound) {
          firstFound = true;
          xBinFirst = xbin;
        }
        xBinLast = xbin;
      }
    }
    if (xBinFirst != 0 && xBinFirst < xBinLow)
      xBinLow = xBinFirst;
    if (xBinLast > xBinHigh)
      xBinHigh = xBinLast;
  }

  sensorVerticalWidthBin = yBinHigh - yBinLow + 1;
  double yLow = h2Hitmap->GetYaxis()->GetBinLowEdge(yBinLow);
  double yHigh = h2Hitmap->GetYaxis()->GetBinUpEdge(yBinHigh);
  sensorHorizontalWidthBin = xBinHigh - xBinLow + 1;
  double xLow = h2Hitmap->GetXaxis()->GetBinLowEdge(xBinLow);
  double xHigh = h2Hitmap->GetXaxis()->GetBinUpEdge(xBinHigh);

  // cout << "***FINAL RESULTS***" << endl;
  // cout << "HORIZONTAL:\t" << sensorHorizontalWidthBin << "\t" << xBinLow <<
  // "\t"
  //      << xBinHigh << "\t" << xLow << "\t" << xHigh << endl;
  // cout << "VERTICAL:\t" << sensorVerticalWidthBin << "\t" << yBinLow << "\t"
  //      << yBinHigh << "\t" << yLow << "\t" << yHigh << endl;

  extremes.push_back(xLow);
  extremes.push_back(yLow);
  extremes.push_back(xHigh);
  extremes.push_back(yHigh);
  return extremes;
}
