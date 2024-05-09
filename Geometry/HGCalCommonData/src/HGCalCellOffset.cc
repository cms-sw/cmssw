#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalCellOffset.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include <vector>
#include <iostream>

//#define EDM_ML_DEBUG

HGCalCellOffset::HGCalCellOffset(
    double waferSize, int32_t nFine, int32_t nCoarse, double guardRingOffset_, double mouseBiteCut_) {
  ncell_[0] = nFine;
  ncell_[1] = nCoarse;
  hgcalcell_ = std::make_unique<HGCalCell>(waferSize, nFine, nCoarse);
  for (int k = 0; k < 2; ++k) {  // k refers to type of wafer fine or coarse
    cellX_[k] = waferSize / (3 * ncell_[k]);
    cellY_[k] = sqrt3By2_ * cellX_[k];
    // For formulas used please refer to https://indico.cern.ch/event/1297259/contributions/5455745/attachments/2667954/4722855/Cell_centroid.pdf
    for (int j = 0; j < 6; ++j) {  // j refers to type of cell : corner, truncated, extended, truncatedMB, extendedMB
      if (j == HGCalCell::fullCell) {
        for (int i = 0; i < 6; ++i) {
          offsetX[k][j][i] = 0.0;
          offsetY[k][j][i] = 0.0;
          cellArea[k][j] = 3 * sqrt3By2_ * cellY_[k];
        }
      } else if (j == HGCalCell::cornerCell) {  // Offset for corner cells
        if (k == 0) {
          double h = (mouseBiteCut_ - sqrt3By2_ * cellY_[k]);
          double totalArea = 11.0 * sqrt3_ * std::pow(cellY_[k], 2) / 8.0;
          double cutArea1 = (sqrt3By2_ * cellY_[k] * guardRingOffset_);
          double cutArea2 = (sqrt3_ * cellY_[k] * guardRingOffset_);
          double A1 = 2.0 * cellY_[k] * h - std::pow(h, 2) / (sqrt3_);
          double A2 = sqrt3By2_ * cellY_[k] * cellY_[k];
          double A3 = sqrt3By2_ * cellY_[k] * cellY_[k] / 4.0;
          double cutArea3 = A1 + A2 + A3;

          double x3_1 = -(((2.0 * std::pow(h, 3)) / (3.0 * sqrt3_) - cellY_[k] * std::pow(h, 2)) / A1);
          double y3_1 = 0;
          double x3_2 = -(sqrt3By2_ * cellY_[k] / 3);
          double y3_2 = cellY_[k] / 6.0;
          double x3_3 = -(cellY_[k] * sqrt3_ / 4.0);
          double y3_3 = cellY_[k] * 11.0 / 12.0;

          double x1 = -(3.0 * sqrt3_ * cellY_[k] / 8.0 - sqrt3_ * guardRingOffset_ / 4.0);
          double y1 = 5.0 * cellY_[k] / 12.0 - guardRingOffset_ / 4.0;
          double x2 = -((0.5 * cellY_[k] - 0.5 * guardRingOffset_) * sqrt3By2_);
          double y2 = -(0.5 * cellY_[k] - 0.5 * guardRingOffset_) * 0.5;
          double x3 = (A1 * x3_1 + A2 * x3_2 + A3 * x3_3) / cutArea3;
          double y3 = (A1 * y3_1 + A2 * y3_2 + A3 * y3_3) / cutArea3;
          double cellArea = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag1 =
              ((5.0 * sqrt3_ * cellY_[k] / 132.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
              (cellArea);
          double yMag1 =
              ((19.0 * cellY_[k] / 132.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
              (cellArea);

          double xMag = 0.5 * xMag1 + sqrt3By2_ * yMag1;
          double yMag = sqrt3By2_ * xMag1 - 0.5 * yMag1;

          std::array<double, 6> tempOffsetX = {{(sqrt3By2_ * xMag - 0.5 * yMag),
                                                yMag,
                                                yMag,
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag - 0.5 * yMag)}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag,
                                                -xMag,
                                                (-0.5 * xMag - sqrt3By2_ * yMag),
                                                (0.5 * xMag - sqrt3By2_ * yMag),
                                                (-0.5 * xMag + sqrt3By2_ * yMag)}};

          for (int i = 0; i < 6; ++i) {
            offsetX[k][j][i] = tempOffsetX[i];
            offsetY[k][j][i] = tempOffsetY[i];
          }
        } else if (k == 1) {
          double h = (mouseBiteCut_ - guardRingOffset_) / sqrt3By2_ - cellY_[k] / 2;
          double totalArea = 11 * sqrt3_ * std::pow(cellY_[k], 2) / 8;
          double cutArea1 = (sqrt3_ * cellY_[k] * guardRingOffset_);
          double cutArea2 = (sqrt3By2_ * cellY_[k] * guardRingOffset_) + std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
          double cutArea3 = sqrt3_ * std::pow((mouseBiteCut_ - guardRingOffset_), 2) - sqrt3By2_ * std::pow(h, 2);

          double x2_0 = (0.375 * cellY_[k] * cellY_[k] - (0.25 * cellY_[k] * guardRingOffset_) +
                         std::pow(guardRingOffset_, 2) / 18) /
                        (sqrt3By2_ * cellY_[k] + guardRingOffset_ / (2 * sqrt3_));
          double y2_0 = (sqrt3_ * cellY_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (6 * sqrt3_)) /
                        (sqrt3By2_ * cellY_[k] + guardRingOffset_ / (2 * sqrt3_));
          double x3_1 = -(cellY_[k] - guardRingOffset_ - 2 * (mouseBiteCut_ - guardRingOffset_) / 3) * sqrt3By2_;
          double y3_1 = 0.5 * (cellY_[k] - guardRingOffset_ - 2 * (mouseBiteCut_ - guardRingOffset_) / 3);
          double x3_2 = -((3 * cellY_[k] / 2 - h / 3) * sqrt3By2_ + sqrt3_ * h / 6);
          double y3_2 = -(cellY_[k] / 4 + 4 * h / 6);
          double A1 = sqrt3_ * std::pow((mouseBiteCut_ - guardRingOffset_), 2);
          double A2 = sqrt3By2_ * std::pow(h, 2);

          double x1 = 0;
          double y1 = 0.5 * cellY_[k] - 0.5 * guardRingOffset_;
          double x2 = -(1.5 * sqrt3By2_ * cellY_[k] - x2_0 * 0.5 - y2_0 * sqrt3By2_);
          double y2 = -(0.25 * cellY_[k] - x2_0 * sqrt3By2_ + y2_0 / 2);
          double x3 = (A1 * x3_1 - A2 * x3_2) / (A1 - A2);
          double y3 = (A1 * y3_1 - A2 * y3_2) / (A1 - A2);
          double cellArea = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag1 = ((0.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) / (cellArea);
          double yMag1 =
              ((-5 * cellY_[k] / 42) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) / (cellArea);

          double xMag = -0.5 * xMag1 - sqrt3By2_ * yMag1;
          double yMag = sqrt3By2_ * xMag1 - 0.5 * yMag1;

          std::array<double, 6> tempOffsetX = {{(sqrt3By2_ * xMag - 0.5 * yMag),
                                                yMag,
                                                yMag,
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag - 0.5 * yMag)}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag,
                                                -xMag,
                                                (-0.5 * xMag - sqrt3By2_ * yMag),
                                                (0.5 * xMag - sqrt3By2_ * yMag),
                                                (-0.5 * xMag + sqrt3By2_ * yMag)}};
          for (int i = 0; i < 6; ++i) {
            offsetX[k][j][i] = tempOffsetX[i];
            offsetY[k][j][i] = tempOffsetY[i];
          }
        }
      } else if (j == HGCalCell::truncatedCell) {                          // Offset for truncated cells
        double totalArea = (5.0 * sqrt3_ / 4.0) * std::pow(cellY_[k], 2);  // Area of cell without any dead zone
        double cutArea =
            cellY_[k] * sqrt3_ * guardRingOffset_;  // Area of inactive region form guardring and other effects
        cellArea[k][j] = totalArea - cutArea;
        double offMag = (((-2.0 / 15.0) * totalArea * cellY_[k]) - ((cellY_[k] - (0.5 * guardRingOffset_)) * cutArea)) /
                        (cellArea[k][j]);  // Magnitude of offset
        // (x, y) coordinates of offset for 6 sides of wafer starting from bottom left edge in clockwise direction
        // offset_x = -Offset_magnitude * sin(30 + 60*i) i in (0-6)
        // offset_y = -Offset_magnitude * cos(30 + 60*i) i in (0-6)
        std::array<double, 6> tempOffsetX = {
            {-0.5 * offMag, -offMag, -0.5 * offMag, 0.5 * offMag, offMag, 0.5 * offMag}};
        std::array<double, 6> tempOffsetY = {
            {-sqrt3By2_ * offMag, 0.0, sqrt3By2_ * offMag, sqrt3By2_ * offMag, 0.0, -sqrt3By2_ * offMag}};
        for (int i = 0; i < 6; ++i) {
          offsetX[k][j][i] = tempOffsetX[i];
          offsetY[k][j][i] = tempOffsetY[i];
        }
      } else if (j == HGCalCell::extendedCell) {                           //Offset for extended cells
        double totalArea = (7.0 * sqrt3_ / 4.0) * std::pow(cellY_[k], 2);  // Area of cell without any dead zone
        double cutArea =
            cellY_[k] * sqrt3_ * guardRingOffset_;  // Area of inactive region form guardring and other effects
        cellArea[k][j] = totalArea - cutArea;
        double offMag =  // Magnitude of offset
            (((5.0 / 42.0) * totalArea * cellY_[k]) - ((cellY_[k] - (0.5 * guardRingOffset_))) * (cutArea)) /
            (cellArea[k][j]);
        // (x, y) coordinates of offset for 6 sides of wafer starting from bottom left edge in clockwise direction
        // offset_x = -Offset_magnitude * sin(30 + 60*i) i in (0-6)
        // offset_y = -Offset_magnitude * cos(30 + 60*i) i in (0-6)
        std::array<double, 6> tempOffsetX = {
            {-0.5 * offMag, -offMag, -0.5 * offMag, 0.5 * offMag, offMag, 0.5 * offMag}};
        std::array<double, 6> tempOffsetY = {
            {-sqrt3By2_ * offMag, 0.0, sqrt3By2_ * offMag, sqrt3By2_ * offMag, 0.0, -sqrt3By2_ * offMag}};
        for (int i = 0; i < 6; ++i) {
          offsetX[k][j][i] = tempOffsetX[i];
          offsetY[k][j][i] = tempOffsetY[i];
        }
      } else if (j == HGCalCell::truncatedMBCell) {
        double h = (mouseBiteCut_ - sqrt3By2_ * cellY_[k]);
        if (h > 0) {
          double totalArea = 5.0 * sqrt3_ * std::pow(cellY_[k], 2) / 4.0;
          double cutArea1 = (sqrt3_ * cellY_[k] * guardRingOffset_);
          double cutArea2 = std::pow(h, 2) / sqrt3By2_;

          double x1 = -(0.5 * cellY_[k] - 0.5 * guardRingOffset_) * sqrt3By2_;
          double y1 = -(0.5 * cellY_[k] - 0.5 * guardRingOffset_) * 0.5;
          double x2 = -((sqrt3By2_ * cellY_[k]) - (2.0 * h) / 3.0);
          double y2 = 0.5 * cellY_[k] - (2.0 * h) / (3.0 * sqrt3_);
          double cellArea = totalArea - cutArea1 - cutArea2;
          double xMag1 = ((sqrt3_ * cellY_[k] / 15.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) / (cellArea);
          double yMag1 = ((cellY_[k] / 15.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) / (cellArea);
          double xMag = -yMag1;
          double yMag = -xMag1;

          std::array<double, 6> tempOffsetX = {{(sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                yMag,
                                                yMag}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag + sqrt3By2_ * yMag),
                                                (0.5 * xMag - sqrt3By2_ * yMag),
                                                (-0.5 * xMag + sqrt3By2_ * yMag),
                                                (-0.5 * xMag - sqrt3By2_ * yMag),
                                                xMag,
                                                -xMag}};
          for (int i = 0; i < 6; ++i) {
            offsetX[k][j][i] = tempOffsetX[i];
            offsetY[k][j][i] = tempOffsetY[i];
          }
        } else {
          std::array<double, 6> tempOffsetX = {{offsetX[k][HGCalCell::truncatedCell][0],
                                                offsetX[k][HGCalCell::truncatedCell][0],
                                                offsetX[k][HGCalCell::truncatedCell][2],
                                                offsetX[k][HGCalCell::truncatedCell][2],
                                                offsetX[k][HGCalCell::truncatedCell][4],
                                                offsetX[k][HGCalCell::truncatedCell][4]}};
          std::array<double, 6> tempOffsetY = {{offsetY[k][HGCalCell::truncatedCell][0],
                                                offsetY[k][HGCalCell::truncatedCell][0],
                                                offsetY[k][HGCalCell::truncatedCell][2],
                                                offsetY[k][HGCalCell::truncatedCell][2],
                                                offsetY[k][HGCalCell::truncatedCell][4],
                                                offsetY[k][HGCalCell::truncatedCell][4]}};
          for (int i = 0; i < 6; ++i) {
            offsetX[k][j][i] = tempOffsetX[i];
            offsetY[k][j][i] = tempOffsetY[i];
          }
        }
      } else if (j == HGCalCell::extendedMBCell) {
        double h = (mouseBiteCut_ - sqrt3By2_ * cellY_[k]);
        double A = h / sqrt3By2_ + cellY_[k] / 2.0;
        double totalArea = 7.0 * sqrt3_ * std::pow(cellY_[k], 2) / 4.0;
        double cutArea1 = (sqrt3_ * cellY_[k] * guardRingOffset_);
        double cutArea2 = std::pow(A, 2) * sqrt3By2_;

        double x1 = -(sqrt3By2_ * cellY_[k] - sqrt3By2_ * guardRingOffset_ / 2.0);
        double y1 = (0.5 * cellY_[k] - 0.25 * guardRingOffset_);
        double x2 = -(sqrt3By2_ * 1.5 * cellY_[k] - A / sqrt3_);
        double y2 = -0.25 * cellY_[k] + A / 3.0;
        double cellArea = totalArea - cutArea1 - cutArea2;
        double xMag1 =
            ((-10.0 * sqrt3_ * cellY_[k] / 168.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) / (cellArea);
        double yMag1 = ((10.0 * cellY_[k] / 168.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) / (cellArea);

        double xMag = yMag1;
        double yMag = -xMag1;

        std::array<double, 6> tempOffsetX = {{(sqrt3By2_ * xMag - 0.5 * yMag),
                                              yMag,
                                              yMag,
                                              (sqrt3By2_ * xMag - 0.5 * yMag),
                                              (-sqrt3By2_ * xMag - 0.5 * yMag),
                                              (-sqrt3By2_ * xMag - 0.5 * yMag)}};
        std::array<double, 6> tempOffsetY = {{(0.5 * xMag + sqrt3By2_ * yMag),
                                              xMag,
                                              -xMag,
                                              (-0.5 * xMag - sqrt3By2_ * yMag),
                                              (0.5 * xMag - sqrt3By2_ * yMag),
                                              (-0.5 * xMag + sqrt3By2_ * yMag)}};
        for (int i = 0; i < 6; ++i) {
          offsetX[k][j][i] = tempOffsetX[i];
          offsetY[k][j][i] = tempOffsetY[i];
        }
      }
    }
    for (int j = HGCalCell::partiaclWaferCellsOffset; j < (11 + HGCalCell::partiaclWaferCellsOffset);
         ++j) {  //For cells in partial wafers
      if (j == (HGCalCell::halfCell)) {
        double totalArea = (3.0 * sqrt3_ / 4.0) * std::pow(cellY_[k], 2);
        double cutArea = cellY_[k] * 2.0 * guardRingOffset_ - std::pow(guardRingOffset_, 2) / sqrt3_;
        double cellArea = totalArea - cutArea;

        double x1 = (-cellY_[k] * guardRingOffset_ + 2 * std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellY_[k] - guardRingOffset_ / sqrt3_);
        double y1 = 0;
        double xMag = ((-2.0 * sqrt3_ * cellY_[k] / 9.0) * totalArea - (cutArea * x1)) / (cellArea);
        double yMag = (0 * totalArea - (cutArea * y1)) / (cellArea);

        std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                              (-sqrt3By2_ * xMag + 0.5 * yMag),
                                              yMag,
                                              (sqrt3By2_ * xMag + 0.5 * yMag),
                                              (sqrt3By2_ * xMag - 0.5 * yMag),
                                              -yMag}};
        std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                              (-sqrt3By2_ * yMag - 0.5 * xMag),
                                              -xMag,
                                              (-0.5 * xMag + sqrt3By2_ * yMag),
                                              (0.5 * xMag + sqrt3By2_ * yMag),
                                              xMag}};

        for (int i = 0; i < 6; ++i) {
          offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
          offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
        }
      } else if (j == (HGCalCell::halfTrunCell)) {
        double totalArea = 5 * sqrt3_ * std::pow(cellY_[k], 2) / 8;
        double cutArea1 = (sqrt3By2_ * cellY_[k] * guardRingOffset_);
        double cutArea2 = (3 * cellY_[k] * guardRingOffset_) / 2 - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);

        double x1 = -sqrt3_ * cellY_[k] / 4;
        double y1 = (0.5 * cellY_[k] - 0.5 * guardRingOffset_);
        double x2 = (-3 * cellY_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (3 * cellY_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellY_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18 -
                     3 * std::pow(cellY_[k], 2) / 8) /
                    (3 * cellY_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        double cellArea = totalArea - cutArea1 - cutArea2;
        double xMag1 = ((-7 * sqrt3_ * cellY_[k] / 30) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) / (cellArea);
        double yMag = ((-2 * cellY_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) / (cellArea);
        double xMag = -xMag1;

        std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                              (-sqrt3By2_ * xMag + 0.5 * yMag),
                                              yMag,
                                              (sqrt3By2_ * xMag + 0.5 * yMag),
                                              (sqrt3By2_ * xMag - 0.5 * yMag),
                                              -yMag}};
        std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                              (-sqrt3By2_ * yMag - 0.5 * xMag),
                                              -xMag,
                                              (-0.5 * xMag + sqrt3By2_ * yMag),
                                              (0.5 * xMag + sqrt3By2_ * yMag),
                                              xMag}};
        for (int i = 0; i < 6; ++i) {
          offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
          offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
        }
      } else if (j == (HGCalCell::halfExtCell)) {
        double totalArea = (7.0 * sqrt3_ / 8.0) * std::pow(cellY_[k], 2);
        double cutArea1 = cellY_[k] * sqrt3By2_ * guardRingOffset_;
        double cutArea2 = cellY_[k] * 2.0 * guardRingOffset_ - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        double cellArea = totalArea - cutArea1 - cutArea2;

        double x1 = -sqrt3By2_ * cellY_[k] / 2;
        double y1 = -(cellY_[k] - guardRingOffset_ / 2);
        double x2 = (-cellY_[k] * guardRingOffset_ + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellY_[k] - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellY_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18) /
                    (2 * cellY_[k] - guardRingOffset_ / (2 * sqrt3_));
        double xMag = ((-5 * sqrt3_ * cellY_[k] / 21.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) / (cellArea);
        double yMag = ((-5 * cellY_[k] / 42.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) / (cellArea);

        std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                              (-sqrt3By2_ * xMag + 0.5 * yMag),
                                              yMag,
                                              (sqrt3By2_ * xMag + 0.5 * yMag),
                                              (sqrt3By2_ * xMag - 0.5 * yMag),
                                              -yMag}};
        std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                              (-sqrt3By2_ * yMag - 0.5 * xMag),
                                              -xMag,
                                              (-0.5 * xMag + sqrt3By2_ * yMag),
                                              (0.5 * xMag + sqrt3By2_ * yMag),
                                              xMag}};

        for (int i = 0; i < 6; ++i) {
          offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
          offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
        }
      } else if (j == (HGCalCell::LDPartial0714Cell)) {
        if (k == 1) {
          double totalArea = (9.0 * sqrt3_ / 4.0) * std::pow(cellY_[k], 2);
          double cutArea1 =
              (3 * cellY_[k] * sqrt3By2_ * guardRingOffset_) - (std::pow(guardRingOffset_, 2) / (2 * sqrt3_));
          double cutArea2 = (3 * cellY_[k] * sqrt3By2_ * guardRingOffset_);
          double cutArea3 = sqrt3_ * std::pow((mouseBiteCut_ - (guardRingOffset_ / sqrt3By2_)), 2) / 2;
          double x1_0 = ((3.375 * cellY_[k] * cellY_[k]) - (cellY_[k] * 0.75 * guardRingOffset_) +
                         (std::pow(guardRingOffset_, 2) / 18)) /
                        ((3 * cellY_[k] * sqrt3By2_) - (guardRingOffset_ / (2 * sqrt3_)));
          double y1_0 =
              ((3 * cellY_[k] * sqrt3By2_ * guardRingOffset_ / 2) - (std::pow(guardRingOffset_, 2) / (6 * sqrt3_))) /
              ((3 * cellY_[k] * sqrt3By2_) - (guardRingOffset_ / (2 * sqrt3_)));

          double x2_0 = (3 * sqrt3By2_ * cellY_[k] / 2);
          double y2_0 = guardRingOffset_ / 2;

          double x1 = (cellY_[k] / 2 - guardRingOffset_) * sqrt3By2_ + x1_0 * 0.5 + y1_0 * sqrt3By2_;
          double y1 = cellY_[k] + (cellY_[k] / 2 - guardRingOffset_) * 0.5 - x1_0 * sqrt3By2_ + y1_0 * 0.5;

          double x2 = x2_0 - sqrt3By2_ * cellY_[k];
          double y2 = -(cellY_[k] - y2_0);

          double x3 = sqrt3_ * cellY_[k] - mouseBiteCut_ + (mouseBiteCut_ - (guardRingOffset_ / sqrt3By2_)) / 3;
          double y3 = -(cellY_[k] - sqrt3_ * (mouseBiteCut_ - (guardRingOffset_ / sqrt3By2_)) / 3 - guardRingOffset_);

          double cellArea = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag =
              ((sqrt3_ * cellY_[k] / 8) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) / (cellArea);
          double yMag =
              ((-1 * cellY_[k] / 8) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) / (cellArea);

          std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag + 0.5 * yMag),
                                                yMag,
                                                (sqrt3By2_ * xMag + 0.5 * yMag),
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                -yMag}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                                (-sqrt3By2_ * yMag - 0.5 * xMag),
                                                -xMag,
                                                (-0.5 * xMag + sqrt3By2_ * yMag),
                                                (0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag}};

          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
          }
        } else {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial0209Cell)) {
        if (k == 1) {
          double totalArea = (23.0 * sqrt3_ / 8.0) * std::pow(cellY_[k], 2);
          double cutArea1 =
              (5 * cellY_[k] * sqrt3By2_ * guardRingOffset_) - (std::pow(guardRingOffset_, 2) / (2 * sqrt3_));
          double cutArea2 = (4 * cellY_[k] * guardRingOffset_);
          double cutArea3 = std::pow(mouseBiteCut_, 2) / sqrt3_;

          double x1_0 = (9.375 * cellY_[k] * cellY_[k] - (cellY_[k] * 1.25 * guardRingOffset_) +
                         (std::pow(guardRingOffset_, 2) / 18)) /
                        ((5 * cellY_[k] * sqrt3By2_) - (guardRingOffset_ / (2 * sqrt3_)));
          double y1_0 =
              ((5 * cellY_[k] * sqrt3By2_ * guardRingOffset_ / 2) - (std::pow(guardRingOffset_, 2) / (6 * sqrt3_))) /
              ((5 * cellY_[k] * sqrt3By2_) - (guardRingOffset_ / (2 * sqrt3_)));

          double x1 = (1.5 * cellY_[k]) * sqrt3By2_ - x1_0 * 0.5 - y1_0 * sqrt3By2_;
          double y1 = -0.25 * cellY_[k] + x1_0 * sqrt3By2_ - y1_0 * 0.5;
          double x2 = -(sqrt3By2_ * cellY_[k] - 0.5 * guardRingOffset_);
          double y2 = 1.5 * cellY_[k];
          double x3 = -(sqrt3By2_ * cellY_[k] - mouseBiteCut_ / 3);
          double y3 = 3.5 * cellY_[k] - (5 * mouseBiteCut_) / 3 * sqrt3_;

          double cellArea = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag =
              ((-9 * cellY_[k] / (sqrt3_ * 92)) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
              (cellArea);
          double yMag =
              ((199 * cellY_[k] / (sqrt3_ * 276)) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
              (cellArea);

          std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag + 0.5 * yMag),
                                                yMag,
                                                (sqrt3By2_ * xMag + 0.5 * yMag),
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                -yMag}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                                (-sqrt3By2_ * yMag - 0.5 * xMag),
                                                -xMag,
                                                (-0.5 * xMag + sqrt3By2_ * yMag),
                                                (0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag}};
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
          }
        } else {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial0007Cell)) {
        if (k == 1) {
          double totalArea = (5.0 * sqrt3_ / 4.0) * std::pow(cellY_[k], 2);
          double cutArea1 = (cellY_[k] * guardRingOffset_);
          double cutArea2 = (sqrt3_ * cellY_[k] * guardRingOffset_);
          double h = cellY_[k] - (sqrt3By2_ * cellY_[k] - mouseBiteCut_) / sqrt3By2_;
          double cutArea3 = sqrt3_ * std::pow(h, 2) / 2;

          double x1 = cellY_[k] * sqrt3By2_ - guardRingOffset_ / 2;
          double y1 = 0;
          double x2 = 0;
          double y2 = 0.5 * cellY_[k] - guardRingOffset_ / 2;
          double x3 = sqrt3By2_ * cellY_[k] - guardRingOffset_ - h / sqrt3_;
          double y3 = 0.5 * cellY_[k] - guardRingOffset_ - h / 3;

          double cellArea = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag = ((0.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) / (cellArea);
          double yMag =
              ((-2 * cellY_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) / (cellArea);

          std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag + 0.5 * yMag),
                                                yMag,
                                                (sqrt3By2_ * xMag + 0.5 * yMag),
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                -yMag}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                                (-sqrt3By2_ * yMag - 0.5 * xMag),
                                                -xMag,
                                                (-0.5 * xMag + sqrt3By2_ * yMag),
                                                (0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag}};
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
          }
        } else {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial0815Cell)) {
        if (k == 1) {
          double totalArea = sqrt3_ * std::pow(cellY_[k], 2);
          double cutArea1 = (sqrt3_ * cellY_[k] * guardRingOffset_);
          double cutArea2 = (sqrt3_ * cellY_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
          double cutArea3 = sqrt3_ * std::pow((mouseBiteCut_ - guardRingOffset_ / sqrt3By2_), 2) / 2;

          double x2_0 = (1.5 * cellY_[k] * cellY_[k] - (0.5 * cellY_[k] * guardRingOffset_) +
                         std::pow(guardRingOffset_, 2) / 18) /
                        (sqrt3_ * cellY_[k] - guardRingOffset_ / (2 * sqrt3_));
          double y2_0 = (sqrt3By2_ * cellY_[k] * guardRingOffset_ - std::pow(guardRingOffset_, 2) / (sqrt3_ * 3)) /
                        (sqrt3_ * cellY_[k] - guardRingOffset_ / (2 * sqrt3_));
          double x1 = 0;
          double y1 = 0.5 * cellY_[k] - guardRingOffset_ / 2;
          double x2 = x2_0 * 0.5 - y2_0 * sqrt3By2_;
          double y2 = -(cellY_[k] - (x2_0 * sqrt3By2_ + y2_0 * 0.5));
          double x3 = sqrt3By2_ * cellY_[k] - mouseBiteCut_ + (mouseBiteCut_ - guardRingOffset_ / sqrt3By2_) / 3;
          double y3 = cellY_[k] - (mouseBiteCut_ - guardRingOffset_ / sqrt3By2_) / sqrt3_ - guardRingOffset_;

          double cellArea = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag = -((-sqrt3_ * cellY_[k] / 9) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
                        (cellArea);
          double yMag =
              ((-cellY_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) / (cellArea);

          std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag + 0.5 * yMag),
                                                yMag,
                                                (sqrt3By2_ * xMag + 0.5 * yMag),
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                -yMag}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                                (-sqrt3By2_ * yMag - 0.5 * xMag),
                                                -xMag,
                                                (-0.5 * xMag + sqrt3By2_ * yMag),
                                                (0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag}};
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
          }
        } else {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial1415Cell)) {
        if (k == 1) {
          double totalArea = 7 * sqrt3_ * std::pow(cellY_[k], 2) / 4;
          double cutArea1 = (3 * cellY_[k] * guardRingOffset_);
          double cutArea2 = (2 * sqrt3_ * cellY_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) * sqrt3By2_;
          double cutArea3 = std::pow((mouseBiteCut_ - guardRingOffset_), 2) / sqrt3_;

          double x2_0 = (6 * cellY_[k] * cellY_[k] - std::pow(guardRingOffset_, 2)) /
                        (2 * sqrt3_ * cellY_[k] - guardRingOffset_ * sqrt3By2_);
          double y2_0 = (sqrt3_ * cellY_[k] * guardRingOffset_ - std::pow(guardRingOffset_, 2) / sqrt3_) /
                        (2 * sqrt3_ * cellY_[k] - guardRingOffset_ * sqrt3By2_);
          double x1 = -sqrt3By2_ * cellY_[k] + guardRingOffset_ / 2;
          double y1 = -cellY_[k];
          double x2 = sqrt3By2_ * cellY_[k] - x2_0 * 0.5 - y2_0 * sqrt3By2_;
          double y2 = 0.5 * cellY_[k] - x2_0 * sqrt3By2_ + y2_0 * 0.5;
          double h = (mouseBiteCut_ - guardRingOffset_) / sqrt3By2_;
          double x3 = -(cellY_[k] - h / 3 - guardRingOffset_) * sqrt3By2_;
          double y3 = 5 * h / 6 - 5 * cellY_[k] / 2;

          double cellArea = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag =
              ((-2 * cellY_[k] / (7 * sqrt3_)) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
              (cellArea);
          double yMag =
              ((-cellY_[k] / 3) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) / (cellArea);

          std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag + 0.5 * yMag),
                                                yMag,
                                                (sqrt3By2_ * xMag + 0.5 * yMag),
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                -yMag}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                                (-sqrt3By2_ * yMag - 0.5 * xMag),
                                                -xMag,
                                                (-0.5 * xMag + sqrt3By2_ * yMag),
                                                (0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag}};
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
          }
        } else {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial1515Cell)) {
        if (k == 1) {
          double totalArea = 7 * sqrt3_ * std::pow(cellY_[k], 2) / 8;
          double cutArea1 = (2 * cellY_[k] * guardRingOffset_);
          double cutArea2 = (sqrt3By2_ * cellY_[k] * guardRingOffset_);
          double cutArea3 = cellY_[k] * (mouseBiteCut_ - guardRingOffset_) - (sqrt3_ * cellY_[k] * cellY_[k] / 8);

          double x1 = -guardRingOffset_ / 2;
          double y1 = 0;
          double x2 = -(sqrt3By2_ * cellY_[k] / 2);
          double y2 = -(cellY_[k] - 0.5 * guardRingOffset_);
          double x3 = (cellY_[k] * cellY_[k] / 8 - sqrt3_ * cellY_[k] * (mouseBiteCut_ - guardRingOffset_) / 4) /
                      ((mouseBiteCut_ - guardRingOffset_) - sqrt3_ * cellY_[k] / 8);
          double y3 =
              (std::pow((mouseBiteCut_ - guardRingOffset_), 2) / sqrt3_ -
               (1.25 * cellY_[k] * (mouseBiteCut_ - guardRingOffset_)) + 7 * sqrt3_ * cellY_[k] * cellY_[k] / 48) /
              ((mouseBiteCut_ - guardRingOffset_) - sqrt3_ * cellY_[k] / 8);

          double cellArea = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag =
              (-(cellY_[k] / (sqrt3_)) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) / (cellArea);
          double yMag =
              ((-5 * cellY_[k] / 42) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) / (cellArea);

          std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag + 0.5 * yMag),
                                                yMag,
                                                (sqrt3By2_ * xMag + 0.5 * yMag),
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                -yMag}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                                (-sqrt3By2_ * yMag - 0.5 * xMag),
                                                -xMag,
                                                (-0.5 * xMag + sqrt3By2_ * yMag),
                                                (0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag}};
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
          }
        } else {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::HDPartial0920Cell)) {
        if (k == 0) {
          double totalArea = 37 * sqrt3_ * std::pow(cellY_[k], 2) / 24;
          double cutArea1 = (4 * cellY_[k] * guardRingOffset_) / sqrt3_;
          double cutArea2 =
              (7 * cellY_[k] * guardRingOffset_) / (2 * sqrt3_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);

          double x1 = cellY_[k] / (2 * sqrt3_);
          double y1 = -(0.5 * cellY_[k] - 0.5 * guardRingOffset_);
          double x2_0 = ((2.041 * cellY_[k] * cellY_[k]) - (cellY_[k] * 0.583 * guardRingOffset_) +
                         (std::pow(guardRingOffset_, 2) / 18)) /
                        ((7 * cellY_[k] / (2 * sqrt3_)) - (guardRingOffset_ / (2 * sqrt3_)));
          double y2_0 =
              ((7 * cellY_[k] * guardRingOffset_ / (4 * sqrt3_)) - std::pow(guardRingOffset_, 2) / (6 * sqrt3_)) /
              ((7 * cellY_[k] / (2 * sqrt3_)) - (guardRingOffset_ / (2 * sqrt3_)));

          double x2 = (0.5 * x2_0) - (sqrt3By2_ * y2_0) + (cellY_[k] * 0.5 * sqrt3By2_);
          double y2 = -(0.5 * y2_0) - (sqrt3By2_ * x2_0) + (cellY_[k] * 1.25);
          double cellArea = totalArea - cutArea1 - cutArea2;
          double xMag = ((25 * sqrt3_ * cellY_[k] / 148) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) / (cellArea);
          double yMag = ((73 * cellY_[k] / 444) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) / (cellArea);

          std::array<double, 6> tempOffsetX = {{(-sqrt3By2_ * xMag - 0.5 * yMag),
                                                (-sqrt3By2_ * xMag + 0.5 * yMag),
                                                yMag,
                                                (sqrt3By2_ * xMag + 0.5 * yMag),
                                                (sqrt3By2_ * xMag - 0.5 * yMag),
                                                -yMag}};
          std::array<double, 6> tempOffsetY = {{(0.5 * xMag - sqrt3By2_ * yMag),
                                                (-sqrt3By2_ * yMag - 0.5 * xMag),
                                                -xMag,
                                                (-0.5 * xMag + sqrt3By2_ * yMag),
                                                (0.5 * xMag + sqrt3By2_ * yMag),
                                                xMag}};
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
          }
        } else {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::HDPartial1021Cell)) {
        if (k == 0) {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        } else {
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      }
    }
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalCellOffset initialized with waferSize " << waferSize << " number of cells "
                                << nFine << ":" << nCoarse << " Guardring offset " << guardRingOffset_ << " Mousebite "
                                << mouseBiteCut_;
#endif
}

std::pair<double, double> HGCalCellOffset::cellOffsetUV2XY1(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  double x_off(0), y_off(0);
  std::pair<int, int> cell = hgcalcell_->cellType(u, v, ncell_[type], placementIndex);
  int cellPos = cell.first;
  int cellType = cell.second;
  if (cellType == HGCalCell::truncatedCell || cellType == HGCalCell::extendedCell) {
    x_off = offsetX[type][cellType][cellPos - HGCalCell::bottomLeftEdge];
    y_off = offsetY[type][cellType][cellPos - HGCalCell::bottomLeftEdge];
  } else if ((cellType == HGCalCell::cornerCell) || (cellType == HGCalCell::truncatedMBCell) ||
             (cellType == HGCalCell::extendedMBCell)) {
    // The offset fo corner cells, is flipped along y-axis for 60 degree rotation of wafer
    // and from forward to backward wafers
    if (((placementIndex >= HGCalCell::cellPlacementExtra) && (placementIndex % 2 == 0)) ||
        ((placementIndex < HGCalCell::cellPlacementExtra) && (placementIndex % 2 == 1))) {
      cellPos = HGCalCell::bottomCorner + (6 + HGCalCell::bottomCorner - cellPos) % 6;
    }
    x_off = offsetX[type][cellType][cellPos - HGCalCell::bottomCorner];
    y_off = offsetY[type][cellType][cellPos - HGCalCell::bottomCorner];
    if (((placementIndex >= HGCalCell::cellPlacementExtra) && (placementIndex % 2 == 0)) ||
        ((placementIndex < HGCalCell::cellPlacementExtra) && (placementIndex % 2 == 1))) {
      x_off = -1 * x_off;
    }
  }
  return std::make_pair(x_off, y_off);
}

std::pair<double, double> HGCalCellOffset::cellOffsetUV2XY1(
    int32_t u, int32_t v, int32_t placementIndex, int32_t type, int32_t partialType) {
  if (type != 0)
    type = 1;
  std::pair<double, double> offset = HGCalCellOffset::cellOffsetUV2XY1(u, v, placementIndex, type);
  double x_off = offset.first;
  double y_off = offset.second;
  std::pair<int, int> cell = hgcalcell_->cellType(u, v, ncell_[type], placementIndex, partialType);
  int cellPos = cell.first;
  int cellType = cell.second;
  if ((cellType >= HGCalCell::partiaclWaferCellsOffset) || (cellPos >= HGCalCell::partiaclCellsPosOffset)) {
    if (cellType == HGCalCell::truncatedCell || cellType == HGCalCell::extendedCell) {
      if (cellPos == HGCalCell::topCell) {
        int Pos(0);
        Pos = (placementIndex + HGCalCell::topRightEdge - HGCalCell::bottomLeftEdge) % HGCalCell::cellPlacementExtra;
        x_off = (placementIndex >= HGCalCell::cellPlacementExtra) ? -offsetX[type][cellType][Pos]
                                                                  : offsetX[type][cellType][Pos];
        y_off = offsetY[type][cellType][Pos];
      } else if (cellPos == HGCalCell::bottomCell) {
        int Pos(0);
        Pos = (placementIndex) % HGCalCell::cellPlacementExtra;
        x_off = (placementIndex >= HGCalCell::cellPlacementExtra) ? -offsetX[type][cellType][Pos]
                                                                  : offsetX[type][cellType][Pos];
        y_off = offsetY[type][cellType][Pos];
      }
    } else if ((cellType == HGCalCell::halfCell) || (cellType == HGCalCell::halfTrunCell) ||
               (cellType == HGCalCell::halfExtCell) || (cellType == HGCalCell::LDPartial0714Cell) ||
               (cellType == HGCalCell::LDPartial0815Cell) || (cellType == HGCalCell::HDPartial0920Cell) ||
               (cellType == HGCalCell::HDPartial1021Cell)) {
      int cellType1 = cellType - HGCalCell::partiaclWaferCellsOffset;
      if (cellType == HGCalCell::halfCell) {
        std::cout << u << ":" << v << " 2" << std::endl;
      }
      if (cellPos == HGCalCell::leftCell) {
        int placeIndex = placementIndex % HGCalCell::cellPlacementExtra;
        x_off = offsetPartialX[type][cellType1][placeIndex];
        y_off = offsetPartialY[type][cellType1][placeIndex];
      } else if (cellPos == HGCalCell::rightCell) {
        int placeIndex = (HGCalCell::cellPlacementExtra - placementIndex) % HGCalCell::cellPlacementExtra;
        x_off = -offsetPartialX[type][cellType1][placeIndex];
        y_off = offsetPartialY[type][cellType1][placeIndex];
      }
      x_off = placementIndex < HGCalCell::cellPlacementExtra ? x_off : -x_off;
    } else if ((cellType == HGCalCell::LDPartial0209Cell) || (cellType == HGCalCell::LDPartial0007Cell) ||
               (cellType == HGCalCell::LDPartial1415Cell) || (cellType == HGCalCell::LDPartial1515Cell)) {
      int cellType1 = cellType - HGCalCell::partiaclWaferCellsOffset;
      int placeIndex = placementIndex % HGCalCell::cellPlacementExtra;
      x_off = offsetPartialX[type][cellType1][placeIndex];
      y_off = offsetPartialY[type][cellType1][placeIndex];
      x_off = placementIndex < HGCalCell::cellPlacementExtra ? x_off : -x_off;
    }
  }
  return std::make_pair(x_off, y_off);
}

double HGCalCellOffset::cellAreaUV(int32_t u, int32_t v, int32_t placementIndex, int32_t type, bool reco) {
  if (type != 0)
    type = 1;
  double area(0);
  std::pair<int, int> cell = hgcalcell_->cellType(u, v, ncell_[type], placementIndex);
  int cellType = cell.second;
  area = reco ? cellArea[type][cellType] : HGCalParameters::k_ScaleToDDD2 * cellArea[type][cellType];
  return area;
}
