#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalCellOffset.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include <vector>
#include <iostream>

//#define EDM_ML_DEBUG

HGCalCellOffset::HGCalCellOffset(double waferSize,
                                 int32_t nFine,
                                 int32_t nCoarse,
                                 double guardRingOffset_,
                                 double mouseBiteCut_,
                                 double sizeOffset_) {
  ncell_[0] = nFine;
  ncell_[1] = nCoarse;
  hgcalcell_ = std::make_unique<HGCalCell>(waferSize, nFine, nCoarse);
  double guardRingSizeOffset_ = guardRingOffset_ + sizeOffset_;
  for (int k = 0; k < 2; ++k) {  // k refers to type of wafer fine or coarse
    cellX_[k] = waferSize / (3 * ncell_[k]);
    cellY_[k] = sqrt3By2_ * cellX_[k];
    // For formulas used please refer to https://indico.cern.ch/event/1297259/contributions/5455745/attachments/2667954/4722855/Cell_centroid.pdf
    for (int j = 0; j < 6; ++j) {  // j refers to type of cell : corner, truncated, extended, truncatedMB, extendedMB
      if (j == HGCalCell::fullCell) {
        for (int i = 0; i < 6; ++i) {
          offsetX[k][j][i] = 0.0;
          offsetY[k][j][i] = 0.0;
          cellArea[k][j] = 3 * sqrt3By2_ * std::pow(cellX_[k], 2);
        }
      } else if (j == HGCalCell::cornerCell) {  // Offset for corner cells
        if (k == 0) {
          double h = (mouseBiteCut_ - sqrt3By2_ * cellX_[k]);
          double H = mouseBiteCut_ - (1 / sqrt3By2_ * guardRingSizeOffset_);
          double h1 = H - (sqrt3_ / 4 * cellX_[k]) + (guardRingSizeOffset_ / (2 * sqrt3_));
          double h2 = H - (sqrt3_ / 2 * cellX_[k]) + (guardRingSizeOffset_ / (2 * sqrt3_));
          double totalArea = 11.0 * sqrt3_ * std::pow(cellX_[k], 2) / 8.0;
          double cutArea1 =
              (sqrt3_ * cellX_[k] * guardRingSizeOffset_) - (0.5 / sqrt3_ * std::pow(guardRingSizeOffset_, 2));
          double cutArea2 =
              (sqrt3By2_ * cellX_[k] * guardRingSizeOffset_) - (0.5 / sqrt3_ * std::pow(guardRingSizeOffset_, 2));
          double A1 = 2.0 * cellX_[k] * h - std::pow(h, 2) / (sqrt3_);
          double A2 = sqrt3By2_ * cellX_[k] * cellX_[k];
          double A3 = sqrt3By2_ * cellX_[k] * cellX_[k] / 4.0;
          double cutArea3 =
              sqrt3_ * std::pow(H, 2) - (1 / sqrt3By2_ * std::pow(h1, 2)) - (1 / sqrt3By2_ * std::pow(h2, 2));
          double x3_1 = -(((2.0 * std::pow(h, 3)) / (3.0 * sqrt3_) - cellX_[k] * std::pow(h, 2)) / A1);
          double y3_1 = 0;
          double x3_2 = -(sqrt3By2_ * cellX_[k] / 3);
          double y3_2 = cellX_[k] / 6.0;
          double x3_3 = -(cellX_[k] * sqrt3_ / 4.0);
          double y3_3 = cellX_[k] * 11.0 / 12.0;

          double x1 = -(3.0 * sqrt3_ * cellX_[k] / 8.0 - sqrt3_ * guardRingOffset_ / 4.0);
          double y1 = 5.0 * cellX_[k] / 12.0 - guardRingOffset_ / 4.0;
          double x2 = -((0.5 * cellX_[k] - 0.5 * guardRingOffset_) * sqrt3By2_);
          double y2 = -(0.5 * cellX_[k] - 0.5 * guardRingOffset_) * 0.5;
          double x3 = (A1 * x3_1 + A2 * x3_2 + A3 * x3_3) / cutArea3;
          double y3 = (A1 * y3_1 + A2 * y3_2 + A3 * y3_3) / cutArea3;
          cellArea[k][j] = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag1 =
              ((5.0 * sqrt3_ * cellX_[k] / 132.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
              (cellArea[k][j]);
          double yMag1 =
              ((19.0 * cellX_[k] / 132.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
              (cellArea[k][j]);

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
          double h = (mouseBiteCut_ - guardRingOffset_) / sqrt3By2_ - cellX_[k] / 2;
          double H = mouseBiteCut_ - (1 / sqrt3By2_ * guardRingSizeOffset_);
          double h1 = H - ((sqrt3_ / 4) * cellX_[k]) + (guardRingSizeOffset_ / (2 * sqrt3_));
          double totalArea = 11.0 * sqrt3_ * std::pow(cellX_[k], 2) / 8.0;
          double cutArea1 =
              (sqrt3_ * cellX_[k] * guardRingSizeOffset_) - (0.5 / sqrt3_ * std::pow(guardRingSizeOffset_, 2));
          double cutArea2 =
              (sqrt3By2_ * cellX_[k] * guardRingSizeOffset_) - (0.5 / sqrt3_ * std::pow(guardRingSizeOffset_, 2));
          double cutArea3 = sqrt3_ * std::pow(H, 2) - (1 / sqrt3By2_ * std::pow(h1, 2));
          //double cutArea3 = sqrt3_ * std::pow((mouseBiteCut_ - guardRingOffset_), 2) - sqrt3By2_ * std::pow(h, 2);

          double x2_0 = (0.375 * cellX_[k] * cellX_[k] - (0.25 * cellX_[k] * guardRingOffset_) +
                         std::pow(guardRingOffset_, 2) / 18) /
                        (sqrt3By2_ * cellX_[k] + guardRingOffset_ / (2 * sqrt3_));
          double y2_0 = (sqrt3_ * cellX_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (6 * sqrt3_)) /
                        (sqrt3By2_ * cellX_[k] + guardRingOffset_ / (2 * sqrt3_));
          double x3_1 = -(cellX_[k] - guardRingOffset_ - 2 * (mouseBiteCut_ - guardRingOffset_) / 3) * sqrt3By2_;
          double y3_1 = 0.5 * (cellX_[k] - guardRingOffset_ - 2 * (mouseBiteCut_ - guardRingOffset_) / 3);
          double x3_2 = -((3 * cellX_[k] / 2 - h / 3) * sqrt3By2_ + sqrt3_ * h / 6);
          double y3_2 = -(cellX_[k] / 4 + 4 * h / 6);
          double A1 = sqrt3_ * std::pow((mouseBiteCut_ - guardRingOffset_), 2);
          double A2 = sqrt3By2_ * std::pow(h, 2);

          double x1 = 0;
          double y1 = 0.5 * cellX_[k] - 0.5 * guardRingOffset_;
          double x2 = -(1.5 * sqrt3By2_ * cellX_[k] - x2_0 * 0.5 - y2_0 * sqrt3By2_);
          double y2 = -(0.25 * cellX_[k] - x2_0 * sqrt3By2_ + y2_0 / 2);
          double x3 = (A1 * x3_1 - A2 * x3_2) / (A1 - A2);
          double y3 = (A1 * y3_1 - A2 * y3_2) / (A1 - A2);
          cellArea[k][j] = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag1 = ((0.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) / (cellArea[k][j]);
          double yMag1 = ((-5 * cellX_[k] / 42) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
                         (cellArea[k][j]);

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
        double totalArea = (5.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);  // Area of cell without any dead zone
        double cutArea =
            cellX_[k] * sqrt3_ * guardRingSizeOffset_;  // Area of inactive region form guardring and other effects
        cellArea[k][j] = totalArea - cutArea;
        double offMag = (((-2.0 / 15.0) * totalArea * cellX_[k]) - ((cellX_[k] - (0.5 * guardRingOffset_)) * cutArea)) /
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
        double totalArea = (7.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);  // Area of cell without any dead zone
        double cutArea =
            cellX_[k] * sqrt3_ * guardRingSizeOffset_;  // Area of inactive region form guardring and other effects
        cellArea[k][j] = totalArea - cutArea;
        double offMag =  // Magnitude of offset
            (((5.0 / 42.0) * totalArea * cellX_[k]) - ((cellX_[k] - (0.5 * guardRingOffset_))) * (cutArea)) /
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
        double H = mouseBiteCut_ - (1 / sqrt3By2_ * guardRingSizeOffset_);
        double h = H - (sqrt3_ / 2 * cellX_[k]) + (guardRingSizeOffset_ / (2 * sqrt3_));
        if (h > 0) {
          double totalArea = 5.0 * sqrt3_ * std::pow(cellX_[k], 2) / 4.0;

          double cutArea1 = (sqrt3_ * cellX_[k] * guardRingSizeOffset_);
          double cutArea2 = std::pow(h, 2) / sqrt3By2_;

          double x1 = -(0.5 * cellX_[k] - 0.5 * guardRingOffset_) * sqrt3By2_;
          double y1 = -(0.5 * cellX_[k] - 0.5 * guardRingOffset_) * 0.5;
          double x2 = -((sqrt3By2_ * cellX_[k]) - (2.0 * h) / 3.0);
          double y2 = 0.5 * cellX_[k] - (2.0 * h) / (3.0 * sqrt3_);
          cellArea[k][j] = totalArea - cutArea1 - cutArea2;
          double xMag1 =
              ((sqrt3_ * cellX_[k] / 15.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) / (cellArea[k][j]);
          double yMag1 = ((cellX_[k] / 15.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) / (cellArea[k][j]);
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
          cellArea[k][j] = cellArea[k][HGCalCell::truncatedCell];
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
        double H = mouseBiteCut_ - (1 / sqrt3By2_ * guardRingSizeOffset_);
        double h = H - (sqrt3_ / 4 * cellX_[k]) + (guardRingSizeOffset_ / (2 * sqrt3_));

        double totalArea = 7.0 * sqrt3_ * std::pow(cellX_[k], 2) / 4.0;
        double cutArea1 = (sqrt3_ * cellX_[k] * guardRingSizeOffset_);
        double cutArea2 = std::pow(h, 2) / sqrt3By2_;

        double x1 = -(sqrt3By2_ * cellX_[k] - sqrt3By2_ * guardRingOffset_ / 2.0);
        double y1 = (0.5 * cellX_[k] - 0.25 * guardRingOffset_);
        double x2 = -(sqrt3By2_ * 1.5 * cellX_[k] - h / sqrt3_);
        double y2 = -0.25 * cellX_[k] + h / 3.0;
        cellArea[k][j] = totalArea - cutArea1 - cutArea2;
        double xMag1 =
            ((-10.0 * sqrt3_ * cellX_[k] / 168.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) / (cellArea[k][j]);
        double yMag1 = ((10.0 * cellX_[k] / 168.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) / (cellArea[k][j]);

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
    for (int j = HGCalCell::partiaclWaferCellsOffset; j < (25 + HGCalCell::partiaclWaferCellsOffset);
         ++j) {  //For cells in partial wafers
      if (j == (HGCalCell::halfCell)) {
        double totalArea = (3.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);
        double cutArea = cellX_[k] * 2.0 * guardRingOffset_ - std::pow(guardRingOffset_, 2) / sqrt3_;
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea;
        double x1 = (-cellX_[k] * guardRingOffset_ + 2 * std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / sqrt3_);
        double y1 = 0;
        double xMag = ((-2.0 * sqrt3_ * cellX_[k] / 9.0) * totalArea - (cutArea * x1)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = (0 * totalArea - (cutArea * y1)) / (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == (HGCalCell::extHalfTrunCell)) {
        double totalArea = 5 * sqrt3_ * std::pow(cellX_[k], 2) / 8;
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingSizeOffset_) - guardRingOffset_ * guardRingSizeOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);

        double x1 = -sqrt3_ * cellX_[k] / 4;
        double y1 = (0.5 * cellX_[k] - 0.5 * guardRingOffset_);
        double x2 = (-3 * cellX_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18 -
                     3 * std::pow(cellX_[k], 2) / 8) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2;
        double xMag1 = ((-7 * sqrt3_ * cellX_[k] / 30) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                       (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-2 * cellX_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
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
      } else if (j == (HGCalCell::extHalfExtCell)) {
        double totalArea = (7.0 * sqrt3_ / 8.0) * std::pow(cellX_[k], 2);
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingSizeOffset_) - guardRingOffset_ * guardRingSizeOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2;

        double x1 = -sqrt3By2_ * cellX_[k] / 2;
        double y1 = -(cellX_[k] - guardRingOffset_ / 2);
        double x2 = (-cellX_[k] * guardRingOffset_ + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double xMag = ((-5 * sqrt3_ * cellX_[k] / 21.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-5 * cellX_[k] / 42.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == (HGCalCell::extTrunCellCenCut)) {
        double totalArea = 5 * sqrt3_ * std::pow(cellX_[k], 2) / 8;
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingSizeOffset_) - guardRingOffset_ * guardRingSizeOffset_;
        double cutArea2 = (cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);

        double x1 = -sqrt3_ * cellX_[k] / 4;
        double y1 = (0.5 * cellX_[k] - 0.5 * guardRingOffset_);
        double x2 = (-3 * cellX_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18 -
                     3 * std::pow(cellX_[k], 2) / 8) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] =
            2 * cellAreaPartial[k][HGCalCell::extHalfTrunCell - HGCalCell::partiaclWaferCellsOffset];
        double xMag1 = ((-7 * sqrt3_ * cellX_[k] / 30) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                       (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-2 * cellX_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
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
      } else if (j == (HGCalCell::extExtCellCenCut)) {
        double totalArea = (7.0 * sqrt3_ / 8.0) * std::pow(cellX_[k], 2);
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingSizeOffset_) - guardRingOffset_ * guardRingSizeOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] =
            2 * cellAreaPartial[k][HGCalCell::extHalfExtCell - HGCalCell::partiaclWaferCellsOffset];

        double x1 = -sqrt3By2_ * cellX_[k] / 2;
        double y1 = -(cellX_[k] - guardRingOffset_ / 2);
        double x2 = (-cellX_[k] * guardRingOffset_ + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double xMag = ((-5 * sqrt3_ * cellX_[k] / 21.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-5 * cellX_[k] / 42.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == (HGCalCell::extTrunCellEdgeCut)) {
        double totalArea = 5 * sqrt3_ * std::pow(cellX_[k], 2) / 4;
        double cutArea1 = (sqrt3_ * cellX_[k] * guardRingSizeOffset_) - guardRingOffset_ * guardRingSizeOffset_;
        double cutArea2 = (cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);

        double x1 = -sqrt3_ * cellX_[k] / 4;
        double y1 = (0.5 * cellX_[k] - 0.5 * guardRingOffset_);
        double x2 = (-3 * cellX_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18 -
                     3 * std::pow(cellX_[k], 2) / 8) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2;
        double xMag1 = ((-7 * sqrt3_ * cellX_[k] / 30) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                       (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-2 * cellX_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
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
      } else if (j == (HGCalCell::extExtCellEdgeCut)) {
        double totalArea = (7.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);
        double cutArea1 = (sqrt3_ * cellX_[k] * guardRingSizeOffset_) - guardRingOffset_ * guardRingSizeOffset_;
        double cutArea2 = (1.5 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2;

        double x1 = -sqrt3By2_ * cellX_[k] / 2;
        double y1 = -(cellX_[k] - guardRingOffset_ / 2);
        double x2 = (-cellX_[k] * guardRingOffset_ + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double xMag = ((-5 * sqrt3_ * cellX_[k] / 21.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-5 * cellX_[k] / 42.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == (HGCalCell::fullCellCenCut)) {
        double totalArea = (3.0 * sqrt3_ / 2.0) * std::pow(cellX_[k], 2);
        double cutArea = cellX_[k] * 4.0 * guardRingOffset_ - std::pow(guardRingOffset_, 2) / sqrt3By2_;
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea;
        double x1 = (-cellX_[k] * guardRingOffset_ + 2 * std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / sqrt3_);
        double y1 = 0;
        double xMag = ((-2.0 * sqrt3_ * cellX_[k] / 9.0) * totalArea - (cutArea * x1)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = (0 * totalArea - (cutArea * y1)) / (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == (HGCalCell::fullCellEdgeCut)) {
        double totalArea = (3.0 * sqrt3_ / 2.0) * std::pow(cellX_[k], 2);
        double cutArea = cellX_[k] * guardRingOffset_ - std::pow(guardRingOffset_, 2) / sqrt3_;
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea;
        double x1 = (-cellX_[k] * guardRingOffset_ + 2 * std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / sqrt3_);
        double y1 = 0;
        double xMag = ((-2.0 * sqrt3_ * cellX_[k] / 9.0) * totalArea - (cutArea * x1)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = (0 * totalArea - (cutArea * y1)) / (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == HGCalCell::intTrunCell) {                            // Offset for truncated cells
        double totalArea = (5.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);  // Area of cell without any dead zone
        double cutArea =
            cellX_[k] * sqrt3_ * guardRingOffset_;  // Area of inactive region form guardring and other effects
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea;
        double offMag = (((-2.0 / 15.0) * totalArea * cellX_[k]) - ((cellX_[k] - (0.5 * guardRingOffset_)) * cutArea)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);  // Magnitude of offset
        // (x, y) coordinates of offset for 6 sides of wafer starting from bottom left edge in clockwise direction
        // offset_x = -Offset_magnitude * sin(30 + 60*i) i in (0-6)
        // offset_y = -Offset_magnitude * cos(30 + 60*i) i in (0-6)
        std::array<double, 6> tempOffsetX = {
            {-0.5 * offMag, -offMag, -0.5 * offMag, 0.5 * offMag, offMag, 0.5 * offMag}};
        std::array<double, 6> tempOffsetY = {
            {-sqrt3By2_ * offMag, 0.0, sqrt3By2_ * offMag, sqrt3By2_ * offMag, 0.0, -sqrt3By2_ * offMag}};
        for (int i = 0; i < 6; ++i) {
          offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
          offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
        }
      } else if (j == HGCalCell::intExtCell) {                             //Offset for extended cells
        double totalArea = (7.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);  // Area of cell without any dead zone
        double cutArea =
            cellX_[k] * sqrt3_ * guardRingOffset_;  // Area of inactive region form guardring and other effects
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea;
        double offMag =  // Magnitude of offset
            (((5.0 / 42.0) * totalArea * cellX_[k]) - ((cellX_[k] - (0.5 * guardRingOffset_))) * (cutArea)) /
            (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        // (x, y) coordinates of offset for 6 sides of wafer starting from bottom left edge in clockwise direction
        // offset_x = -Offset_magnitude * sin(30 + 60*i) i in (0-6)
        // offset_y = -Offset_magnitude * cos(30 + 60*i) i in (0-6)
        std::array<double, 6> tempOffsetX = {
            {-0.5 * offMag, -offMag, -0.5 * offMag, 0.5 * offMag, offMag, 0.5 * offMag}};
        std::array<double, 6> tempOffsetY = {
            {-sqrt3By2_ * offMag, 0.0, sqrt3By2_ * offMag, sqrt3By2_ * offMag, 0.0, -sqrt3By2_ * offMag}};
        for (int i = 0; i < 6; ++i) {
          offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetX[i];
          offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = tempOffsetY[i];
        }
      } else if (j == (HGCalCell::intHalfExtCell)) {
        double totalArea = (7.0 * sqrt3_ / 8.0) * std::pow(cellX_[k], 2);
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingOffset_) - guardRingOffset_ * guardRingOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2;
        double x1 = -sqrt3By2_ * cellX_[k] / 2;
        double y1 = -(cellX_[k] - guardRingOffset_ / 2);
        double x2 = (-cellX_[k] * guardRingOffset_ + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double xMag = ((-5 * sqrt3_ * cellX_[k] / 21.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-5 * cellX_[k] / 42.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == (HGCalCell::intHalfTrunCell)) {
        double totalArea = 5 * sqrt3_ * std::pow(cellX_[k], 2) / 8;
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingOffset_) - guardRingOffset_ * guardRingOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        double x1 = -sqrt3_ * cellX_[k] / 4;
        double y1 = (0.5 * cellX_[k] - 0.5 * guardRingOffset_);
        double x2 = (-3 * cellX_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18 -
                     3 * std::pow(cellX_[k], 2) / 8) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2;
        double xMag1 = ((-7 * sqrt3_ * cellX_[k] / 30) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                       (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-2 * cellX_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
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
      } else if (j == (HGCalCell::intTrunCellCenCut)) {
        double totalArea = (7.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingOffset_) - guardRingOffset_ * guardRingOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] =
            2 * cellAreaPartial[k][HGCalCell::intHalfTrunCell - HGCalCell::partiaclWaferCellsOffset];

        double x1 = -sqrt3By2_ * cellX_[k] / 2;
        double y1 = -(cellX_[k] - guardRingOffset_ / 2);
        double x2 = (-cellX_[k] * guardRingOffset_ + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double xMag = ((-5 * sqrt3_ * cellX_[k] / 21.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-5 * cellX_[k] / 42.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == (HGCalCell::intExtCellCenCut)) {
        double totalArea = 5 * sqrt3_ * std::pow(cellX_[k], 2) / 8;
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingOffset_) - guardRingOffset_ * guardRingOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        double x1 = -sqrt3_ * cellX_[k] / 4;
        double y1 = (0.5 * cellX_[k] - 0.5 * guardRingOffset_);
        double x2 = (-3 * cellX_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18 -
                     3 * std::pow(cellX_[k], 2) / 8) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] =
            2 * cellAreaPartial[k][HGCalCell::intHalfExtCell - HGCalCell::partiaclWaferCellsOffset];
        double xMag1 = ((-7 * sqrt3_ * cellX_[k] / 30) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                       (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-2 * cellX_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
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
      } else if (j == (HGCalCell::intTrunCellEdgeCut)) {
        double totalArea = (7.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingOffset_) - guardRingOffset_ * guardRingOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] =
            2 * cellAreaPartial[k][HGCalCell::intHalfTrunCell - HGCalCell::partiaclWaferCellsOffset];

        double x1 = -sqrt3By2_ * cellX_[k] / 2;
        double y1 = -(cellX_[k] - guardRingOffset_ / 2);
        double x2 = (-cellX_[k] * guardRingOffset_ + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18) /
                    (2 * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
        double xMag = ((-5 * sqrt3_ * cellX_[k] / 21.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-5 * cellX_[k] / 42.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
      } else if (j == (HGCalCell::intExtCellEdgeCut)) {
        double totalArea = 5 * sqrt3_ * std::pow(cellX_[k], 2) / 8;
        double cutArea1 = (sqrt3By2_ * cellX_[k] * guardRingOffset_) - guardRingOffset_ * guardRingOffset_;
        double cutArea2 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
        double x1 = -sqrt3_ * cellX_[k] / 4;
        double y1 = (0.5 * cellX_[k] - 0.5 * guardRingOffset_);
        double x2 = (-3 * cellX_[k] * guardRingOffset_ / 4 + std::pow(guardRingOffset_, 2) / (3 * sqrt3_)) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        double y2 = (-cellX_[k] * guardRingOffset_ / (2 * sqrt3_) + std::pow(guardRingOffset_, 2) / 18 -
                     3 * std::pow(cellX_[k], 2) / 8) /
                    (3 * cellX_[k] / 2 - guardRingOffset_ / (2 * sqrt3_));
        cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] =
            2 * cellAreaPartial[k][HGCalCell::intHalfExtCell - HGCalCell::partiaclWaferCellsOffset];
        double xMag1 = ((-7 * sqrt3_ * cellX_[k] / 30) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                       (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
        double yMag = ((-2 * cellX_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                      (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
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
      } else if (j == (HGCalCell::LDPartial0714Cell)) {
        if (k == 1) {
          double totalArea = (9.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);
          double cutArea1 =
              (3 * cellX_[k] * sqrt3By2_ * guardRingSizeOffset_) - (std::pow(guardRingSizeOffset_, 2) / (2 * sqrt3_));
          double cutArea2 = (3 * cellX_[k] * sqrt3By2_ * guardRingOffset_);
          double cutArea3 = (sqrt3_ * std::pow((mouseBiteCut_ - (guardRingSizeOffset_ / sqrt3By2_)), 2) / 2) -
                            (mouseBiteCut_ * guardRingOffset_);
          double x1_0 = ((3.375 * cellX_[k] * cellX_[k]) - (cellX_[k] * 0.75 * guardRingOffset_) +
                         (std::pow(guardRingOffset_, 2) / 18)) /
                        ((3 * cellX_[k] * sqrt3By2_) - (guardRingOffset_ / (2 * sqrt3_)));
          double y1_0 =
              ((3 * cellX_[k] * sqrt3By2_ * guardRingOffset_ / 2) - (std::pow(guardRingOffset_, 2) / (6 * sqrt3_))) /
              ((3 * cellX_[k] * sqrt3By2_) - (guardRingOffset_ / (2 * sqrt3_)));

          double x2_0 = (3 * sqrt3By2_ * cellX_[k] / 2);
          double y2_0 = guardRingOffset_ / 2;

          double x1 = (cellX_[k] / 2 - guardRingOffset_) * sqrt3By2_ + x1_0 * 0.5 + y1_0 * sqrt3By2_;
          double y1 = cellX_[k] + (cellX_[k] / 2 - guardRingOffset_) * 0.5 - x1_0 * sqrt3By2_ + y1_0 * 0.5;

          double x2 = x2_0 - sqrt3By2_ * cellX_[k];
          double y2 = -(cellX_[k] - y2_0);

          double x3 = sqrt3_ * cellX_[k] - mouseBiteCut_ + (mouseBiteCut_ - (guardRingOffset_ / sqrt3By2_)) / 3;
          double y3 = -(cellX_[k] - sqrt3_ * (mouseBiteCut_ - (guardRingOffset_ / sqrt3By2_)) / 3 - guardRingOffset_);

          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag = ((sqrt3_ * cellX_[k] / 8) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
          double yMag = ((-1 * cellX_[k] / 8) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = 0.0;
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial0209Cell)) {
        if (k == 1) {
          double totalArea = (23.0 * sqrt3_ / 8.0) * std::pow(cellX_[k], 2);
          double cutArea1 =
              (5 * cellX_[k] * sqrt3By2_ * guardRingSizeOffset_) - (std::pow(guardRingSizeOffset_, 2) / (2 * sqrt3_));
          double cutArea2 = (4 * cellX_[k] * guardRingOffset_) - (2 * guardRingSizeOffset_ * guardRingOffset_) +
                            (1 / sqrt3By2_ * (std::pow(guardRingOffset_, 2)));
          double cutArea3 =
              std::pow(mouseBiteCut_, 2) / sqrt3_ -
              (2 * mouseBiteCut_ - sqrt3_ * guardRingSizeOffset_) * guardRingSizeOffset_ -
              (1 / sqrt3By2_ * mouseBiteCut_ - 2 * guardRingSizeOffset_ - guardRingOffset_ / sqrt3_) * guardRingOffset_;

          double x1_0 = (9.375 * cellX_[k] * cellX_[k] - (cellX_[k] * 1.25 * guardRingOffset_) +
                         (std::pow(guardRingOffset_, 2) / 18)) /
                        ((5 * cellX_[k] * sqrt3By2_) - (guardRingOffset_ / (2 * sqrt3_)));
          double y1_0 =
              ((5 * cellX_[k] * sqrt3By2_ * guardRingOffset_ / 2) - (std::pow(guardRingOffset_, 2) / (6 * sqrt3_))) /
              ((5 * cellX_[k] * sqrt3By2_) - (guardRingOffset_ / (2 * sqrt3_)));

          double x1 = (1.5 * cellX_[k]) * sqrt3By2_ - x1_0 * 0.5 - y1_0 * sqrt3By2_;
          double y1 = -0.25 * cellX_[k] + x1_0 * sqrt3By2_ - y1_0 * 0.5;
          double x2 = -(sqrt3By2_ * cellX_[k] - 0.5 * guardRingOffset_);
          double y2 = 1.5 * cellX_[k];
          double x3 = -(sqrt3By2_ * cellX_[k] - mouseBiteCut_ / 3);
          double y3 = 3.5 * cellX_[k] - (5 * mouseBiteCut_) / 3 * sqrt3_;

          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag =
              ((-9 * cellX_[k] / (sqrt3_ * 92)) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
              (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
          double yMag =
              ((199 * cellX_[k] / (sqrt3_ * 276)) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
              (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = 0.0;
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial0007Cell)) {
        if (k == 1) {
          double totalArea = (5.0 * sqrt3_ / 4.0) * std::pow(cellX_[k], 2);
          double cutArea1 = (cellX_[k] * guardRingOffset_);
          double cutArea2 = (sqrt3_ * cellX_[k] - guardRingOffset_) * guardRingSizeOffset_;
          double h = 2 * mouseBiteCut_ - sqrt3_ * guardRingSizeOffset_ - guardRingOffset_;
          double cutArea3 = 1 / (2 * sqrt3_) * std::pow(h, 2) / 2;

          double x1 = cellX_[k] * sqrt3By2_ - guardRingOffset_ / 2;
          double y1 = 0;
          double x2 = 0;
          double y2 = 0.5 * cellX_[k] - guardRingOffset_ / 2;
          double x3 = sqrt3By2_ * cellX_[k] - guardRingOffset_ - h / sqrt3_;
          double y3 = 0.5 * cellX_[k] - guardRingOffset_ - h / 3;

          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag = ((0.0) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
          double yMag = ((-2 * cellX_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = 0.0;
          for (int i = 0; i < 6; ++i) {
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial0815Cell)) {
        if (k == 1) {
          double totalArea = sqrt3_ * std::pow(cellX_[k], 2);
          double cutArea1 = (sqrt3_ * cellX_[k] * guardRingSizeOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
          ;
          double cutArea2 = (sqrt3_ * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_) -
                            guardRingOffset_ * guardRingSizeOffset_ / sqrt3By2_;
          double cutArea3 =
              sqrt3_ * std::pow((mouseBiteCut_ - guardRingSizeOffset_ / sqrt3By2_ - guardRingOffset_ / sqrt3_), 2) / 2;

          double x2_0 = (1.5 * cellX_[k] * cellX_[k] - (0.5 * cellX_[k] * guardRingOffset_) +
                         std::pow(guardRingOffset_, 2) / 18) /
                        (sqrt3_ * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
          double y2_0 = (sqrt3By2_ * cellX_[k] * guardRingOffset_ - std::pow(guardRingOffset_, 2) / (sqrt3_ * 3)) /
                        (sqrt3_ * cellX_[k] - guardRingOffset_ / (2 * sqrt3_));
          double x1 = 0;
          double y1 = 0.5 * cellX_[k] - guardRingOffset_ / 2;
          double x2 = x2_0 * 0.5 - y2_0 * sqrt3By2_;
          double y2 = -(cellX_[k] - (x2_0 * sqrt3By2_ + y2_0 * 0.5));
          double x3 = sqrt3By2_ * cellX_[k] - mouseBiteCut_ + (mouseBiteCut_ - guardRingOffset_ / sqrt3By2_) / 3;
          double y3 = cellX_[k] - (mouseBiteCut_ - guardRingOffset_ / sqrt3By2_) / sqrt3_ - guardRingOffset_;

          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag = -((-sqrt3_ * cellX_[k] / 9) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
          double yMag = ((-cellX_[k] / 15) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
            cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = 0.0;
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial1415Cell)) {
        if (k == 1) {
          double totalArea = 7 * sqrt3_ * std::pow(cellX_[k], 2) / 4;
          double cutArea1 = (3 * cellX_[k] - 2 * guardRingSizeOffset_ - guardRingOffset_ / sqrt3_) * guardRingOffset_;
          double cutArea2 =
              (2 * sqrt3_ * cellX_[k] * guardRingSizeOffset_) - std::pow(guardRingSizeOffset_, 2) * sqrt3By2_;
          double cutArea3 =
              std::pow(mouseBiteCut_, 2) / sqrt3_ -
              (2 * mouseBiteCut_ - sqrt3_ * guardRingSizeOffset_) * guardRingSizeOffset_ -
              (1 / sqrt3By2_ * mouseBiteCut_ - 2 * guardRingSizeOffset_ - guardRingOffset_ / sqrt3_) * guardRingOffset_;

          double x2_0 = (6 * cellX_[k] * cellX_[k] - std::pow(guardRingOffset_, 2)) /
                        (2 * sqrt3_ * cellX_[k] - guardRingOffset_ * sqrt3By2_);
          double y2_0 = (sqrt3_ * cellX_[k] * guardRingOffset_ - std::pow(guardRingOffset_, 2) / sqrt3_) /
                        (2 * sqrt3_ * cellX_[k] - guardRingOffset_ * sqrt3By2_);
          double x1 = -sqrt3By2_ * cellX_[k] + guardRingOffset_ / 2;
          double y1 = -cellX_[k];
          double x2 = sqrt3By2_ * cellX_[k] - x2_0 * 0.5 - y2_0 * sqrt3By2_;
          double y2 = 0.5 * cellX_[k] - x2_0 * sqrt3By2_ + y2_0 * 0.5;
          double h = (mouseBiteCut_ - guardRingOffset_) / sqrt3By2_;
          double x3 = -(cellX_[k] - h / 3 - guardRingOffset_) * sqrt3By2_;
          double y3 = 5 * h / 6 - 5 * cellX_[k] / 2;

          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag =
              ((-2 * cellX_[k] / (7 * sqrt3_)) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
              (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
          double yMag = ((-cellX_[k] / 3) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
            cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = 0.0;
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::LDPartial1515Cell)) {
        if (k == 1) {
          double totalArea = 7 * sqrt3_ * std::pow(cellX_[k], 2) / 8;
          double cutArea1 = (2 * cellX_[k] * guardRingOffset_) - std::pow(guardRingOffset_, 2) / (2 * sqrt3_);
          double cutArea2 = (sqrt3By2_ * cellX_[k] - guardRingOffset_) * guardRingSizeOffset_;
          double cutArea3 = mouseBiteCut_ * cellX_[k] - sqrt3_ * std::pow(cellX_[k], 2) / 8 -
                            (sqrt3By2_ * cellX_[k] - guardRingOffset_) * guardRingSizeOffset_ -
                            (1 / sqrt3By2_ * mouseBiteCut_ - guardRingOffset_ / (2 * sqrt3_)) * guardRingOffset_;

          double x1 = -guardRingOffset_ / 2;
          double y1 = 0;
          double x2 = -(sqrt3By2_ * cellX_[k] / 2);
          double y2 = -(cellX_[k] - 0.5 * guardRingOffset_);
          double x3 = (cellX_[k] * cellX_[k] / 8 - sqrt3_ * cellX_[k] * (mouseBiteCut_ - guardRingOffset_) / 4) /
                      ((mouseBiteCut_ - guardRingOffset_) - sqrt3_ * cellX_[k] / 8);
          double y3 =
              (std::pow((mouseBiteCut_ - guardRingOffset_), 2) / sqrt3_ -
               (1.25 * cellX_[k] * (mouseBiteCut_ - guardRingOffset_)) + 7 * sqrt3_ * cellX_[k] * cellX_[k] / 48) /
              ((mouseBiteCut_ - guardRingOffset_) - sqrt3_ * cellX_[k] / 8);

          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2 - cutArea3;
          double xMag = (-(cellX_[k] / (sqrt3_)) * totalArea - (cutArea1 * x1) - (cutArea2 * x2) - (cutArea3 * x3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
          double yMag = ((-5 * cellX_[k] / 42) * totalArea - (cutArea1 * y1) - (cutArea2 * y2) - (cutArea3 * y3)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
            cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = 0.0;
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::HDPartial0920Cell)) {
        if (k == 0) {
          double totalArea = 37 * sqrt3_ * std::pow(cellX_[k], 2) / 24;
          double cutArea1 =
              (4 * cellX_[k] - 2 * guardRingSizeOffset_ - 0.5 * guardRingOffset_) * guardRingOffset_ / sqrt3_;
          double cutArea2 =
              (7 * cellX_[k] * guardRingSizeOffset_) / (2 * sqrt3_) - std::pow(guardRingSizeOffset_, 2) / (2 * sqrt3_);

          double x1 = cellX_[k] / (2 * sqrt3_);
          double y1 = -(0.5 * cellX_[k] - 0.5 * guardRingOffset_);
          double x2_0 = ((2.041 * cellX_[k] * cellX_[k]) - (cellX_[k] * 0.583 * guardRingOffset_) +
                         (std::pow(guardRingOffset_, 2) / 18)) /
                        ((7 * cellX_[k] / (2 * sqrt3_)) - (guardRingOffset_ / (2 * sqrt3_)));
          double y2_0 =
              ((7 * cellX_[k] * guardRingOffset_ / (4 * sqrt3_)) - std::pow(guardRingOffset_, 2) / (6 * sqrt3_)) /
              ((7 * cellX_[k] / (2 * sqrt3_)) - (guardRingOffset_ / (2 * sqrt3_)));

          double x2 = (0.5 * x2_0) - (sqrt3By2_ * y2_0) + (cellX_[k] * 0.5 * sqrt3By2_);
          double y2 = -(0.5 * y2_0) - (sqrt3By2_ * x2_0) + (cellX_[k] * 1.25);
          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2;
          double xMag = ((25 * sqrt3_ * cellX_[k] / 148) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
          double yMag = ((73 * cellX_[k] / 444) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
            cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = 0.0;
            offsetPartialX[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
            offsetPartialY[k][j - HGCalCell::partiaclWaferCellsOffset][i] = 0.0;
          }
        }
      } else if (j == (HGCalCell::HDPartial1021Cell)) {
        if (k == 0) {
          double totalArea = 11 * sqrt3_ * std::pow(cellX_[k], 2) / 6;
          double cutArea1 = (5 * cellX_[k] * guardRingSizeOffset_ - std::pow(guardRingSizeOffset_, 2)) / (2 * sqrt3_);
          double cutArea2 = (5 * cellX_[k] * guardRingOffset_) / (2 * sqrt3_) -
                            std::pow(guardRingOffset_, 2) / (2 * sqrt3_) -
                            guardRingOffset_ * guardRingSizeOffset_ / sqrt3By2_;

          double x1 = -cellX_[k] / (4 * sqrt3_);
          double y1 = cellX_[k] - 0.5 * guardRingOffset_;
          double x2_0 = ((1.041 * cellX_[k] * cellX_[k]) - (cellX_[k] * 0.416 * guardRingOffset_) +
                         (std::pow(guardRingOffset_, 2) / 18.0)) /
                        ((5.0 * cellX_[k] / (2.0 * sqrt3_)) - (guardRingOffset_ / (2.0 * sqrt3_)));
          double y2_0 =
              ((5.0 * cellX_[k] * guardRingOffset_ / (4.0 * sqrt3_)) - std::pow(guardRingOffset_, 2) / (6 * sqrt3_)) /
              ((5.0 * cellX_[k] / (2.0 * sqrt3_)) - (guardRingOffset_ / (2.0 * sqrt3_)));

          double x2 = -(0.5 * x2_0) + (sqrt3By2_ * y2_0) + (cellX_[k] * 1.5 * sqrt3By2_);
          double y2 = -(0.5 * y2_0) + (sqrt3By2_ * x2_0) - cellX_[k];
          cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = totalArea - cutArea1 - cutArea2;
          double xMag = ((47.0 * cellX_[k] / (528.0 * sqrt3_)) * totalArea - (cutArea1 * x1) - (cutArea2 * x2)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);
          double yMag = ((47.0 * cellX_[k] / 528.0) * totalArea - (cutArea1 * y1) - (cutArea2 * y2)) /
                        (cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset]);

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
            cellAreaPartial[k][j - HGCalCell::partiaclWaferCellsOffset] = 0.0;
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
    } else if ((cellType == HGCalCell::halfCell) || (cellType == HGCalCell::LDPartial0714Cell) ||
               (cellType == HGCalCell::LDPartial0815Cell) || (cellType == HGCalCell::HDPartial0920Cell) ||
               (cellType == HGCalCell::HDPartial1021Cell)) {
      int cellType1 = cellType - HGCalCell::partiaclWaferCellsOffset;
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

double HGCalCellOffset::cellAreaUV(
    int32_t u, int32_t v, int32_t placementIndex, int32_t type, int32_t partialType, bool reco) {
  if (type != 0)
    type = 1;
  double area(0);
  area = cellAreaUV(u, v, placementIndex, type, reco);
  std::pair<int, int> cell = hgcalcell_->cellType(u, v, ncell_[type], placementIndex, partialType);
  int cellPos = cell.first;
  int cellType = cell.second;
  if ((cellType >= HGCalCell::partiaclWaferCellsOffset) || (cellPos >= HGCalCell::partiaclCellsPosOffset)) {
    if (cellType == HGCalCell::truncatedCell || cellType == HGCalCell::extendedCell) {
      area = reco ? cellArea[type][cellType] : HGCalParameters::k_ScaleToDDD2 * cellArea[type][cellType];
    } else {
      area = reco ? cellAreaPartial[type][cellType - HGCalCell::partiaclWaferCellsOffset]
                  : HGCalParameters::k_ScaleToDDD2 * cellArea[type][cellType - HGCalCell::partiaclWaferCellsOffset];
    }
  }
  return area;
}
