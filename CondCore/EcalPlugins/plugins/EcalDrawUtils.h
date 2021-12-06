#include "TH2F.h"
#include "TLine.h"

inline void DrawEB(TH2F* ebmap, float min, float max) {
  ebmap->SetXTitle("i#phi");
  ebmap->SetYTitle("i#eta");
  ebmap->GetXaxis()->SetNdivisions(-418, kFALSE);
  ebmap->GetYaxis()->SetNdivisions(-1702, kFALSE);
  ebmap->GetXaxis()->SetLabelSize(0.03);
  ebmap->GetYaxis()->SetLabelSize(0.03);
  ebmap->GetXaxis()->SetTickLength(0.01);
  ebmap->GetYaxis()->SetTickLength(0.01);
  ebmap->SetMaximum(max);
  ebmap->SetMinimum(min);
  ebmap->Draw("colz");
  TLine* l = new TLine;
  l->SetLineWidth(1);
  for (int i = 0; i < 17; i++) {
    Double_t x = 20. + (i * 20);
    l = new TLine(x, -85., x, 85.);
    l->Draw();
  }
  l = new TLine(0., 85., 360., 85.);
  l->Draw();
  l = new TLine(0., 0., 360., 0.);
  l->Draw();
}  //   DrawEB method

inline void DrawEE(TH2F* endc, float min, float max) {
  int ixSectorsEE[202] = {
      62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 56, 56, 46, 46, 44, 44, 43, 43, 42, 42,  41,  41,  40,  40, 41, 41,
      42, 42, 43, 43, 44, 44, 46, 46, 56, 56, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62,  0,   101, 101, 98, 98, 96,
      96, 93, 93, 88, 88, 86, 86, 81, 81, 76, 76, 66, 66, 61, 61, 41, 41, 36, 36, 26,  26,  21,  21,  16, 16, 14,
      14, 9,  9,  6,  6,  4,  4,  1,  1,  4,  4,  6,  6,  9,  9,  14, 14, 16, 16, 21,  21,  26,  26,  36, 36, 41,
      41, 61, 61, 66, 66, 76, 76, 81, 81, 86, 86, 88, 88, 93, 93, 96, 96, 98, 98, 101, 101, 0,   62,  66, 66, 71,
      71, 81, 81, 91, 91, 93, 0,  62, 66, 66, 91, 91, 98, 0,  58, 61, 61, 66, 66, 71,  71,  76,  76,  81, 81, 0,
      51, 51, 0,  44, 41, 41, 36, 36, 31, 31, 26, 26, 21, 21, 0,  40, 36, 36, 11, 11,  4,   0,   40,  36, 36, 31,
      31, 21, 21, 11, 11, 9,  0,  46, 46, 41, 41, 36, 36, 0,  56, 56, 61, 61, 66, 66};

  int iySectorsEE[202] = {51, 56, 56, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 61, 61, 60, 60, 59, 59, 58,  58,  56,  56,
                          46, 46, 44, 44, 43, 43, 42, 42, 41, 41, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44,  46,  46,  51,
                          0,  51, 61, 61, 66, 66, 76, 76, 81, 81, 86, 86, 88, 88, 93, 93, 96, 96, 98, 98,  101, 101, 98,
                          98, 96, 96, 93, 93, 88, 88, 86, 86, 81, 81, 76, 76, 66, 66, 61, 61, 41, 41, 36,  36,  26,  26,
                          21, 21, 16, 16, 14, 14, 9,  9,  6,  6,  4,  4,  1,  1,  4,  4,  6,  6,  9,  9,   14,  14,  16,
                          16, 21, 21, 26, 26, 36, 36, 41, 41, 51, 0,  46, 46, 41, 41, 36, 36, 31, 31, 26,  26,  0,   51,
                          51, 56, 56, 61, 61, 0,  61, 61, 66, 66, 71, 71, 76, 76, 86, 86, 88, 0,  62, 101, 0,   61,  61,
                          66, 66, 71, 71, 76, 76, 86, 86, 88, 0,  51, 51, 56, 56, 61, 61, 0,  46, 46, 41,  41,  36,  36,
                          31, 31, 26, 26, 0,  40, 31, 31, 16, 16, 6,  0,  40, 31, 31, 16, 16, 6};

  TLine* l = new TLine;
  l->SetLineWidth(1);

  endc->SetXTitle("ix");
  endc->SetYTitle("iy");
  endc->SetMaximum(max);
  endc->SetMinimum(min);
  endc->Draw("colz1");
  for (int i = 0; i < 201; i = i + 1) {
    if ((ixSectorsEE[i] != 0 || iySectorsEE[i] != 0) && (ixSectorsEE[i + 1] != 0 || iySectorsEE[i + 1] != 0)) {
      l->DrawLine(ixSectorsEE[i], iySectorsEE[i], ixSectorsEE[i + 1], iySectorsEE[i + 1]);
    }
  }
}  //   DrawEE method

inline void DrawEE_Tower(TH2F* endc, TLine* l, double minScale, double maxScale) {
  endc->SetStats(false);
  endc->SetMinimum(minScale);
  endc->SetMaximum(maxScale);
  endc->Draw("colz");

  int ixSectorsEE[136] = {8,  14, 14, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 14, 14,
                          8,  8,  5,  5,  4,  4,  3,  3,  2,  2,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  8,  8,  8,
                          9,  9,  10, 10, 12, 12, 13, 13, 12, 12, 10, 10, 9,  9,  10, 10, 0,  11, 11, 0,  10, 9,  9,
                          8,  8,  7,  7,  6,  6,  5,  5,  0,  12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 0,  9,  8,
                          8,  3,  3,  1,  0,  13, 14, 14, 19, 19, 21, 0,  9,  8,  8,  7,  7,  5,  5,  3,  3,  2,  0,
                          13, 14, 14, 15, 15, 17, 17, 19, 19, 20, 0,  14, 14, 13, 13, 12, 12, 0};

  int iySectorsEE[136] = {1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  8,  8,  14, 14, 17, 17, 18, 18, 19, 19, 20, 20, 21,
                          21, 20, 20, 19, 19, 18, 18, 17, 17, 14, 14, 8,  8,  5,  5,  4,  4,  3,  3,  2,  2,  1,  4,
                          4,  7,  7,  9,  9,  10, 10, 12, 12, 13, 13, 12, 12, 10, 10, 9,  0,  13, 21, 0,  13, 13, 14,
                          14, 15, 15, 16, 16, 18, 18, 19, 0,  13, 13, 14, 14, 15, 15, 16, 16, 18, 18, 19, 0,  11, 11,
                          12, 12, 13, 13, 0,  11, 11, 12, 12, 13, 13, 0,  10, 10, 9,  9,  8,  8,  7,  7,  6,  6,  0,
                          10, 10, 9,  9,  8,  8,  7,  7,  6,  6,  0,  2,  4,  4,  7,  7,  9,  0};

  for (int i = 0; i < 136; i = i + 1)
    if ((ixSectorsEE[i] != 0 || iySectorsEE[i] != 0) && (ixSectorsEE[i + 1] != 0 || iySectorsEE[i + 1] != 0))
      l->DrawLine(ixSectorsEE[i], iySectorsEE[i], ixSectorsEE[i + 1], iySectorsEE[i + 1]);

}  //draw EE in case of a tower

inline void drawTable(int nbRows, int nbColumns) {
  TLine* l = new TLine;
  l->SetLineWidth(1);
  for (int i = 1; i < nbRows; i++) {
    double y = (double)i;
    l = new TLine(0., y, nbColumns, y);
    l->Draw();
  }

  for (int i = 1; i < nbColumns; i++) {
    double x = (double)i;
    double y = (double)nbRows;
    l = new TLine(x, 0., x, y);
    l->Draw();
  }
}