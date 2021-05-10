/*
 * Authors: William Nash (original), Sven Dildick (adapted)
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCPatternBank.h"

#include <TCanvas.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TSystem.h>
#include <TH1F.h>
#include <TFile.h>
#include <TString.h>

//c++
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <memory>
#include <bitset>

using namespace std;

// uncertainty per layer in half strips 1/sqrt(12)
const float HALF_STRIP_ERROR = 0.288675;
const unsigned halfpatternwidth = (CSCConstants::CLCT_PATTERN_WIDTH - 1) / 2;

//labels of all envelopes
const int DEBUG = 0;

// forward declarations
class CSCPattern {
public:
  CSCPattern(unsigned int id, const CSCPatternBank::LCTPattern& pat);
  ~CSCPattern() {}

  void printCode(const unsigned code) const;
  void getLayerPattern(const unsigned code, unsigned layerPattern[CSCConstants::NUM_LAYERS]) const;
  int recoverPatternCCCombination(const unsigned code,
                                  int code_hits[CSCConstants::NUM_LAYERS][CSCConstants::CLCT_PATTERN_WIDTH]) const;
  string getName() const { return name_; }
  unsigned getId() const { return id_; }

private:
  const unsigned int id_;
  string name_;
  CSCPatternBank::LCTPattern pat_;
};

CSCPattern::CSCPattern(unsigned int id, const CSCPatternBank::LCTPattern& pat)
    : id_(id), name_(std::to_string(id)), pat_(pat) {}

//given a code, prints out how it looks within the pattern
void CSCPattern::printCode(const unsigned code) const {
  // comparator code per layer
  unsigned layerPattern[CSCConstants::NUM_LAYERS];
  getLayerPattern(code, layerPattern);

  std::cout << "Pattern " << id_ << ", Code " << code << " " << std::bitset<12>(code) << std::endl;
  for (unsigned int j = 0; j < CSCConstants::NUM_LAYERS; j++) {
    unsigned trueCounter = 0;  //for each layer, should only have 3
    std::cout << "L" << j + 1 << ": ";
    for (unsigned int i = 0; i < CSCConstants::CLCT_PATTERN_WIDTH; i++) {
      if (!pat_[j][i]) {
        printf("-");
      } else {
        trueCounter++;
        if (trueCounter == layerPattern[j])
          printf("X");
        else
          printf("_");
      }
    }
    std::cout << " -> " << std::bitset<2>(layerPattern[j]) << std::endl;
  }

  printf("    ");
  for (unsigned int i = 0; i < CSCConstants::CLCT_PATTERN_WIDTH; i++) {
    printf("%1x", i);
  }
  printf("\n");
}

void CSCPattern::getLayerPattern(const unsigned code, unsigned layerPattern[CSCConstants::NUM_LAYERS]) const {
  // 12-bit comparator code
  std::bitset<12> cc(code);

  for (unsigned ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++) {
    std::bitset<2> cclayer;
    cclayer[1] = cc[2 * ilayer + 1];
    cclayer[0] = cc[2 * ilayer];
    layerPattern[ilayer] = cclayer.to_ulong();
  }
}

//fills "code_hits" with how the code "code" would look inside the pattern
int CSCPattern::recoverPatternCCCombination(
    const unsigned code, int code_hits[CSCConstants::NUM_LAYERS][CSCConstants::CLCT_PATTERN_WIDTH]) const {
  // comparator code per layer
  unsigned layerPattern[CSCConstants::NUM_LAYERS];
  getLayerPattern(code, layerPattern);

  // now set the bits in the pattern
  for (unsigned int j = 0; j < CSCConstants::NUM_LAYERS; j++) {
    unsigned trueCounter = 0;  //for each layer, should only have 3
    for (unsigned int i = 0; i < CSCConstants::CLCT_PATTERN_WIDTH; i++) {
      // zeros in the pattern envelope, or CC0
      if (!pat_[j][i]) {
        code_hits[j][i] = 0;
      }
      // ones in the pattern envelope
      else {
        trueCounter++;
        if (trueCounter == layerPattern[j])
          code_hits[j][i] = 1;
        else
          code_hits[j][i] = 0;
      }
    }
  }
  return 0;
}

int convertToLegacyPattern(const int code_hits[CSCConstants::NUM_LAYERS][CSCConstants::CLCT_PATTERN_WIDTH],
                           unsigned N_LAYER_REQUIREMENT);
void getErrors(const vector<float>& x, const vector<float>& y, float& sigmaM, float& sigmaB);
void writeHeaderPosOffsetLUT(ofstream& file);
void writeHeaderSlopeLUT(ofstream& file);
unsigned firmwareWord(const unsigned quality, const unsigned slope, const unsigned offset);
void setDataWord(unsigned& word, const unsigned newWord, const unsigned shift, const unsigned mask);
unsigned assignPosition(const float fvalue, const float fmin, const float fmax, const unsigned nbits);
unsigned assignBending(const float fvalue, const float fmin, const float fmax, const unsigned nbits);

int CCLUTLinearFitWriter(unsigned N_LAYER_REQUIREMENT = 3) {
  //all the patterns we will fit
  std::unique_ptr<std::vector<CSCPattern>> newPatterns(new std::vector<CSCPattern>());
  for (unsigned ipat = 0; ipat < 5; ipat++) {
    newPatterns->emplace_back(ipat, CSCPatternBank::clct_pattern_run3_[ipat]);
  }

  // create output directory
  const std::string outdir("output_" + std::to_string(N_LAYER_REQUIREMENT) + "layers/");

  // output ROOT file with fit results
  TFile* file = TFile::Open(TString::Format("figures_%dlayers/fitresults.root", N_LAYER_REQUIREMENT), "RECREATE");

  TH1D* all_offsets = new TH1D("all_offsets", "All Half-strip offsets", 200, -2.5, 2.5);
  all_offsets->GetYaxis()->SetTitle("Entries");
  all_offsets->GetXaxis()->SetTitle("Half-strip offset");

  TH1D* all_slopes = new TH1D("all_slopes", "All slopes", 200, -3, 3);
  all_slopes->GetYaxis()->SetTitle("Entries");
  all_slopes->GetXaxis()->SetTitle("Slope [half-strips/layer]");

  std::map<unsigned, TH1D*> offsets;
  std::map<unsigned, TH1D*> slopes;
  std::map<unsigned, TH1D*> offsetuncs;
  std::map<unsigned, TH1D*> slopeuncs;
  std::map<unsigned, TH1D*> chi2s;
  std::map<unsigned, TH1D*> chi2ndfs;
  std::map<unsigned, TH1D*> layers;
  std::map<unsigned, TH1D*> legacypatterns;

  for (unsigned i = 0; i < 5; ++i) {
    std::string title("Half-strip offset, PID " + std::to_string(i));
    std::string name("offset_pat" + std::to_string(i));

    offsets[i] = new TH1D(name.c_str(), title.c_str(), 100, -2.5, 2.5);
    offsets[i]->GetYaxis()->SetTitle("Entries");
    offsets[i]->GetXaxis()->SetTitle("Half-strip offset");

    title = "Slope, PID " + std::to_string(i);
    name = "slope_pat" + std::to_string(i);
    slopes[i] = new TH1D(name.c_str(), title.c_str(), 100, -3, 3);
    slopes[i]->GetYaxis()->SetTitle("Entries");
    slopes[i]->GetXaxis()->SetTitle("Slope [half-strips/layer]");

    title = "Half-strip offset uncertainty, PID " + std::to_string(i);
    name = "offsetunc_pat" + std::to_string(i);
    offsetuncs[i] = new TH1D(name.c_str(), title.c_str(), 100, -2, 2);
    offsetuncs[i]->GetYaxis()->SetTitle("Entries");
    offsetuncs[i]->GetXaxis()->SetTitle("Half-strip offset uncertainty");

    title = "Slope uncertainty, PID " + std::to_string(i);
    name = "slopeunc_pat" + std::to_string(i);
    slopeuncs[i] = new TH1D(name.c_str(), title.c_str(), 100, -2, 2);
    slopeuncs[i]->GetYaxis()->SetTitle("Entries");
    slopeuncs[i]->GetXaxis()->SetTitle("Slope uncertainty [half-strips/layer]");

    title = "Chi2, PID " + std::to_string(i);
    name = "chi2_pat" + std::to_string(i);
    chi2s[i] = new TH1D(name.c_str(), title.c_str(), 100, 0, 100);
    chi2s[i]->GetYaxis()->SetTitle("Entries");
    chi2s[i]->GetXaxis()->SetTitle("#chi^{2}");

    title = "Chi2/ndf, PID " + std::to_string(i);
    name = "chi2ndf_pat" + std::to_string(i);
    chi2ndfs[i] = new TH1D(name.c_str(), title.c_str(), 40, 0, 20);
    chi2ndfs[i]->GetYaxis()->SetTitle("Entries");
    chi2ndfs[i]->GetXaxis()->SetTitle("#chi^{2}/NDF");

    title = "Number of layers, PID " + std::to_string(i);
    name = "layers_pat" + std::to_string(i);
    layers[i] = new TH1D(name.c_str(), title.c_str(), 7, 0, 7);
    layers[i]->GetYaxis()->SetTitle("Entries");
    layers[i]->GetXaxis()->SetTitle("Number of layers in pattern");

    title = "Run-1/2 pattern, PID " + std::to_string(i);
    name = "legacypatterns_pat" + std::to_string(i);
    legacypatterns[i] = new TH1D(name.c_str(), title.c_str(), 11, 0, 11);
    legacypatterns[i]->GetYaxis()->SetTitle("Entries");
    legacypatterns[i]->GetXaxis()->SetTitle("Equivalent Run-1/2 pattern number");
  }

  for (auto& p : offsets) {
    (p.second)->GetYaxis()->SetTitle("Entries");
  }

  //Used to calculate span of position offsets
  float maxOffset = -1;
  float maxPatt = -1;
  float maxCode = -1;
  float minOffset = -2;
  float minPatt = 0;
  float minCode = 1;

  // for each pattern
  for (auto patt = newPatterns->begin(); patt != newPatterns->end(); ++patt) {
    std::cout << "Processing pattern " << patt - newPatterns->begin() << std::endl;

    // create 3 output files per pattern: 1) position offset for CMSSW, 2) slope for CMSSW, 3) output for firmware
    std::cout << "Create output files for pattern " << patt->getName() << std::endl;

    // floating point
    ofstream outoffset_sw;
    outoffset_sw.open(outdir + "CSCComparatorCodePosOffsetLUT_pat" + patt->getName() + "_float_v1.txt");
    writeHeaderPosOffsetLUT(outoffset_sw);

    ofstream outslope_sw;
    outslope_sw.open(outdir + "CSCComparatorCodeSlopeLUT_pat" + patt->getName() + "_float_v1.txt");
    writeHeaderSlopeLUT(outslope_sw);

    // unsigned
    ofstream outoffset_sw_bin;
    outoffset_sw_bin.open(outdir + "CSCComparatorCodePosOffsetLUT_pat" + patt->getName() + "_v1.txt");
    writeHeaderPosOffsetLUT(outoffset_sw_bin);

    ofstream outslope_sw_bin;
    outslope_sw_bin.open(outdir + "CSCComparatorCodeSlopeLUT_pat" + patt->getName() + "_v1.txt");
    writeHeaderSlopeLUT(outslope_sw_bin);

    // format: [8:0] is quality, [13, 9] is bending , [17:14] is offset
    ofstream outfile_fw;
    outfile_fw.open(outdir + "rom_pat" + patt->getName() + ".mem");

    // pattern conversions
    ofstream outpatternconv;
    outpatternconv.open(outdir + "CSCComparatorCodePatternConversionLUT_pat" + patt->getName() + "_v1.txt");

    // iterate through each possible comparator code
    for (unsigned code = 0; code < CSCConstants::NUM_COMPARATOR_CODES; code++) {
      if (DEBUG > 0) {
        cout << "Evaluating..." << endl;
        patt->printCode(code);
      }
      int hits[CSCConstants::NUM_LAYERS][CSCConstants::CLCT_PATTERN_WIDTH];

      if (patt->recoverPatternCCCombination(code, hits)) {
        cout << "Error: CC evaluation has failed" << endl;
      }

      //put the coordinates in the hits into a vector
      // x = layer, y = position in that layer
      vector<float> x;
      vector<float> y;
      vector<float> xe;
      vector<float> ye;

      for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
        for (unsigned j = 0; j < CSCConstants::CLCT_PATTERN_WIDTH; j++) {
          if (hits[i][j]) {
            //shift to key half strip layer (layer 3)
            x.push_back(i - 2);
            y.push_back(float(j));
            xe.push_back(float(0));
            ye.push_back(HALF_STRIP_ERROR);
            if (DEBUG > 0)
              cout << "L " << x.back() << " HS " << y.back() << endl;
          }
        }
        if (DEBUG > 0)
          cout << endl;
      }

      // fit results
      float offset = 0;
      float slope = 0;
      float offsetunc = 0;
      float slopeunc = 0;
      float chi2 = -1;
      unsigned ndf;
      unsigned layer = 0;
      unsigned legacypattern = 0;

      // consider at least patterns with 3 hits
      if (x.size() >= N_LAYER_REQUIREMENT) {
        // for each combination fit a straight line
        std::unique_ptr<TGraphErrors> gr(new TGraphErrors(x.size(), &x[0], &y[0], &xe[0], &ye[0]));
        std::unique_ptr<TF1> fit(new TF1("fit", "pol1", -3, 4));
        gr->Fit("fit", "EMQ");

        // fit results
        // center at the key half-strip
        offset = fit->GetParameter(0) - halfpatternwidth;
        slope = fit->GetParameter(1);
        chi2 = fit->GetChisquare();
        ndf = fit->GetNDF();
        layer = ndf + 2;
        legacypattern = convertToLegacyPattern(hits, N_LAYER_REQUIREMENT);
        // mean half-strip error; slope half-strip error
        getErrors(x, y, offsetunc, slopeunc);

        // save fit results
        const unsigned ipat(patt - newPatterns->begin());
        offsets[ipat]->Fill(offset);
        slopes[ipat]->Fill(slope);
        offsetuncs[ipat]->Fill(offsetunc);
        slopeuncs[ipat]->Fill(slopeunc);
        chi2s[ipat]->Fill(chi2);
        chi2ndfs[ipat]->Fill(chi2 / ndf);
        layers[ipat]->Fill(layer);
        legacypatterns[ipat]->Fill(legacypattern);
        // all results
        all_offsets->Fill(offset);
        all_slopes->Fill(slope);
      }

      // everything is in half-strips
      const float fmaxOffset = 2;
      const float fminOffset = -1.75;
      const float fmaxSlope = 2.5;
      const float fminSlope = 0;

      // negative bending -> 0
      // positive bending -> 1
      const bool slope_sign(slope >= 0);

      const unsigned offset_bin = assignPosition(offset, fminOffset, fmaxOffset, 4);
      unsigned slope_bin = assignBending(std::abs(slope), fminSlope, fmaxSlope, 4);
      if (slope_sign)
        slope_bin += 16;
      const unsigned fwword = firmwareWord(0, slope_bin, offset_bin);

      // write to output files
      outoffset_sw << code << " " << offset << "\n";
      outslope_sw << code << " " << slope << "\n";
      outoffset_sw_bin << code << " " << offset_bin << "\n";
      outslope_sw_bin << code << " " << slope_bin << "\n";
      outpatternconv << code << " " << legacypattern << "\n";
      outfile_fw << setfill('0');
      outfile_fw << setw(5) << std::hex << fwword << "\n";

      // calculate min and max codes
      if (layer >= N_LAYER_REQUIREMENT) {
        if (offset < minOffset) {
          minOffset = offset;
          minPatt = patt->getId();
          minCode = code;
        }
        if (offset > maxOffset) {
          maxOffset = offset;
          maxPatt = patt->getId();
          maxCode = code;
        }
      }
    }  // end loop on comparator codes

    // write to files
    outoffset_sw.close();
    outslope_sw.close();
    outoffset_sw_bin.close();
    outslope_sw_bin.close();
    outfile_fw.close();
  }

  cout << "minOffset = " << minOffset << endl;
  for (auto patt = newPatterns->begin(); patt != newPatterns->end(); ++patt) {
    if (patt->getId() == minPatt) {
      patt->printCode(minCode);
    }
  }
  cout << "maxOffset = " << maxOffset << endl;
  for (auto patt = newPatterns->begin(); patt != newPatterns->end(); ++patt) {
    if (patt->getId() == maxPatt) {
      patt->printCode(maxCode);
    }
  }

  // plot the figures
  for (auto& q : {offsets, slopes, offsetuncs, slopeuncs, chi2ndfs, layers, legacypatterns}) {
    for (auto& p : q) {
      TCanvas* c = new TCanvas((p.second)->GetName(), (p.second)->GetTitle(), 800, 600);
      c->cd();
      gPad->SetLogy(1);
      (p.second)->Draw();
      c->SaveAs(TString::Format("figures_%dlayers/", N_LAYER_REQUIREMENT) + TString((p.second)->GetName()) + ".pdf");
    }
  }

  file->Write();
  file->Close();

  return 0;
}

/// helpers

int searchForHits(const int code_hits[CSCConstants::NUM_LAYERS][CSCConstants::CLCT_PATTERN_WIDTH],
                  int delta,
                  unsigned N_LAYER_REQUIREMENT) {
  unsigned returnValue = 0;
  unsigned maxlayers = 0;
  for (unsigned iPat = 2; iPat < CSCPatternBank::clct_pattern_legacy_.size(); iPat++) {
    unsigned nlayers = 0;
    for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
      bool layerhit = false;
      for (unsigned j = 0; j < CSCConstants::CLCT_PATTERN_WIDTH; j++) {
        // shifted index
        int jdelta = j + delta;
        if (jdelta < 0)
          jdelta = 0;
        if (jdelta >= CSCConstants::CLCT_PATTERN_WIDTH)
          jdelta = CSCConstants::CLCT_PATTERN_WIDTH - 1;

        // do not consider invalid pattern hits
        if (CSCPatternBank::clct_pattern_legacy_[iPat][i][jdelta]) {
          // is the new pattern half-strip hit?
          if (code_hits[i][j]) {
            layerhit = true;
          }
        }
      }
      // increase the number of layers hit
      if (layerhit) {
        nlayers++;
      }
    }
    if (nlayers > maxlayers and nlayers >= N_LAYER_REQUIREMENT) {
      maxlayers = nlayers;
      returnValue = iPat;
    }
  }
  return returnValue;
}

// function to convert
int convertToLegacyPattern(const int code_hits[CSCConstants::NUM_LAYERS][CSCConstants::CLCT_PATTERN_WIDTH],
                           unsigned N_LAYER_REQUIREMENT) {
  int returnValue = searchForHits(code_hits, 0, N_LAYER_REQUIREMENT);
  // try the search on a half-strip to the left
  if (!returnValue)
    returnValue = searchForHits(code_hits, -1, N_LAYER_REQUIREMENT);
  // try the search on a half-strip to the right
  if (!returnValue)
    returnValue = searchForHits(code_hits, 1, N_LAYER_REQUIREMENT);
  return returnValue;
}

void getErrors(const vector<float>& x, const vector<float>& y, float& sigmaM, float& sigmaB) {
  int N = x.size();
  if (!N) {
    sigmaM = -9;
    sigmaB = -9;
    return;
  }

  float sumx = 0;
  float sumx2 = 0;
  for (int i = 0; i < N; i++) {
    sumx += x[i];
    sumx2 += x[i] * x[i];
  }

  float delta = N * sumx2 - sumx * sumx;

  sigmaM = HALF_STRIP_ERROR * sqrt(1. * N / delta);
  sigmaB = HALF_STRIP_ERROR * sqrt(sumx2 / delta);
}

void writeHeaderPosOffsetLUT(ofstream& file) {
  file << "#CSC Position Offset LUT\n"
       << "#Structure is comparator code (iCC) -> Stub position offset\n"
       << "#iCC bits: 12\n"
       << "#<header> v1.0 12 32 </header>\n";
}

void writeHeaderSlopeLUT(ofstream& file) {
  file << "#CSC Slope LUT\n"
       << "#Structure is comparator code (iCC) -> Stub slope\n"
       << "#iCC bits: 12\n"
       << "#<header> v1.0 12 32 </header>\n";
}

unsigned assignPosition(const float fvalue, const float fmin, const float fmax, const unsigned nbits) {
  bool debug;
  unsigned value = 0;
  const unsigned range = pow(2, nbits);
  const unsigned minValue = 0;
  const unsigned maxValue = range - 1;
  const double fdelta = (fmax - fmin) / range;

  if (fvalue >= fmax) {
    value = maxValue;
  } else if (fvalue <= fmin) {
    value = minValue;
  } else {
    value = std::min(unsigned(std::ceil((fvalue - fmin) / fdelta)), maxValue);
  }
  if (debug)
    std::cout << "fvalue " << fvalue << " " << fmin << " " << fmax << " " << nbits << " " << value << std::endl;

  return value;
}

unsigned assignBending(const float fvalue, const float fmin, const float fmax, const unsigned nbits) {
  bool debug;
  unsigned value = 0;
  const unsigned range = pow(2, nbits);
  const unsigned minValue = 0;
  const unsigned maxValue = range - 1;
  const double fdelta = (fmax - fmin) / range;

  if (fvalue >= fmax) {
    value = maxValue;
  } else if (fvalue <= fmin) {
    value = minValue;
  } else {
    value = std::min(unsigned(std::floor((fvalue - fmin) / fdelta)), maxValue);
  }
  if (debug)
    std::cout << "fvalue " << fvalue << " " << fmin << " " << fmax << " " << nbits << " " << value << std::endl;

  return value;
}

unsigned firmwareWord(const unsigned quality, const unsigned slope, const unsigned offset) {
  /* construct fw dataword:
     [8:0] is quality (set all to 0 for now)
     [12:9] is slope value
     [13] is slope sign
     [17:14] is offset
  */
  enum Masks { OffsetMask = 0xf, SlopeMask = 0x1f, QualityMask = 0x1ff };
  enum Shifts { OffsetShift = 14, SlopeShift = 9, QualityShift = 0 };

  unsigned fwword = 0;
  setDataWord(fwword, quality, QualityShift, QualityMask);
  setDataWord(fwword, slope, SlopeShift, SlopeMask);
  setDataWord(fwword, offset, OffsetShift, OffsetMask);

  return fwword;
}

void setDataWord(unsigned& word, const unsigned newWord, const unsigned shift, const unsigned mask) {
  // clear the old value
  word &= ~(mask << shift);

  // set the new value
  word |= newWord << shift;
}
