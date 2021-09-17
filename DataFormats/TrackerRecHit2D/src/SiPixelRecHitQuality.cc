#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiPixelRecHitQuality::Packing::Packing() {
  // Constructor: pre-computes masks and shifts from field widths
  // X is now XY
  // Y is now Q
  probX_width = 14;
  probY_width = 8;
  qBin_width = 3;
  edge_width = 1;
  bad_width = 1;
  twoROC_width = 1;
  hasFilledProb_width = 1;
  spare_width = 3;

  if (probX_width + probY_width + qBin_width + edge_width + bad_width + twoROC_width + hasFilledProb_width +
          spare_width !=
      32) {
    throw cms::Exception("SiPixelRecHitQuality::Packing: ")
        << "\nERROR: The allocated bits for the quality word to not sum to 32."
        << "\n\n";
  }

  probX_units = 1.0018;
  probY_units = 1.0461;
  probX_1_over_log_units = 1.0 / log(probX_units);
  probY_1_over_log_units = 1.0 / log(probY_units);

  // Fields are counted from right to left!
  probX_shift = 0;
  probY_shift = probX_shift + probX_width;
  qBin_shift = probY_shift + probY_width;
  edge_shift = qBin_shift + qBin_width;
  bad_shift = edge_shift + edge_width;
  twoROC_shift = bad_shift + bad_width;
  hasFilledProb_shift = twoROC_shift + twoROC_width;

  // Ensure the complement of the correct
  // number of bits:
  QualWordType zero32 = 0;  // 32-bit wide set of 0's

  probX_mask = ~(~zero32 << probX_width);
  probY_mask = ~(~zero32 << probY_width);
  qBin_mask = ~(~zero32 << qBin_width);
  edge_mask = ~(~zero32 << edge_width);
  bad_mask = ~(~zero32 << bad_width);
  twoROC_mask = ~(~zero32 << twoROC_width);
  hasFilledProb_mask = ~(~zero32 << hasFilledProb_width);
}

//  Initialize the packing format singleton
const SiPixelRecHitQuality::Packing SiPixelRecHitQuality::thePacking;

void SiPixelRecHitQuality::warningObsolete() {
  edm::LogWarning("ObsoleteVariable")
      << "Since 39x, probabilityX and probabilityY have been replaced by probabilityXY and probabilityQ";
}

void SiPixelRecHitQuality::warningOutOfBoundQbin(int iValue, QualWordType const& iQualWord) {
  edm::LogWarning("OutOfBounds") << "Qbin outside the bounds of the quality word. Defaulting to Qbin=0. Qbin = "
                                 << iValue << " QualityWord = " << iQualWord;
}

void SiPixelRecHitQuality::warningOutOfBoundProb(const char* iName, float iProb, QualWordType const& iQualWord) {
  edm::LogWarning("OutOfBounds") << "Prob " << iName
                                 << " outside the bounds of the quality word. Defaulting to Prob=0. Prob = " << iProb
                                 << " QualityWord = " << iQualWord;
}

void SiPixelRecHitQuality::warningOutOfBoundRaw(const char* iName, int iRaw, QualWordType const& iQualWord) {
  edm::LogWarning("OutOfBounds") << "Probability " << iName
                                 << " outside the bounds of the quality word. Defaulting to Prob=0. Raw = " << iRaw
                                 << " QualityWord = " << iQualWord;
}
