#include "DataFormats/Phase2TrackerDigi/interface/DigiHitRecord.h"

// Describes the 4x4=16 bit hitmap with its row and column numbers and the ADC values
DigiHitRecord::DigiHitRecord(int row_num, int col_num, int adc_num) {
  row_ = row_num;
  col_ = col_num;
  adc_ = adc_num;
}
