#ifndef DataFormats_Phase2TrackerDigi_DigiHitRecord_H
#define DataFormats_Phase2TrackerDigi_DigiHitRecord_H

class DigiHitRecord {
private:
  int row_;
  int col_;
  int adc_;

public:
  DigiHitRecord(int row_num, int col_num, int adc_num);

  void set_row(int row) { row_ = row; }
  void set_col(int col) { col_ = col; }

  int row() const { return row_; }
  int col() const { return col_; }
  int adc() const { return adc_; }
};

#endif  // DataFormats_Phase2TrackerDigi_DigiHitRecord_H
