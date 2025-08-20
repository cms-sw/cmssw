#ifndef DataFormats_Phase2TrackerDigi_Phase2ITDigiHit_H
#define DataFormats_Phase2TrackerDigi_Phase2ITDigiHit_H

class Phase2ITDigiHit {
private:
  int row_;  // Hit position row
  int col_;  // Hit position column
  int adc_;  // Hit position adc

public:
  Phase2ITDigiHit(int row_num, int col_num, int adc_num);

  void set_row(int row) { row_ = row; }
  void set_col(int col) { col_ = col; }

  int row() const { return row_; }
  int col() const { return col_; }
  int adc() const { return adc_; }
};

#endif  // DataFormats_Phase2TrackerDigi_Phase2ITDigiHit_H
