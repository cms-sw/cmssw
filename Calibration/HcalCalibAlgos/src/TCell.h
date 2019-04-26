//
// File:   TCell.h
// Author: Anton
//
// Created on July 4, 2008, 4:39 PM
//
// Simple class to hold information on the
// cell id and energy for the HCAL calibration ntuples

#ifndef _TCELL_H
#define _TCELL_H

#include "TObject.h"

class TCell : public TObject {
private:
  UInt_t _id;
  Float_t _e;

public:
  TCell() {
    _id = 0;
    _e = 0.0;
  }
  ~TCell() override{};
  TCell(UInt_t i, Float_t e) {
    _id = i;
    _e = e;
  }

  Float_t e() { return _e; }
  UInt_t id() { return _id; }

  void SetE(Float_t e) { _e = e; }
  void SetId(UInt_t i) { _id = i; }

  ClassDefOverride(TCell, 1);
};

#endif /* _TCELL_H */
