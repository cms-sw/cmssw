#ifndef SiPixelPhase1RawData_h 
#define SiPixelPhase1RawData_h 
// -*- C++ -*-
// 
// Package:     SiPixelPhase1RawData
// Class  :     SiPixelPhase1RawData
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

using edm::DetSetVector;

class SiPixelPhase1RawData : public SiPixelPhase1Base {
  enum {
    NERRORS,
    FIFOFULL,
    TBMMESSAGE,
    TBMTYPE,
    TYPE_NERRORS
  };

  public:
  explicit SiPixelPhase1RawData(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&);

  private:
  edm::EDGetTokenT<DetSetVector<SiPixelRawDataError>> srcToken_;
};

#endif
