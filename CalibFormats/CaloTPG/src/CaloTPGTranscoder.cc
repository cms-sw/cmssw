#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"
#include "CalibFormats/CaloTPG/interface/EcalTPGCompressor.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

CaloTPGTranscoder::CaloTPGTranscoder() :
  hccompress_(new HcalTPGCompressor(this)),
  eccompress_(new EcalTPGCompressor(this)) {
}

CaloTPGTranscoder::~CaloTPGTranscoder() {
}

void CaloTPGTranscoder::setup(const edm::EventSetup& es, CaloTPGTranscoder::Mode mode) const {
}

void CaloTPGTranscoder::releaseSetup() const {
}
