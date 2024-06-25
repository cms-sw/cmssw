/*
 * StubResult.h
 *
 *  Created on: Feb 6, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef L1T_OmtfP1_STUBRESULT_H_
#define L1T_OmtfP1_STUBRESULT_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"
#include <vector>

class StubResult {
public:
  StubResult() {}  //empty result

  StubResult(float pdfVal, bool valid, int pdfBin, int layer, MuonStubPtr stub)
      : pdfVal(pdfVal), valid(valid), pdfBin(pdfBin), layer(layer), stub(stub) {}

  const MuonStubPtr& getMuonStub() const { return stub; }

  int getPdfBin() const { return pdfBin; }

  float getPdfVal() const { return pdfVal; }

  void setPdfVal(float pdfVal) { this->pdfVal = pdfVal; }

  bool getValid() const { return valid; }

  void setValid(bool valid) { this->valid = valid; }

  int getLayer() const { return layer; }

  void reset() {
    pdfVal = 0;
    valid = false;
    pdfBin = 0;
    layer = 0;
    stub.reset();
  }

private:
  float pdfVal = 0;
  bool valid = false;

  //stub and pdfBin should be needed only for debug, testing, generating patterns, etc, but rather not in the firmware
  int pdfBin = 0;

  //n.b, layer might be different then the the stub->layer, because it might be the result of the bending layer (how about eta?)
  int layer = 0;

  MuonStubPtr stub;
};

typedef std::vector<StubResult> StubResults;

#endif /* L1T_OmtfP1_STUBRESULT_H_ */
