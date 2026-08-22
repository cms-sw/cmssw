#include "L1Trigger/TrackerDTC/interface/StubFE.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

namespace trackerDTC {

  StubFE::StubFE(const Setup* setup, const SensorModule* sm, const TTStubRef& ttStubRef, int bx)
      : setup_(setup), sm_(sm), ttStubRef_(ttStubRef), bx_(bx), channel_(sm->modId()) {
    const int numCol = (sm->psModule() ? setup->mpaNumCol() : setup->cbcNumCol()) / setup->feBaseCol();
    const int numRow = (sm->psModule() ? setup->mpaNumRow() : setup->cbcNumRow()) / setup->feBaseRow();
    // get stub local coordinates
    const std::vector<int>& rows = ttStubRef->clusterRef(0)->getRows();
    const std::vector<int>& cols = ttStubRef->clusterRef(0)->getCols();
    const int row = rows.front() + rows.back();
    const int col = *std::min_element(cols.begin(), cols.end());
    // calculate cic
    cic_ = col / numCol;
    // calculate fec
    fec_ = row / numRow;
    // column number in pitch units
    col_ = col % numCol;
    // row number in half pitch units
    row_ = row % numRow;
    // encoded bend
    bend_ = sm->decodeBend(ttStubRef->bendBE());
    // convert to bits
    ttBV_ = TTBV(1, TTBV::S_ / 2 - setup->fePosValid());
    ttBV_ += TTBV(bx_, setup->feWidthBX());
    ttBV_ += TTBV(fec_, setup->feWidthFEC());
    ttBV_ += TTBV(row_, setup->feWidthRow());
    ttBV_ += TTBV(bend_, setup->feWidthBend(), true);
    ttBV_ += TTBV(col_, setup->feWidthCol());
  }

  StubFE::StubFE(const Setup* setup, int channel, int cic, TTBV& ttBV) : cic_(cic), channel_(channel) {
    col_ = ttBV.extract(setup->feWidthCol());
    bend_ = ttBV.extract(setup->feWidthBend(), true);
    row_ = ttBV.extract(setup->feWidthRow());
    fec_ = ttBV.extract(setup->feWidthFEC());
    bx_ = ttBV.extract(setup->feWidthBX());
  }

  // for std::find
  bool StubFE::operator==(const StubFE& s) const {
    return s.channel() == channel_ && s.cic() == cic_ && s.fec() == fec_ && s.bend() == bend_ && s.row() == row_ &&
           s.col() == col_;
  }

}  // namespace trackerDTC
