#include "Fireworks/Muons/interface/SegmentUtils.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include <cmath>

namespace fireworks {
  void createSegment(int detector,
                     bool matchedSegment,
                     float segmentLength,
                     float segmentLimit,
                     float* segmentPosition,
                     float* segmentDirection,
                     float* segmentInnerPoint,
                     float* segmentOuterPoint) {
    if (detector == MuonSubdetId::CSC || detector == MuonSubdetId::GEM || detector == MuonSubdetId::ME0) {
      if (matchedSegment) {
        segmentOuterPoint[0] = segmentPosition[0] + segmentLength * segmentDirection[0];
        segmentOuterPoint[1] = segmentPosition[1] + segmentLength * segmentDirection[1];
        segmentOuterPoint[2] = segmentLength;

        if (fabs(segmentOuterPoint[1]) > segmentLimit)
          segmentOuterPoint[1] = segmentLimit * (segmentOuterPoint[1] / fabs(segmentOuterPoint[1]));

        segmentInnerPoint[0] = segmentPosition[0] - segmentLength * segmentDirection[0];
        segmentInnerPoint[1] = segmentPosition[1] - segmentLength * segmentDirection[1];
        segmentInnerPoint[2] = -segmentLength;

        if (fabs(segmentInnerPoint[1]) > segmentLimit)
          segmentInnerPoint[1] = segmentLimit * (segmentInnerPoint[1] / fabs(segmentInnerPoint[1]));

        return;
      } else {
        segmentOuterPoint[0] = segmentPosition[0] + segmentDirection[0] * (segmentPosition[2] / segmentDirection[2]);
        segmentOuterPoint[1] = segmentPosition[1] + segmentDirection[1] * (segmentPosition[2] / segmentDirection[2]);
        segmentOuterPoint[2] = segmentPosition[2];

        if (fabs(segmentOuterPoint[1]) > segmentLength)
          segmentOuterPoint[1] = segmentLength * (segmentOuterPoint[1] / fabs(segmentOuterPoint[1]));

        segmentInnerPoint[0] = segmentPosition[0] - segmentDirection[0] * (segmentPosition[2] / segmentDirection[2]);
        segmentInnerPoint[1] = segmentPosition[1] - segmentDirection[1] * (segmentPosition[2] / segmentDirection[2]);
        segmentInnerPoint[2] = -segmentPosition[2];

        if (fabs(segmentInnerPoint[1]) > segmentLength)
          segmentInnerPoint[1] = segmentLength * (segmentInnerPoint[1] / fabs(segmentInnerPoint[1]));

        return;
      }
    }

    if (detector == MuonSubdetId::DT) {
      if (matchedSegment) {
        segmentOuterPoint[0] = segmentPosition[0] + segmentLength * segmentDirection[0];
        segmentOuterPoint[1] = segmentPosition[1] + segmentLength * segmentDirection[1];
        segmentOuterPoint[2] = segmentLength;

        segmentInnerPoint[0] = segmentPosition[0] - segmentLength * segmentDirection[0];
        segmentInnerPoint[1] = segmentPosition[1] - segmentLength * segmentDirection[1];
        segmentInnerPoint[2] = -segmentLength;

        return;
      } else {
        double mag = sqrt(segmentDirection[0] * segmentDirection[0] + segmentDirection[1] * segmentDirection[1] +
                          segmentDirection[2] * segmentDirection[2]);

        double theta =
            atan2(sqrt(segmentDirection[0] * segmentDirection[0] + segmentDirection[1] * segmentDirection[1]),
                  segmentDirection[2]);

        double newSegmentLength = segmentLength / cos(theta);

        segmentInnerPoint[0] = segmentPosition[0] + (segmentDirection[0] / mag) * newSegmentLength;
        segmentInnerPoint[1] = segmentPosition[1] + (segmentDirection[1] / mag) * newSegmentLength;
        segmentInnerPoint[2] = segmentPosition[2] + (segmentDirection[2] / mag) * newSegmentLength;

        segmentOuterPoint[0] = segmentPosition[0] - (segmentDirection[0] / mag) * newSegmentLength;
        segmentOuterPoint[1] = segmentPosition[1] - (segmentDirection[1] / mag) * newSegmentLength;
        segmentOuterPoint[2] = segmentPosition[2] - (segmentDirection[2] / mag) * newSegmentLength;

        if (fabs(segmentOuterPoint[0]) > segmentLimit) {
          segmentOuterPoint[0] = segmentLimit * (segmentOuterPoint[0] / fabs(segmentOuterPoint[0]));
          segmentOuterPoint[1] = (segmentOuterPoint[1] / fabs(segmentOuterPoint[1])) * tan(theta);
        }

        return;
      }
    }

    fwLog(fwlog::kWarning) << "MuonSubdetId: " << detector << std::endl;
    return;
  }

}  // namespace fireworks
