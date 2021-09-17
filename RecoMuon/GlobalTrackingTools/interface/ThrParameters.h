#ifndef RecoMuon_GlobalTrackingTools_ThrParameters_h
#define RecoMuon_GlobalTrackingTools_ThrParameters_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/RecoMuonObjects/interface/DYTThrObject.h"
#include "CondFormats/DataRecord/interface/DYTThrObjectRcd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"

class ThrParameters {
public:
  ThrParameters(const edm::EventSetup*);
  ~ThrParameters();

  void setInitialThr(double thr0) { x0 = thr0; };
  const bool isValidThdDB() { return isValidThdDB_; };
  const std::map<DTChamberId, GlobalError>& GetDTApeMap() { return dtApeMap; };
  const std::map<CSCDetId, GlobalError>& GetCSCApeMap() { return cscApeMap; };
  const DYTThrObject* getInitialThresholds() { return dytThresholds; }

private:
  double x0;
  bool isValidThdDB_;
  const DYTThrObject* dytThresholds;
  std::map<DTChamberId, GlobalError> dtApeMap;
  std::map<CSCDetId, GlobalError> cscApeMap;
};

#endif  // RecoMuon_GlobalTrackingTools_ThrParameters_h
