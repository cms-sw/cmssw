#ifndef TrackInfo_TrackInfoEnum_h
#define TrackInfo_TrackInfoEnum_h
/** \class reco::TrackInfoEnum TrackInfoEnum.h AnalysisDataFormats/TrackInfo/interface/TrackInfoEnum.h
 *
 * It contains Enums
 * for TrackInfo
 * 
 *
 * \author Chiara Genta
 *
 *
 */
namespace reco {
  enum StateType { Updated = 0, Combined = 1, FwPredicted = 2, BwPredicted = 3 };

  enum RecHitType { Single = 0, Matched = 1, Projected = 2, Null = 3 };
}  // namespace reco
#endif
