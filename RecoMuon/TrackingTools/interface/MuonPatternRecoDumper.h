#ifndef RecoMuon_MuonPatternRecoDumper_H
#define RecoMuon_MuonPatternRecoDumper_H

/** \class MuonPatternRecoDumper
 *  A class to print information used for debugging
 *
 *  $Date: 2006/05/23 13:48:08 $
 *  $Revision: 1.4 $
 *  \author S. Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include <string>

class DetLayer;
class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class DetId;

class MuonPatternRecoDumper {
public:
  /// Constructor
  MuonPatternRecoDumper();

  /// Destructor
  virtual ~MuonPatternRecoDumper();

  // Operations
  void dumpLayer(const DetLayer* layer) const;

  void dumpFTS(FreeTrajectoryState& fts) const;
  void dumpFTS(const FreeTrajectoryState& fts) const {
      dumpFTS(const_cast<FreeTrajectoryState&>(fts));
  }
  void dumpTSOS(TrajectoryStateOnSurface& tsos) const;
  void dumpTSOS(const TrajectoryStateOnSurface& tsos) const {
      dumpTSOS(const_cast<TrajectoryStateOnSurface&>(tsos));
  }
  void dumpLayer(const DetLayer* layer, std::string &where) const;

  void dumpFTS(FreeTrajectoryState& fts, std::string &where) const;
  void dumpFTS(const FreeTrajectoryState& fts, std::string &where) const {
      dumpFTS(const_cast<FreeTrajectoryState&>(fts), where);
  }
  void dumpTSOS(TrajectoryStateOnSurface& tsos, std::string &where) const;
  void dumpTSOS(const TrajectoryStateOnSurface& tsos, std::string &where) const {
      dumpTSOS(const_cast<TrajectoryStateOnSurface&>(tsos), where);
  }
  
  void dumpMuonId(const DetId &id) const;
  void dumpMuonId(const DetId &id, std::string &where) const;

protected:

private:

};
#endif

