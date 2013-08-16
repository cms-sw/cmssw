#ifndef RecoMuon_MuonPatternRecoDumper_H
#define RecoMuon_MuonPatternRecoDumper_H

/** \class MuonPatternRecoDumper
 *  A class to print information used for debugging
 *
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
  std::string dumpLayer(const DetLayer* layer) const;

  std::string dumpFTS(const FreeTrajectoryState& fts) const;

  std::string dumpTSOS(const TrajectoryStateOnSurface& tsos) const;

  std::string dumpMuonId(const DetId &id) const;
protected:

private:
};
#endif

