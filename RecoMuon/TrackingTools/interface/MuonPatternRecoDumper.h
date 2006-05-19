#ifndef RecoMuon_MuonPatternRecoDumper_H
#define RecoMuon_MuonPatternRecoDumper_H

/** \class MuonPatternRecoDumper
 *  A class to print information used for debugging
 *
 *  $Date: 2006/05/18 15:03:58 $
 *  $Revision: 1.1 $
 *  \author S. Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include <string>

class DetLayer;
class FreeTrajectoryState;

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

  void dumpLayer(const DetLayer* layer, std::string &where) const;
  void dumpFTS(FreeTrajectoryState& fts, std::string &where) const;
  void dumpFTS(const FreeTrajectoryState& fts, std::string &where) const {
      dumpFTS(const_cast<FreeTrajectoryState&>(fts), where);
  }



protected:

private:

};
#endif

