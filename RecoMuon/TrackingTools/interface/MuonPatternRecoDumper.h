#ifndef RecoMuon_MuonPatternRecoDumper_H
#define RecoMuon_MuonPatternRecoDumper_H

/** \class MuonPatternRecoDumper
 *  A class to print information used for debugging
 *
 *  $Date: $
 *  $Revision: $
 *  \author S. Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

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

protected:

private:

};
#endif

