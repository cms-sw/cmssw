#ifndef RecoMuon_MuonSeedGenerator_MuonSeedPtExtractor_H
#define RecoMuon_MuonSeedGenerator_MuonSeedPtExtractor_H

/** \class MuonSeedPtExtractor
 */

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVPtExtractor.h" 
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include <map>


namespace edm {class ParameterSet;}

class MuonSeedPtExtractor {

 public:
  /// Constructor with Parameter set and MuonServiceProxy
  MuonSeedPtExtractor(const edm::ParameterSet&);

  /// Destructor
  virtual ~MuonSeedPtExtractor();

  /// If firstHit and secondHit are the same, it calculates the pT 
  /// from the phi bend of the single segment
  virtual std::vector<double> pT_extract(MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstHit,
                                 MuonTransientTrackingRecHit::ConstMuonRecHitPointer secondHit) const;


  void setBeamSpot(const GlobalVector & gv) {theBeamSpot = gv;}

 private:
  int stationCode(MuonTransientTrackingRecHit::ConstMuonRecHitPointer hit) const;
  // because compiler duplicaes constructors
  void init(const edm::ParameterSet& par);
  void fillParametersForCombo(const std::string & name, const edm::ParameterSet&pset);
  void fillScalesForCombo(const std::string & name, const edm::ParameterSet&pset);

  std::vector<double> getPt(const std::vector<double> & vPara, double eta, double dPhi ) const;
 
  std::vector<double> getPt(const std::vector<double> & vPara, double eta, double dPhi, const std::string & combination, const DTChamberId & outerDetId ) const;



  typedef std::map<std::string, std::vector<double> > ParametersMap;
  typedef std::map<std::string, std::vector<double> > ScalesMap;
  ParametersMap theParametersForCombo;
  ScalesMap theScalesForCombo;
  GlobalVector theBeamSpot;
  bool scaleDT_;

};
#endif
