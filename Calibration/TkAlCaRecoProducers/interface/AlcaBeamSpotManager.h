#ifndef AlcaBeamSpotManager_H
#define AlcaBeamSpotManager_H

/** \class AlcaBeamSpotManager
 *  No description available.
 *
 *  $Date: 2010/06/30 20:49:56 $
 *  $Revision: 1.2 $
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
//#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include <string>
#include <map>
#include <utility>


class AlcaBeamSpotManager{
 public:
  AlcaBeamSpotManager         (void);
  AlcaBeamSpotManager         (const edm::ParameterSet&);
  virtual ~AlcaBeamSpotManager(void);

  void reset(void);   
  void readLumi(const edm::LuminosityBlock&);   
  void createWeightedPayloads(void);   
  const std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot>& getPayloads(void){return beamSpotMap_;}   

  typedef std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot>::iterator bsMap_iterator;
 private:
  reco::BeamSpot         weight  (const bsMap_iterator& begin,
                                  const bsMap_iterator& end);
  void                   weight  (double& mean,double& meanError,const double& val,const double& valError);
  std::pair<float,float> delta   (const float& x, const float& xError, const float& nextX, const float& nextXError);
  float                  deltaSig(const float& num, const float& den);
  std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot> beamSpotMap_;

  std::string beamSpotOutputBase_;
  std::string beamSpotModuleName_;
  std::string beamSpotLabel_;

};

#endif
