#ifndef AlcaBeamSpotManager_H
#define AlcaBeamSpotManager_H

/** \class AlcaBeamSpotManager
 *  No description available.
 *
 *  $Date: 2010/06/18 14:25:43 $
 *  $Revision: 1.1 $
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
//#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include <string>
#include <map>

class AlcaBeamSpotManager{
 public:
  AlcaBeamSpotManager         (void);
  AlcaBeamSpotManager         (const edm::ParameterSet&);
  virtual ~AlcaBeamSpotManager(void);

  void readLumi(const edm::LuminosityBlock&);   
  void createWeightedPayloads(void);   
  const std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot>& getPayloads(void){return beamSpotMap_;}   

  typedef std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot>::iterator bsMap_iterator;
 private:
  reco::BeamSpot weight(const bsMap_iterator& begin,
                        const bsMap_iterator& end);
  void           weight(double& mean,double& meanError,const double& val,const double& valError);
  std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot> beamSpotMap_;

  std::string beamSpotOutputBase_;

};

#endif
