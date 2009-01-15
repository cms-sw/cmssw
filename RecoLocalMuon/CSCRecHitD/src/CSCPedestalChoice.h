#ifndef CSCRecHitD_CSCPedestalChoice_h
#define CSCRecHitD_CSCPedestalChoice_h

/**
 * \class CSCPedestalChoice
 *
 * ABC for concrete classes which estimate SCA pedestal in alternative ways
 *
 */
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h"
#include <vector>

class CSCPedestalChoice {
public:
	CSCPedestalChoice() : defaultPed( 0. ){}
	virtual ~CSCPedestalChoice(){};
	/// Return default pedestal (typically zero)
	float getDefault() const { return defaultPed; } 
	/// Allow reseting of default pedestal (not currently used)
	void setDefault( float ped ) { defaultPed = ped; }
	/** 
	* Return appropriate pedestal for supplied SCA vector.
	* If using conditions data then must also supply pointer to CSCRecoConditions
	* and CSCDetId + channel
	*/
	virtual float pedestal( const std::vector<float>& sca, 
	   const CSCRecoConditions* cond=0, const CSCDetId id=0, int ichan=0 ) = 0;
private:
	float defaultPed;
};

/**
 * \class CSCDynamicPedestal2
 *
 * Concrete CSCPedestalChoice... 
 * Pedestal is dynamic, averages first two SCA time bins
 *
 */
class CSCDynamicPedestal2 : public CSCPedestalChoice {
public:
	CSCDynamicPedestal2(){}
	~CSCDynamicPedestal2(){}
	float pedestal( const std::vector<float>& sca, 
	    const CSCRecoConditions*, const CSCDetId, int ){
		float ped = getDefault();
		if ( !sca.empty() ){
			ped = ( sca[0]+sca[1] )/2.;
		}
		return ped;
	}
};

/**
 * \class CSCDynamicPedestal1
 *
 * Concrete CSCPedestalChoice... 
 * Pedestal is dynamic, take first SCA time bin
*
 */
class CSCDynamicPedestal1 : public CSCPedestalChoice {
public:
	CSCDynamicPedestal1(){}
	~CSCDynamicPedestal1(){}
	float pedestal( const std::vector<float>& sca,
	    const CSCRecoConditions*, const CSCDetId, int ){
		float ped = getDefault();
		if ( !sca.empty() ){
			ped = sca[0];
		}
	  return ped;
	}
};

/**
 * \class CSCStaticPedestal
 *
 * Concrete CSCPedestalChoice... 
 * Pedestal is static, taken from conditions data
*
 */
class CSCStaticPedestal : public CSCPedestalChoice {
public:
	CSCStaticPedestal(){}
	~CSCStaticPedestal(){}
	float pedestal( const std::vector<float>& sca,
	    const CSCRecoConditions* cond, const CSCDetId id, int ichan ){
		float ped = cond->pedestal(id, ichan );
		return ped;
	}
};

/**
 * \class CSCSubtractPedestal
 *
 * A class to be used as a function in a for_each algorithm
 * to subtract the pedestal. That is set as the ctor arg.
 *
 */
class CSCSubtractPedestal {
  public:
    CSCSubtractPedestal( float ped ): ped_(ped) {}
    void operator()( float& elem ) const {
      elem -= ped_;
    }
    void operator()( int& elem ) const {
      elem -= static_cast<int>(ped_); // not strictly correct but OK for the typical large pedestals
    }

  private:
     float ped_;
};

#endif
