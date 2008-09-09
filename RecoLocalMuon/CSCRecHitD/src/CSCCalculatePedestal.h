#ifndef CSCRecHitD_CSCCalculatePedestal_h
#define CSCRecHitD_CSCCalculatePedestal_h

/**
 * \class CSCCalculatePedestal
 *
 * ABC for concrete classes which estimate SCA pedestal in alternative ways
 *
 */
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h"
#include <vector>

class CSCCalculatePedestal {
public:
	CSCCalculatePedestal() : defaultPed( 0. ){}
	virtual ~CSCCalculatePedestal(){};
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
 * Concrete CSCCalculatePedestal... 
 * Pedestal is dynamic, averages first two SCA time bins
 *
 */
class CSCDynamicPedestal2 : public CSCCalculatePedestal {
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
 * Concrete CSCCalculatePedestal... 
 * Pedestal is dynamic, take first SCA time bin
*
 */
class CSCDynamicPedestal1 : public CSCCalculatePedestal {
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
 * Concrete CSCCalculatePedestal... 
 * Pedestal is static, taken from conditions data
*
 */
class CSCStaticPedestal : public CSCCalculatePedestal {
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
  private:
     float ped_;
};

#endif
