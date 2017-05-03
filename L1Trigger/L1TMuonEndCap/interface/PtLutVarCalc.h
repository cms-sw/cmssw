#ifndef L1TMuonEndCap_PtLutVarCalc_h
#define L1TMuonEndCap_PtLutVarCalc_h

int CalcTrackTheta( unsigned short theta, int st1_ring2, short mode );

void CalcDeltaPhis( short& dPh12,   short& dPh13,    short& dPh14,   short& dPh23,    short& dPh24,   short& dPh34, short& dPhSign,
                    short& dPhSum4, short& dPhSum4A, short& dPhSum3, short& dPhSum3A, short& outStPh,
                    unsigned short mode );

void CalcDeltaThetas( short& dTh12, short& dTh13, short& dTh14, short& dTh23, short& dTh24, short& dTh34, short mode );

void CalcBends( int& bend1, int& bend2, int& bend3, int& bend4,
		const int pat1, const int pat2, const int pat3, const int pat4,
		const int dPhSign, const int endcap, const int mode, const bool BIT_COMP=false );

void CalcRPCs( int& RPC1, int& RPC2, int& RPC3, int& RPC4, 
	       const int mode, const bool BIT_COMP=false );

int CalcBendFromPattern( const int pattern, const int endcap );


#endif
