#ifndef MuonReco_MuonTime_h
#define MuonReco_MuonTime_h


namespace reco {
    struct MuonTime {
       enum Direction { OutsideIn = -1, Undefined = 0, InsideOut = 1 };
	    
       /// number of muon stations used
       int nStations;
       
       /// 1/beta for prompt particle hypothesis 
       /// (time is constraint to the bunch crossing time)
       float inverseBeta;
       float inverseBetaErr;
       
       /// unconstrained 1/beta (time is free)
       /// Sign convention:
       ///   positive - outward moving particle
       ///   negative - inward moving particle
       float freeInverseBeta;
       float freeInverseBetaErr;

       /// time of arrival at the IP for the Beta=1 hypothesis
       ///  a) particle is moving from inside out
       float timeAtIpInOut;
       float timeAtIpInOutErr;
       ///  b) particle is moving from outside in
       float timeAtIpOutIn;
       float timeAtIpOutInErr;
       
       /// direction estimation based on time dispersion
       Direction direction() const
	 {
	    if (nStations<2) return Undefined;
	    if ( timeAtIpInOutErr > timeAtIpOutInErr ) return OutsideIn;
	    return InsideOut;
	 }
       
       
       MuonTime():
       nStations(0), inverseBeta(0), inverseBetaErr(0), 
       freeInverseBeta(0), freeInverseBetaErr(0), 
       timeAtIpInOut(0), timeAtIpInOutErr(0), timeAtIpOutIn(0), timeAtIpOutInErr(0)
	 {}
    };
}
#endif
