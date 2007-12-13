#ifndef MuonReco_MuonTime_h
#define MuonReco_MuonTime_h


namespace reco {
    struct MuonTime {
       enum Direction { OutsideIn = -1, Undefined = 0, InsideOut = 1 };
	    
       /// number of muon stations used
       int nStations;
       
       /// 1/beta for prompt particle hypothesis
       float inverseBeta;
       float inverseBetaErr;
       
       /// unconstrained 1/beta
       /// Sign convention:
       ///   positive - outward moving particle
       ///   negative - inward moving particle
       float freeInverseBeta;
       float freeInverseBetaErr;

       /// time of arrival at the IP for the Beta=1 hypothesis
       float vertexTime;
       float vertexTimeErr;
       /// direction of the particle for the Beta=1 hypothesis
       Direction direction;
       
       MuonTime():
       nStations(0), inverseBeta(0), inverseBetaErr(0), 
       freeInverseBeta(0), freeInverseBetaErr(0), vertexTime(0), vertexTimeErr(0), 
       direction(Undefined) {}
    };
}
#endif
