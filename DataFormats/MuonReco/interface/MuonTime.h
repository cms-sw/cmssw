#ifndef MuonReco_MuonTime_h
#define MuonReco_MuonTime_h

namespace reco {
  struct MuonTime {
    enum Direction { OutsideIn = -1, Undefined = 0, InsideOut = 1 };

    /// number of muon stations used
    int nDof;

    /// time of arrival at the IP for the Beta=1 hypothesis
    ///  a) particle is moving from inside out
    float timeAtIpInOut;
    float timeAtIpInOutErr;
    ///  b) particle is moving from outside in
    float timeAtIpOutIn;
    float timeAtIpOutInErr;

    /// direction estimation based on time dispersion
    Direction direction() const {
      if (nDof < 2)
        return Undefined;
      if (timeAtIpInOutErr > timeAtIpOutInErr)
        return OutsideIn;
      return InsideOut;
    }

    MuonTime() : nDof(0), timeAtIpInOut(0), timeAtIpInOutErr(0), timeAtIpOutIn(0), timeAtIpOutInErr(0) {}
  };
}  // namespace reco
#endif
