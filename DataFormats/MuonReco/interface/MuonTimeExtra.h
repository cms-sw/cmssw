#ifndef MuonReco_MuonTimeExtra_h
#define MuonReco_MuonTimeExtra_h

/** \class reco::MuonTimeExtra MuonTimeExtra.h DataFormats/MuonReco/interface/MuonTimeExtra.h
 *  
 * A class holding timing information calculated for a muon. 
 *
 * \author Piotr Traczyk, CERN
 *
 * \version $Id: MuonTimeExtra.h,v 1.1 2009/03/27 02:35:08 ptraczyk Exp $
 *
 */

namespace reco {
 
  class MuonTimeExtra {
    public:
      MuonTimeExtra();

      enum Direction { OutsideIn = -1, Undefined = 0, InsideOut = 1 };

      /// number of measurements used in timing calculation
      int nDof() const { return nDof_; };
      void setNDof( const int nDof ) { nDof_=nDof; };
       
      /// 1/beta for prompt particle hypothesis 
      /// (time is constraint to the bunch crossing time)
      float inverseBeta()    const { return inverseBeta_; };
      float inverseBetaErr() const { return inverseBetaErr_; };
      void setInverseBeta( const float iBeta ) { inverseBeta_=iBeta; };
      void setInverseBetaErr( const float iBetaErr) { inverseBetaErr_=iBetaErr; };
       
      /// unconstrained 1/beta (time is free)
      /// Sign convention:
      ///   positive - outward moving particle
      ///   negative - inward moving particle
      float freeInverseBeta()    const { return freeInverseBeta_; };
      float freeInverseBetaErr() const { return freeInverseBetaErr_; };
      void setFreeInverseBeta( const float iBeta ) { freeInverseBeta_=iBeta; };
      void setFreeInverseBetaErr( const float iBetaErr) { freeInverseBetaErr_=iBetaErr; };

      /// time of arrival at the IP for the Beta=1 hypothesis
      ///  a) particle is moving from inside out
      float timeAtIpInOut()    const { return timeAtIpInOut_; }; 
      float timeAtIpInOutErr() const { return timeAtIpInOutErr_; };
      void setTimeAtIpInOut( const float timeIp )     { timeAtIpInOut_=timeIp; }; 
      void setTimeAtIpInOutErr( const float timeErr ) { timeAtIpInOutErr_=timeErr; };
      ///  b) particle is moving from outside in
      float timeAtIpOutIn()    const { return timeAtIpOutIn_; };
      float timeAtIpOutInErr() const { return timeAtIpOutInErr_; };
      void setTimeAtIpOutIn( const float timeIp )     { timeAtIpOutIn_=timeIp; }; 
      void setTimeAtIpOutInErr( const float timeErr ) { timeAtIpOutInErr_=timeErr; };
       
      /// direction estimation based on time dispersion
      Direction direction() const
      {
        if (nDof_<2) return Undefined;
        if ( timeAtIpInOutErr_ > timeAtIpOutInErr_ ) return OutsideIn;
        return InsideOut;
      }
     
    private:

      /// number of measurements used in timing calculation
      int nDof_;
       
      /// 1/beta for prompt particle hypothesis 
      float inverseBeta_;
      float inverseBetaErr_;
       
      /// unconstrained 1/beta (time is free)
      float freeInverseBeta_;
      float freeInverseBetaErr_;

      /// time of arrival at the IP for the Beta=1 hypothesis
      float timeAtIpInOut_;
      float timeAtIpInOutErr_;
      float timeAtIpOutIn_;
      float timeAtIpOutInErr_;
      
  };

}

#endif
