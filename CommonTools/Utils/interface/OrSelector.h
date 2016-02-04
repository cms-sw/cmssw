#ifndef CommonTools_Utils_OrSelector_h
#define CommonTools_Utils__OrSelector_h
/* \class OrSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: OrSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

namespace helpers {
  struct NullOrOperand;
}

namespace reco {
  namespace modules {
    template<typename T1, typename T2, typename T3, typename T4, typename T5> struct CombinedEventSetupInit;
  }
}

template<typename S1, typename S2, 
	 typename S3 = helpers::NullOrOperand, typename S4 = helpers::NullOrOperand,
	 typename S5 = helpers::NullOrOperand>
struct OrSelector {
  OrSelector( const S1 & s1, const S2 & s2, const S3 & s3, const S4 & s4, const S5 & s5 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ), s4_( s4 ), s5_( s5 ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return s1_( t ) || s2_( t ) || s3_( t ) || s4_( t ) || s5_( t );
  }
private:
  friend class reco::modules::CombinedEventSetupInit<S1, S2, S3, S4, S5>;
  S1 s1_;
  S2 s2_;
  S3 s3_;
  S4 s4_;
  S5 s5_;
};


template<typename S1, typename S2>
struct OrSelector<S1, S2, helpers::NullOrOperand, helpers::NullOrOperand, helpers::NullOrOperand> {
  OrSelector( const S1 & s1, const S2 & s2 ) :
    s1_( s1 ), s2_( s2 ) { }

  template<typename T>
  bool operator()( const T & t ) const { 
    return s1_( t ) || s2_( t );
  }
private:
  friend class reco::modules::CombinedEventSetupInit<S1, S2, helpers::NullOrOperand, helpers::NullOrOperand, helpers::NullOrOperand>;
  S1 s1_;
  S2 s2_;
};

template<typename S1, typename S2, typename S3>
struct OrSelector<S1, S2, S3, helpers::NullOrOperand, helpers::NullOrOperand> {
  OrSelector( const S1 & s1, const S2 & s2, const S3 & s3 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return s1_( t ) || s2_( t ) || s3_( t );
  }
private:
  friend class reco::modules::CombinedEventSetupInit<S1, S2, S3, helpers::NullOrOperand, helpers::NullOrOperand>;
  S1 s1_;
  S2 s2_;
  S3 s3_;
};

template<typename S1, typename S2, typename S3, typename S4>
struct OrSelector<S1, S2, S3, S4, helpers::NullOrOperand> {
  OrSelector( const S1 & s1, const S2 & s2, const S3 & s3, const S4 & s4 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ), s4_( s4 ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return s1_( t ) || s2_( t ) || s3_( t ) || s4_( t );
  }
private:
  friend class reco::modules::CombinedEventSetupInit<S1, S2, S3, S4, helpers::NullOrOperand>;
  S1 s1_;
  S2 s2_;
  S3 s3_;
  S4 s4_;
};

#endif
