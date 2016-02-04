#ifndef CommonTools_Utils_AndSelector_h
#define CommonTools_Utils_AndSelector_h
/* \class AndSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AndSelector.h,v 1.1 2009/02/20 16:20:38 llista Exp $
 */

namespace helpers {
  struct NullAndOperand;
}

namespace reco {
  namespace modules {
    template<typename T1, typename T2, typename T3, typename T4, typename T5> struct CombinedEventSetupInit;
  }
}

template<typename S1, typename S2, 
	 typename S3 = helpers::NullAndOperand, typename S4 = helpers::NullAndOperand,
	 typename S5 = helpers::NullAndOperand>
struct AndSelector {
  AndSelector( const S1 & s1, const S2 & s2, const S3 & s3, const S4 & s4, const S5 & s5 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ), s4_( s4 ), s5_( s5 ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return s1_( t ) && s2_( t ) && s3_( t ) && s4_( t ) && s5_( t );
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
struct AndSelector<S1, S2, helpers::NullAndOperand, helpers::NullAndOperand, helpers::NullAndOperand> {
  AndSelector( const S1 & s1, const S2 & s2 ) :
    s1_( s1 ), s2_( s2 ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return s1_( t ) && s2_( t );
  }
  template<typename T1, typename T2>
  bool operator()( const T1 & t1, const T2 & t2 ) const { 
    return s1_( t1 ) && s2_( t2 );
  }
private:
  friend class reco::modules::CombinedEventSetupInit<S1, S2, helpers::NullAndOperand, helpers::NullAndOperand, helpers::NullAndOperand>;
  S1 s1_;
  S2 s2_;
};

template<typename S1, typename S2, typename S3>
struct AndSelector<S1, S2, S3, helpers::NullAndOperand, helpers::NullAndOperand> {
  AndSelector( const S1 & s1, const S2 & s2, const S3 & s3 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return s1_( t ) && s2_( t ) && s3_( t );
  }
  template<typename T1, typename T2, typename T3>
  bool operator()( const T1 & t1,  const T2 & t2,  const T3 & t3 ) const {
    return s1_( t1 ) && s2_( t2 ) && s3_( t3 );
  }
private:
  friend class reco::modules::CombinedEventSetupInit<S1, S2, S3, helpers::NullAndOperand, helpers::NullAndOperand>;
  S1 s1_;
  S2 s2_;
  S3 s3_;
};

template<typename S1, typename S2, typename S3, typename S4>
struct AndSelector<S1, S2, S3, S4, helpers::NullAndOperand> {
  AndSelector( const S1 & s1, const S2 & s2, const S3 & s3, const S4 & s4 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ), s4_( s4 ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return s1_( t ) && s2_( t ) && s3_( t ) && s4_( t );
  }
  template<typename T1, typename T2, typename T3, typename T4>
  bool operator()( const T1 & t1,  const T2 & t2,  const T3 & t3, const T4 & t4 ) const {
    return s1_( t1 ) && s2_( t2 ) && s3_( t3 ) && s4_( t4 );
  }
private:
  friend class reco::modules::CombinedEventSetupInit<S1, S2, S3, S4, helpers::NullAndOperand>;
  S1 s1_;
  S2 s2_;
  S3 s3_;
  S4 s4_;
};

#endif
