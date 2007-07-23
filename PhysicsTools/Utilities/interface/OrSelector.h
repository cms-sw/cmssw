#ifndef Utilities_OrSelector_h
#define Utilities__OrSelector_h
/* \class OrSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: OrSelector.h,v 1.3 2007/02/26 11:52:01 llista Exp $
 */
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

namespace helpers {
  struct NullOrOperand { };
}

template<typename S1, typename S2, 
	 typename S3 = helpers::NullOrOperand, typename S4 = helpers::NullOrOperand,
	 typename S5 = helpers::NullOrOperand>
struct OrSelector {
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S2::value_type>::value ) );
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S3::value_type>::value ) );
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S4::value_type>::value ) );
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S5::value_type>::value ) );
  typedef typename S1::value_type value_type;
  OrSelector( const S1 & s1, const S2 & s2, const S3 & s3, const S4 & s4, const S5 & s5 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ), s4_( s4 ), s5_( s5 ) { }
  bool operator()( const value_type & t ) const { 
    return s1_( t ) || s2_( t ) || s3_( t ) || s4_( t ) || s5_( t );
  }
private:
  S1 s1_;
  S2 s2_;
  S3 s3_;
  S4 s4_;
  S5 s5_;
};


template<typename S1, typename S2>
struct OrSelector<S1, S2, helpers::NullOrOperand, helpers::NullOrOperand, helpers::NullOrOperand> {
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S2::value_type>::value ) );
  typedef typename S1::value_type value_type;
  OrSelector( const S1 & s1, const S2 & s2 ) :
    s1_( s1 ), s2_( s2 ) { }
  bool operator()( const value_type & t ) const { 
    return s1_( t ) || s2_( t );
  }
private:
  S1 s1_;
  S2 s2_;
};

template<typename S1, typename S2, typename S3>
struct OrSelector<S1, S2, S3, helpers::NullOrOperand, helpers::NullOrOperand> {
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S2::value_type>::value ) );
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S3::value_type>::value ) );
  typedef typename S1::value_type value_type;
  OrSelector( const S1 & s1, const S2 & s2, const S3 & s3 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ) { }
  bool operator()( const value_type & t ) const { 
    return s1_( t ) || s2_( t ) || s3_( t );
  }
private:
  S1 s1_;
  S2 s2_;
  S3 s3_;
};

template<typename S1, typename S2, typename S3, typename S4>
struct OrSelector<S1, S2, S3, S4, helpers::NullOrOperand> {
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S2::value_type>::value ) );
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S3::value_type>::value ) );
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S4::value_type>::value ) );
  typedef typename S1::value_type value_type;
  OrSelector( const S1 & s1, const S2 & s2, const S3 & s3, const S4 & s4 ) :
    s1_( s1 ), s2_( s2 ), s3_( s3 ), s4_( s4 ) { }
  bool operator()( const value_type & t ) const { 
    return s1_( t ) || s2_( t ) || s3_( t ) || s4_( t );
  }
private:
  S1 s1_;
  S2 s2_;
  S3 s3_;
  S4 s4_;
};

#endif
