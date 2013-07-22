#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/HLTReco/interface/HLTResult.h"

class testTrack : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTrack);
  CPPUNIT_TEST(check4);
  CPPUNIT_TEST(check8);
  CPPUNIT_TEST(check16);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void check4(); 
  void check8(); 
  void check16(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testTrack);

void testTrack::check4() {
  bool value[] = { true, false };
  for( int i0 = 0; i0 < 2; ++ i0 )
    for( int i1 = 0; i1 < 2; ++ i1 )
      for( int i2 = 0; i2 < 2; ++ i2 )
	for( int i3 = 0; i3 < 2; ++ i3 ) 
	  {
	    reco::HLTResult<4> r;
	    if ( value[i0] ) r.set<0>();
	    if ( value[i1] ) r.set<1>();
	    if ( value[i2] ) r.set<2>();
	    if ( value[i3] ) r.set<3>();
	    CPPUNIT_ASSERT( r.match<0>() == value[i0] );
	    CPPUNIT_ASSERT( r.match<1>() == value[i1] );
	    CPPUNIT_ASSERT( r.match<2>() == value[i2] );
	    CPPUNIT_ASSERT( r.match<3>() == value[i3] );
	  }
}

void testTrack::check8() {
  bool value[] = { true, false };
  for( int i0 = 0; i0 < 2; ++ i0 )
    for( int i1 = 0; i1 < 2; ++ i1 )
      for( int i2 = 0; i2 < 2; ++ i2 )
	for( int i3 = 0; i3 < 2; ++ i3 ) 
	  for( int i4 = 0; i4 < 2; ++ i4 ) 
	    for( int i5 = 0; i5 < 2; ++ i5 ) 
	      for( int i6 = 0; i6 < 2; ++ i6 ) 
		for( int i7 = 0; i7 < 2; ++ i7 ) 
		  for( int i8 = 0; i8 < 2; ++ i8 )
		    {
		      reco::HLTResult<8> r;
		      if ( value[i0] ) r.set<0>();
		      if ( value[i1] ) r.set<1>();
		      if ( value[i2] ) r.set<2>();
		      if ( value[i3] ) r.set<3>();
		      if ( value[i4] ) r.set<4>();
		      if ( value[i5] ) r.set<5>();
		      if ( value[i6] ) r.set<6>();
		      if ( value[i7] ) r.set<7>();
		      CPPUNIT_ASSERT( r.match<0>() == value[i0] );
		      CPPUNIT_ASSERT( r.match<1>() == value[i1] );
		      CPPUNIT_ASSERT( r.match<2>() == value[i2] );
		      CPPUNIT_ASSERT( r.match<3>() == value[i3] );
		      CPPUNIT_ASSERT( r.match<4>() == value[i4] );
		      CPPUNIT_ASSERT( r.match<5>() == value[i5] );
		      CPPUNIT_ASSERT( r.match<6>() == value[i6] );
		      CPPUNIT_ASSERT( r.match<7>() == value[i7] );
		    }
}


void testTrack::check16() {
  bool value[] = { true, false };
  for( int i0 = 0; i0 < 2; ++ i0 )
    for( int i1 = 0; i1 < 2; ++ i1 )
      for( int i2 = 0; i2 < 2; ++ i2 )
	for( int i3 = 0; i3 < 2; ++ i3 ) 
	  for( int i4 = 0; i4 < 2; ++ i4 ) 
	    for( int i5 = 0; i5 < 2; ++ i5 ) 
	      for( int i6 = 0; i6 < 2; ++ i6 ) 
		for( int i7 = 0; i7 < 2; ++ i7 ) 
		  for( int i8 = 0; i8 < 2; ++ i8 )
		    for( int i9 = 0; i9 < 2; ++ i9 )
		      for( int i10 = 0; i10 < 2; ++ i10 )
			for( int i11 = 0; i11 < 2; ++ i11 )
			  for( int i12 = 0; i12 < 2; ++ i12 )
			    for( int i13 = 0; i13 < 2; ++ i13 )
			      for( int i14 = 0; i14 < 2; ++ i14 )
				for( int i15 = 0; i15 < 2; ++ i15 )
				  {
				    reco::HLTResult<16> r;
				    if ( value[i0] ) r.set<0>();
				    if ( value[i1] ) r.set<1>();
				    if ( value[i2] ) r.set<2>();
				    if ( value[i3] ) r.set<3>();
				    if ( value[i4] ) r.set<4>();
				    if ( value[i5] ) r.set<5>();
				    if ( value[i6] ) r.set<6>();
				    if ( value[i7] ) r.set<7>();
				    if ( value[i8] ) r.set<8>();
				    if ( value[i9] ) r.set<9>();
				    if ( value[i10] ) r.set<10>();
				    if ( value[i11] ) r.set<11>();
				    if ( value[i12] ) r.set<12>();
				    if ( value[i13] ) r.set<13>();
				    if ( value[i14] ) r.set<14>();
				    if ( value[i15] ) r.set<15>();
				    CPPUNIT_ASSERT( r.match<0>() == value[i0] );
				    CPPUNIT_ASSERT( r.match<1>() == value[i1] );
				    CPPUNIT_ASSERT( r.match<2>() == value[i2] );
				    CPPUNIT_ASSERT( r.match<3>() == value[i3] );
				    CPPUNIT_ASSERT( r.match<4>() == value[i4] );
				    CPPUNIT_ASSERT( r.match<5>() == value[i5] );
				    CPPUNIT_ASSERT( r.match<6>() == value[i6] );
				    CPPUNIT_ASSERT( r.match<7>() == value[i7] );
				    CPPUNIT_ASSERT( r.match<8>() == value[i8] );
				    CPPUNIT_ASSERT( r.match<9>() == value[i9] );
				    CPPUNIT_ASSERT( r.match<10>() == value[i10] );
				    CPPUNIT_ASSERT( r.match<11>() == value[i11] );
				    CPPUNIT_ASSERT( r.match<12>() == value[i12] );
				    CPPUNIT_ASSERT( r.match<13>() == value[i13] );
				    CPPUNIT_ASSERT( r.match<14>() == value[i14] );
				    CPPUNIT_ASSERT( r.match<15>() == value[i15] );
				  }
}

