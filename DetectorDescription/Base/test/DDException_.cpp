#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "DetectorDescription/Base/interface/DDException.h"

class testDDException : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testDDException );
  CPPUNIT_TEST( testEquality );
  
  CPPUNIT_TEST_SUITE_END();
  
public:

  void setUp( void ) 
    {
      m_e = new DDException;
    }
  
  void testEquality( void );

private:
  
  DDException *m_e;  
  DDException *m_copy;  
};

void
testDDException::testEquality( void )
{
  m_copy = m_e;
  CPPUNIT_ASSERT( m_e != 0 );
  CPPUNIT_ASSERT( m_copy != 0 );
  CPPUNIT_ASSERT( m_e == m_copy );
  delete m_e;
  
  DDException e;
  m_e = &e;
  m_copy = m_e;
  CPPUNIT_ASSERT( m_e != 0 );
  CPPUNIT_ASSERT( m_copy != 0 );
  CPPUNIT_ASSERT( &e == m_copy );  

  DDException ee(e);
  m_e = &ee;
  m_copy = m_e;
  CPPUNIT_ASSERT( m_e != 0 );
  CPPUNIT_ASSERT( m_copy != 0 );
  CPPUNIT_ASSERT( &ee == m_copy );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testDDException );
