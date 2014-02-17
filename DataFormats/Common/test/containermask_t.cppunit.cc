// $Id: containermask_t.cppunit.cc,v 1.1 2011/12/01 13:02:18 vlimant Exp $

#include <algorithm>

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include <vector>

class testContainerMask : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testContainerMask);
  CPPUNIT_TEST(testVector);
  CPPUNIT_TEST(testDetSetVector);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void testVector();
  void testDetSetVector();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testContainerMask);

void
testContainerMask::testVector()
{
   const unsigned int values[]={0,1,2,3};
   const bool flags[]={true,false,true,false};
   const unsigned int kSize = 4;
   std::vector<unsigned int> c(values,values+kSize);
   std::vector<bool> mask(flags,flags+kSize);
   CPPUNIT_ASSERT(mask.size()==c.size());
   CPPUNIT_ASSERT(mask.size()==kSize);
   
   typedef std::vector<unsigned int> ContainerType;
   edm::TestHandle<ContainerType> h(&c, edm::ProductID(1,1));
   
   const edm::RefProd<ContainerType> rp(h);
   
   edm::ContainerMask<ContainerType> cMask(rp, mask);

   CPPUNIT_ASSERT(cMask.size()==mask.size());
   
   unsigned int index=0;
   for(std::vector<bool>::iterator it = mask.begin(), itEnd = mask.end(); it != itEnd; ++it,++index) {
      CPPUNIT_ASSERT(*it==cMask.mask(index));
   }
   
   index=0;
   for(std::vector<unsigned int>::iterator it = c.begin(), itEnd = c.end(); it != itEnd; ++it, ++index) {
      CPPUNIT_ASSERT(mask[index]==cMask.mask(&(*it)));
   }

   std::vector<bool> alternate(4,false);
   alternate[1]=true;
   cMask.applyOrTo(alternate);
   CPPUNIT_ASSERT(alternate[0]);
   CPPUNIT_ASSERT(alternate[1]);
   CPPUNIT_ASSERT(alternate[2]);
   CPPUNIT_ASSERT(!alternate[3]);
   
   {
     std::vector<bool> alternate(3,false);
     cMask.copyMaskTo(alternate);
     CPPUNIT_ASSERT(alternate[0]);
     CPPUNIT_ASSERT(!alternate[1]);
     CPPUNIT_ASSERT(alternate[2]);
     CPPUNIT_ASSERT(!alternate[3]);
   }
}

void
testContainerMask::testDetSetVector()
{
   const bool flags[]={true,false,true,false};
   const unsigned int kSize = 4;
   edmNew::DetSetVector<unsigned int> c;   
   {
      edmNew::DetSetVector<unsigned int>::FastFiller filler(c,0);
      filler.push_back(0);
      filler.push_back(1);
   }
   {
      edmNew::DetSetVector<unsigned int>::FastFiller filler(c,1);
      filler.push_back(2);
      filler.push_back(3);
   }
   std::vector<bool> mask(flags,flags+kSize);
   CPPUNIT_ASSERT(mask.size()==c.dataSize());
   CPPUNIT_ASSERT(mask.size()==kSize);
   
   typedef edmNew::DetSetVector<unsigned int> ContainerType;
   
   edm::TestHandle<ContainerType> h(&c, edm::ProductID(1,1));
   
   const edm::RefProd<ContainerType> rp(h);
   
   edm::ContainerMask<ContainerType> cMask(rp, mask);

   CPPUNIT_ASSERT(cMask.size()==mask.size());
   
   unsigned int index=0;
   for(std::vector<bool>::iterator it = mask.begin(), itEnd = mask.end(); it != itEnd; ++it,++index) {
      CPPUNIT_ASSERT(*it==cMask.mask(index));
   }
   
   index=0;
   for(std::vector<unsigned int>::const_iterator it = c.data().begin(), itEnd = c.data().end(); it != itEnd; ++it, ++index) {
      CPPUNIT_ASSERT(mask[index]==cMask.mask(&(*it)));
   }

   std::vector<bool> alternate(4,false);
   alternate[1]=true;
   cMask.applyOrTo(alternate);
   CPPUNIT_ASSERT(alternate[0]);
   CPPUNIT_ASSERT(alternate[1]);
   CPPUNIT_ASSERT(alternate[2]);
   CPPUNIT_ASSERT(!alternate[3]);
}
