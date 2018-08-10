#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "Math/RotationX.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

class testDDFilter : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDFilter);
  CPPUNIT_TEST(checkFilters);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() override{}
  void tearDown() override {}
  void checkFilters();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDFilter);

namespace {
  std::vector<std::string> getNames(DDFilteredView& fv) {
    std::vector<std::string> returnValue;
    bool dodet = fv.firstChild();
    while(dodet) {
      auto const& lp = fv.logicalPart();
      returnValue.emplace_back(lp.name().name());
      dodet = fv.next();
    }
    return returnValue;
  }  
}

void testDDFilter::checkFilters() {
  //Create the geometry
  DDCompactView cv{};
  {
    double const kPI = std::acos(-1.);

    auto const& root = cv.root();

    DDMaterial mat{"Stuff"};
    {
      //Central
      auto outershape = DDSolidFactory::tubs("OuterShape",
                                             1.,
                                             0.5,1.,
                                             0.,
                                             2*kPI);

      DDLogicalPart outerlp{ "Outer", mat, outershape };

      cv.position(outerlp,root,0, DDTranslation{}, DDRotation{});
      {
        DDValue val{"Volume","Outer",0};
        DDsvalues_type values;
        values.emplace_back(DDsvalues_Content_type(val,val));
	
        DDSpecifics ds{"OuterVolume", {"//Outer.*"}, values}; 
      }

      auto middleshape = DDSolidFactory::tubs("MiddleShape",
                                              1.,
                                              0.2,0.49,
                                              0., 2*kPI);
      DDLogicalPart middlelp{"Middle",mat,middleshape };
      cv.position(middlelp, outerlp, 0, DDTranslation{}, DDRotation{});
      {
        DDValue val{"Volume","Middle",0};
        DDsvalues_type values;
        values.emplace_back(DDsvalues_Content_type(val,val));

        DDSpecifics ds{"MiddleVolume", {"//Middle.*"}, values}; 
      }
      
      auto innershape = DDSolidFactory::tubs("InnerShape",
                                             1.,
                                             0.19,0.05,
                                             0., 2*kPI);
      DDLogicalPart innerlp{"Inner",mat,innershape };
      cv.position(innerlp, middlelp, 0, DDTranslation{}, DDRotation{});
      {
        DDValue val{"Volume","Inner",0};
        DDsvalues_type values;
        values.emplace_back(DDsvalues_Content_type(val,val));

        DDSpecifics ds{"InnerVolume", {"//Inner.*"}, values}; 
      }
    }
    {
      //Endcaps
      auto endshape = DDSolidFactory::tubs("EndShape",
					   0.1,
					   0.05,1.,
					   0.,
					   2*kPI);

      DDLogicalPart endlp{"End", mat, endshape};
      cv.position(endlp,root,0,DDTranslation{0.,0.,-(1.0+0.1)},DDRotation{});
      {
        DDValue val{"Volume","EMinus",0};
        DDsvalues_type values;
        values.emplace_back(DDsvalues_Content_type(val,val));
        {
          DDValue val{"Side","-",0};
          values.emplace_back(DDsvalues_Content_type(val,val));
        }
        DDSpecifics ds{"EMinusVolume", {"//End[0]"}, values}; 
      } 

      const DDRotation kXFlip = DDrot("xflip", std::unique_ptr< DDRotationMatrix >( new DDRotationMatrix{ ROOT::Math::RotationX{ kPI} } )); 
      cv.position(endlp,root,1,DDTranslation{0.,0.,1.0+0.1},kXFlip);
      {
        DDValue val{"Volume","EPlus",0};
        DDsvalues_type values;
        values.emplace_back(DDsvalues_Content_type(val,val));
        {
          DDValue val{"Side","+",0};
          values.emplace_back(DDsvalues_Content_type(val,val));
        }

        DDSpecifics ds{"EPlusVolume", {"//End[1]"}, values}; 
      }
      {
        DDValue val{"Endcap","",0};
        DDsvalues_type values;
        values.emplace_back(DDsvalues_Content_type(val,val));

        DDSpecifics ds{"Endcap", {"//End.*"}, values}; 
      }
    }
    cv.lockdown();

    {
      DDSpecificsFilter f;
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Outer","Middle","Inner","End","End"};

      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Volume","Outer",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Outer"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDSpecificsMatchesValueFilter f{DDValue("Volume","Outer",0)};
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Outer"};
      CPPUNIT_ASSERT( names == expectedNames ); 
    }
    {
      DDValue tofind("Volume","Outer",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Middle","Inner","End","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Volume","Middle",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Middle"};
      CPPUNIT_ASSERT( names == expectedNames ); 
    }
    {
      DDSpecificsMatchesValueFilter f(DDValue("Volume","Middle",0));
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Middle"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Volume","Middle",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Outer","Inner","End","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Volume","EPlus",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDSpecificsMatchesValueFilter f(DDValue("Volume","EPlus",0));
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDValue tofind("Volume","EPlus",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Outer","Middle","Inner","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDSpecificsMatchesValueFilter f{DDValue("Volume","EMinus",0)};
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      //Test one LogicalPart with different Specifics
      // based on placement
      {
        DDValue tofind("Side","-",0);
        DDSpecificsFilter f;
        f.setCriteria(tofind, DDCompOp::equals);
        DDFilteredView fv(cv,f);
        
        auto const names = getNames(fv);
        std::vector<std::string> const expectedNames = {"End"};
        CPPUNIT_ASSERT( names == expectedNames );      
      }
      {
        DDSpecificsMatchesValueFilter f{DDValue("Side","-",0)};
        DDFilteredView fv(cv,f);
        
        auto const names = getNames(fv);
        std::vector<std::string> const expectedNames = {"End"};
        CPPUNIT_ASSERT( names == expectedNames );      
      }
      {
        DDValue tofind("Side","+",0);
        DDSpecificsFilter f;
        f.setCriteria(tofind, DDCompOp::equals);
        DDFilteredView fv(cv,f);
        
        auto const names = getNames(fv);
        std::vector<std::string> const expectedNames = {"End"};
        CPPUNIT_ASSERT( names == expectedNames );      
      }
      {
        DDSpecificsMatchesValueFilter f{DDValue("Side","+",0)};
        DDFilteredView fv(cv,f);
        
        auto const names = getNames(fv);
        std::vector<std::string> const expectedNames = {"End"};
        CPPUNIT_ASSERT( names == expectedNames );      
      }
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Outer","Middle","Inner","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Volume","DoesntExist",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDSpecificsMatchesValueFilter f{DDValue("Volume","DoesntExist",0)};
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Volume","DoesntExist",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Outer","Middle","Inner","End","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDSpecificsHasNamedValueFilter f("Volume");
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"Outer","Middle","Inner","End","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDSpecificsHasNamedValueFilter f("DoesntExist");
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Endcap","",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      
      DDSpecificsMatchesValueFilter f{DDValue("Endcap","",0)};
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Endcap","",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {};
      for(auto const& n: names) {
        std::cout <<n;
      }
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Endcap","DoesntExist",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {

      DDSpecificsHasNamedValueFilter f("Endcap");
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End","End"};
      CPPUNIT_ASSERT( names == expectedNames );
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDValue tofind2("Endcap","",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      f.setCriteria(tofind2, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {      
      auto f = make_and_ddfilter(DDSpecificsMatchesValueFilter{DDValue("Volume","EMinus",0)},
                               DDSpecificsMatchesValueFilter{DDValue("Endcap","",0)} );
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDValue tofind2("Endcap","",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind2, DDCompOp::equals);
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      auto f = make_and_ddfilter(DDSpecificsMatchesValueFilter{DDValue("Volume","EMinus",0)},
                               DDSpecificsMatchesValueFilter{DDValue("Endcap","",0)});
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDValue tofind2("Endcap","any",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      f.setCriteria(tofind2, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      auto f = make_and_ddfilter(DDSpecificsMatchesValueFilter{DDValue("Volume","EMinus",0)},
                                 DDSpecificsHasNamedValueFilter{"Endcap"});
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDValue tofind2("Endcap","any",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind2, DDCompOp::not_equals);
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      auto f = make_and_ddfilter(DDSpecificsHasNamedValueFilter{"Endcap"},
                                 DDSpecificsMatchesValueFilter{DDValue("Volume","EMinus",0)});
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {"End"};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDValue tofind2("Endcap","",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      f.setCriteria(tofind2, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDValue tofind2("Endcap","",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind, DDCompOp::equals);
      f.setCriteria(tofind2, DDCompOp::not_equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
    {
      DDValue tofind("Volume","EMinus",0);
      DDValue tofind2("Endcap","",0);
      DDSpecificsFilter f;
      f.setCriteria(tofind2, DDCompOp::not_equals);
      f.setCriteria(tofind, DDCompOp::equals);
      DDFilteredView fv(cv,f);

      auto const names = getNames(fv);
      std::vector<std::string> const expectedNames = {};
      CPPUNIT_ASSERT( names == expectedNames );      
    }
  }

  //CPPUNIT_ASSERT (bad==0);
}
