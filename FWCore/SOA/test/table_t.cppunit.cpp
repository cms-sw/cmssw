/*----------------------------------------------------------------------

Test program for edm::SoATuple class.
Changed by Viji on 29-06-2005

 ----------------------------------------------------------------------*/

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <cmath>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/TableView.h"
#include "FWCore/SOA/interface/RowView.h"
#include "FWCore/SOA/interface/Column.h"
#include "FWCore/SOA/interface/TableItr.h"
#include "FWCore/SOA/interface/TableExaminer.h"

class testTable: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testTable);
  
  CPPUNIT_TEST(rowviewCtrTest);
  CPPUNIT_TEST(rawTableItrTest);
  CPPUNIT_TEST(tableCtrTest);
  CPPUNIT_TEST(tableStandardOpsTest);
  CPPUNIT_TEST(tableColumnTest);
  CPPUNIT_TEST(tableViewConversionTest);
  CPPUNIT_TEST(tableExaminerTest);
  CPPUNIT_TEST(tableResizeTest);
  CPPUNIT_TEST(mutabilityTest);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void rowviewCtrTest();
  void rawTableItrTest();
  void tableCtrTest();
  void tableStandardOpsTest();
  void tableColumnTest();
  void tableViewConversionTest();
  void tableExaminerTest();
  void tableResizeTest();
  void mutabilityTest();
};

namespace ts {
  struct Eta : public edm::soa::Column<float,Eta> {
    static constexpr const char * const kLabel = "eta";
  };

  SOA_DECLARE_COLUMN(Phi, float, "phi");
  SOA_DECLARE_COLUMN(Energy, float, "energy");
  SOA_DECLARE_COLUMN(ID, int, "id");
  SOA_DECLARE_COLUMN(Label, std::string, "label");

  SOA_DECLARE_COLUMN(Px,double,"p_x");
  SOA_DECLARE_COLUMN(Py,double,"p_y");
  SOA_DECLARE_COLUMN(Pz,double,"p_z");
  
  using ParticleTable = edm::soa::Table<Px, Py, Pz, Energy>;
  
  using JetTable = edm::soa::Table<Eta,Phi>;

  /* Create a new table that is an extension of an existing table*/
  using MyJetTable = edm::soa::AddColumns_t<JetTable, std::tuple<Label>>;
  
  /* Creat a table that is a sub table of an existing one */
  using MyOtherJetTable = edm::soa::RemoveColumn_t<MyJetTable, Phi>;
}

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTable);

void testTable::rowviewCtrTest()
{
  int id=1;
  float eta = 3.14;
  float phi = 1.5;
  std::string label { "foo" };
  
  std::array<void const*, 4> variables{ {&id, &eta, &phi, &label} };
  
  edm::soa::RowView<ts::ID,ts::Eta,ts::Phi,ts::Label> rv{variables};
  
  CPPUNIT_ASSERT(rv.get<ts::ID>() == id );
  CPPUNIT_ASSERT(rv.get<ts::Eta>() == eta );
  CPPUNIT_ASSERT(rv.get<ts::Phi>() == phi );
  CPPUNIT_ASSERT(rv.get<ts::Label>() == label );
}

void testTable::rawTableItrTest()
{
  int ids[] = {1,2,3};
  float etas[] = {3.14, 2.5, 0.3};
  float phis[] = {1.5,-0.2, 2.9};

  std::array<void*, 3> variables{ {ids, etas, phis} };

  edm::soa::TableItr<ts::ID, ts::Eta, ts::Phi> itr{variables};
  
  for(unsigned int i = 0; i < std::size(ids); ++i, ++itr) {
    auto v = *itr;
    CPPUNIT_ASSERT(v.get<ts::ID>() == ids[i]);
    CPPUNIT_ASSERT(v.get<ts::Eta>() == etas[i]);
    CPPUNIT_ASSERT(v.get<ts::Phi>() == phis[i]);
  }
}

namespace {
  std::vector<double> pTs( edm::soa::TableView<ts::Px,ts::Py> tv) {
    std::vector<double> results;
    results.reserve(tv.size());
    
    for(auto const& r: tv) {
      auto px = r.get<ts::Px>();
      auto py = r.get<ts::Py>();
      results.push_back(std::sqrt(px*px+py*py));
    }
    
    return results;
  }
  
  template<typename C>
  void compareEta( edm::soa::TableView<ts::Eta> iEtas, C const& iContainer) {
    auto it = iContainer.begin();
    for(auto v: iEtas.column<ts::Eta>() ) {
      CPPUNIT_ASSERT( v == *it);
      ++it;
    }
  }
  
  struct JetType {
    double eta_;
    double phi_;
  };
  
  double value_for_column(JetType const& iJ, ts::Eta*) {
    return iJ.eta_;
  }
  
  double value_for_column(JetType const& iJ, ts::Phi*) {
    return iJ.phi_;
  }

  bool tolerance(double a, double b) {
    return std::abs(a-b) < 10E-6;
  }
}

void testTable::tableColumnTest()
{
  using namespace ts;
  using namespace edm::soa;
  std::array<double,3> eta = {{1.,2.,4.}};
  std::array<double,3> phi = {{3.14,0.,1.3}};
  
  JetTable jets{eta,phi};
  {
    auto it = phi.begin();
    for(auto v: jets.column<Phi>() ) {
      CPPUNIT_ASSERT( tolerance(*it, v) );
      ++it;
    }
  }

  {
    auto it = eta.begin();
    for(auto v: jets.column<Eta>() ) {
      CPPUNIT_ASSERT( tolerance(*it, v) );
      ++it;
    }
  }

}

void testTable::tableViewConversionTest()
{
  using namespace ts;
  using namespace edm::soa;
  std::array<double,3> eta = {{1.,2.,4.}};
  std::array<double,3> phi = {{3.14,0.,1.3}};
  
  JetTable jets{eta,phi};

  compareEta(jets,eta);
  
  {
    TableView<Phi,Eta> view{jets};
    auto itEta = eta.cbegin();
    auto itPhi = phi.cbegin();
    for(auto const& v: view) {
      CPPUNIT_ASSERT(tolerance(*itEta,v.get<Eta>()));
      CPPUNIT_ASSERT(tolerance(*itPhi,v.get<Phi>()));
      ++itEta;
      ++itPhi;
    }
  }
  
  std::vector<double> px = { 0.1, 0.9, 1.3 };
  std::vector<double> py = { 0.8, 1.7, 2.1 };
  std::vector<double> pz = { 0.4, 1.0, 0.7 };
  std::vector<double> energy = { 1.4, 3.7, 4.1};
  
  ParticleTable particles{px,py,pz,energy};
  
  {
    std::vector<double> ptCompare;
    ptCompare.reserve(px.size());
    for(unsigned int i=0; i< px.size();++i) {
      ptCompare.push_back(sqrt(px[i]*px[i]+py[i]*py[i]));
    }
    auto it = ptCompare.begin();
    for( auto v : pTs( particles ) ) {
      CPPUNIT_ASSERT(tolerance(*it,v));
      ++it;
    }
  }

}

void testTable::tableCtrTest()
{
  using namespace ts;
  using namespace edm::soa;
  std::array<double,3> eta = {{1.,2.,4.}};
  std::array<double,3> phi = {{3.14,0.,1.3}};
  
  JetTable jets{eta,phi};
  
  {
    auto itEta = eta.begin();
    auto itPhi = phi.begin();
    for(auto const& v: jets) {
      CPPUNIT_ASSERT( tolerance(*itEta, v.get<Eta>()));
      CPPUNIT_ASSERT( tolerance(*itPhi, v.get<Phi>()));
      ++itEta;
      ++itPhi;
    }
  }
  std::vector<double> px = { 0.1, 0.9, 1.3 };
  std::vector<double> py = { 0.8, 1.7, 2.1 };
  std::vector<double> pz = { 0.4, 1.0, 0.7 };
  std::vector<double> energy = { 1.4, 3.7, 4.1};
  
  ParticleTable particles{px,py,pz,energy};
  
  {
    std::vector<JetType> j = {{1.,3.14},{2.,0.},{4.,1.3}};
    std::vector<std::string> labels = {{"jet0","jet1","jet2"}};
    
    int index=0;
    MyJetTable jt{ j, column_fillers(Label::filler([&index](JetType const&)
                                                            { std::ostringstream s;
                                                              s<<"jet"<<index++;
                                                              return s.str();}) ) };
    auto itJ = j.begin();
    auto itLabels = labels.begin();
    for(auto const& v: jt) {
      CPPUNIT_ASSERT(tolerance(itJ->eta_,v.get<Eta>()));
      CPPUNIT_ASSERT(tolerance(itJ->eta_,v.get<Eta>()));
      CPPUNIT_ASSERT(v.get<Label>() == *itLabels);
      ++itJ;
      ++itLabels;
    }

    {
      auto itFillIndex = labels.begin();
      MyJetTable jt{ j, column_fillers(Label::filler([&itFillIndex](JetType const&)
                                                        {return *(itFillIndex++);}) ) };
      auto itLabels = labels.begin();
      for(auto const& v: jt) {
        CPPUNIT_ASSERT(v.get<Label>() == *itLabels);
        ++itLabels;
      }
    }

  }
  
}
void testTable::tableStandardOpsTest()
{
  using namespace ts;
  using namespace edm::soa;
  
  std::vector<double> px = { 0.1, 0.9, 1.3 };
  std::vector<double> py = { 0.8, 1.7, 2.1 };
  std::vector<double> pz = { 0.4, 1.0, 0.7 };
  std::vector<double> energy = { 1.4, 3.7, 4.1};
  
  ParticleTable particles{px,py,pz,energy};

  {
    ParticleTable copyTable{particles};
    
    auto compare = [](const ParticleTable& iLHS, const ParticleTable& iRHS) {
      CPPUNIT_ASSERT(iLHS.size() == iRHS.size());
      for(size_t i = 0; i< iRHS.size(); ++i) {
        CPPUNIT_ASSERT(iLHS.get<Px>(i) == iRHS.get<Px>(i));
        CPPUNIT_ASSERT(iLHS.get<Py>(i) == iRHS.get<Py>(i));
        CPPUNIT_ASSERT(iLHS.get<Pz>(i) == iRHS.get<Pz>(i));
        CPPUNIT_ASSERT(iLHS.get<Energy>(i) == iRHS.get<Energy>(i));
      }
    };
    compare(copyTable,particles);
    
    ParticleTable moveTable(std::move(copyTable));
    compare(moveTable,particles);
    
    ParticleTable opEqTable;
    opEqTable = particles;
    compare(opEqTable,particles);
    
    ParticleTable opEqMvTable;
    opEqMvTable=std::move(moveTable);
    compare(opEqMvTable,particles);
  }
  
}

namespace {
  void checkColumnTypes(edm::soa::TableExaminerBase& reader) {
    auto columns = reader.columnTypes();
    std::array<std::type_index,2> const types{ {typeid(ts::Eta), typeid(ts::Phi) } };

    auto itT = types.begin();
    for( auto c: columns) {
      CPPUNIT_ASSERT(c == *itT);
      ++itT;
    }
  };
  
  void checkColumnDescriptions(edm::soa::TableExaminerBase& reader) {
    auto columns = reader.columnDescriptions();

    std::array<std::string,2> const desc{{"eta", "phi"}};
    std::array<std::type_index,2> const types{ {typeid(float), typeid(float) } };
    
    auto itD = desc.begin();
    auto itT = types.begin();
    
    for( auto c: columns) {
      CPPUNIT_ASSERT(c.first == *itD);
      CPPUNIT_ASSERT(c.second == *itT);
      ++itD;
      ++itT;
    }
  };

}

void testTable::tableExaminerTest()
{
  using namespace edm::soa;
  using namespace ts;
  
  std::array<double,3> eta={{1.,2.,4.}};
  std::array<double,3> phi={{3.14,0.,1.3}};
  int size = eta.size();
  CPPUNIT_ASSERT(size == 3);
  
  JetTable jets{eta,phi};
  
  TableExaminer<JetTable> r(&jets);
  checkColumnTypes(r);
  checkColumnDescriptions(r);
}

void testTable::tableResizeTest() {
  using namespace edm::soa;
  using namespace ts;

  std::vector<double> px = { 0.1, 0.9, 1.3 };
  std::vector<double> py = { 0.8, 1.7, 2.1 };
  std::vector<double> pz = { 0.4, 1.0, 0.7 };
  std::vector<double> energy = { 1.4, 3.7, 4.1};

  ParticleTable particlesStandard{px,py,pz,energy};

  ParticleTable particles{px,py,pz,energy};

  particles.resize(2);
  
  auto compare = [](const ParticleTable& iLHS, const ParticleTable& iRHS, size_t n) {
    for(size_t i = 0; i< n; ++i) {
      CPPUNIT_ASSERT(iLHS.get<Px>(i) == iRHS.get<Px>(i));
      CPPUNIT_ASSERT(iLHS.get<Py>(i) == iRHS.get<Py>(i));
      CPPUNIT_ASSERT(iLHS.get<Pz>(i) == iRHS.get<Pz>(i));
      CPPUNIT_ASSERT(iLHS.get<Energy>(i) == iRHS.get<Energy>(i));
    }
  };
  
  CPPUNIT_ASSERT(particles.size() == 2);
  compare(particlesStandard,particles,2);

  particles.resize(4);
  CPPUNIT_ASSERT(particles.size() == 4);
  compare(particles,particlesStandard,2);
  
  for(size_t i = 2; i<4; ++i) {
    CPPUNIT_ASSERT(particles.get<Px>(i) == 0.);
    CPPUNIT_ASSERT(particles.get<Py>(i) == 0.);
    CPPUNIT_ASSERT(particles.get<Pz>(i) == 0.);
    CPPUNIT_ASSERT(particles.get<Energy>(i) == 0.);

  }
  
}

void testTable::mutabilityTest() {
  using namespace edm::soa;
  using namespace ts;
  
  std::array<double,3> eta={{1.,2.,4.}};
  std::array<double,3> phi={{3.14,0.,1.3}};
  JetTable jets{eta,phi};

  jets.get<Eta>(0) = 0.;
  CPPUNIT_ASSERT(jets.get<Eta>(0) == 0.);
  jets.get<Phi>(1) = 0.03;
  CPPUNIT_ASSERT(tolerance(jets.get<Phi>(1),0.03));

  auto row = jets.row(2);
  CPPUNIT_ASSERT(row.get<Eta>() == 4.);
  CPPUNIT_ASSERT(tolerance(row.get<Phi>(),1.3));
  
  row.copyValuesFrom(JetType{5., 6.});
  CPPUNIT_ASSERT(row.get<Eta>() == 5.);
  CPPUNIT_ASSERT(row.get<Phi>() == 6.);
  
  row.copyValuesFrom(JetType{7.,8.}, column_fillers(Phi::filler([](JetType const&) {return 9.;})));
  CPPUNIT_ASSERT(row.get<Eta>() == 7.);
  CPPUNIT_ASSERT(row.get<Phi>() == 9.);

  row.set<Phi>(10.).set<Eta>(11.);
  CPPUNIT_ASSERT(row.get<Eta>() == 11.);
  CPPUNIT_ASSERT(row.get<Phi>() == 10.);
}


#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
