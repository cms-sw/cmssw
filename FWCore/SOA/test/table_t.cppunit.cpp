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
#include "FWCore/SOA/interface/RowView.h"
#include "FWCore/SOA/interface/Column.h"
#include "FWCore/SOA/interface/TableItr.h"

class testTable: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testTable);
  
  CPPUNIT_TEST(rowviewCtrTest);
  CPPUNIT_TEST(rawTableItrTest);
  CPPUNIT_TEST(tableCtrTest);
  CPPUNIT_TEST(tableStandardOpsTest);
  CPPUNIT_TEST(tableColumnTest);
  CPPUNIT_TEST(tableViewConversionTest);
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
};

namespace ts {
  constexpr const char kEta[] = "eta";
  using Eta = edm::soa::Column<kEta,float>;

  constexpr const char kPhi[] = "eta";
  using Phi = edm::soa::Column<kPhi,float>;

  constexpr const char kEnergy[] = "energy";
  using Energy = edm::soa::Column<kEta,double>;

  constexpr const char kID[] = "id";
  using ID = edm::soa::Column<kID,int>;

  constexpr const char kLabel[] = "label";
  using Label = edm::soa::Column<kLabel,std::string>;
  
  constexpr const char kPx[] = "p_x";
  using Px = edm::soa::Column<kPx, double>;
  
  constexpr const char kPy[] = "p_z";
  using Py = edm::soa::Column<kPy, double>;
  
  constexpr const char kPz[] = "p_z";
  using Pz = edm::soa::Column<kPz, double>;
  
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
  
  std::array<void*, 4> variables{ {&id, &eta, &phi, &label} };
  
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
    MyJetTable jt{ j, column_fillers(filler_for<Label>([&index](JetType const&)
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
      MyJetTable jt{ j, column_fillers(filler_for<Label>([&itFillIndex](JetType const&)
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
      if(iLHS.size() != iRHS.size()) {
        std::cout <<"copy size wrong "<<iLHS.size()<<" "<<iRHS.size()<<std::endl;
      }
      for(size_t i = 0; i< iRHS.size(); ++i) {
        if(iLHS.get<Px>(i) != iRHS.get<Px>(i)) {
          std::cout <<"copy px not the same "<<i<<" "<<iLHS.get<Px>(i)<< " "<<iRHS.get<Px>(i)<<std::endl;
        }
        if(iLHS.get<Py>(i) != iRHS.get<Py>(i)) {
          std::cout <<"copy py not the same "<<i<<" "<<iLHS.get<Py>(i)<< " "<<iRHS.get<Py>(i)<<std::endl;
        }
        if(iLHS.get<Pz>(i) != iRHS.get<Pz>(i)) {
          std::cout <<"copy pz not the same "<<i<<" "<<iLHS.get<Pz>(i)<< " "<<iRHS.get<Pz>(i)<<std::endl;
        }
        if(iLHS.get<Energy>(i) != iRHS.get<Energy>(i)) {
          std::cout <<"copy energy not the same "<<i<<" "<<iLHS.get<Energy>(i)<< " "<<iRHS.get<Energy>(i)<<std::endl;
        }
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
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
