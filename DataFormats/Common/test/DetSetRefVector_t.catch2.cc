#include <catch2/catch_all.hpp>
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/DetSetRefVector.h"
#include "DataFormats/Common/interface/TestHandle.h"

namespace testdetsetrefvector {
  class Value {
  public:
    // VALUES must be default constructible
    Value() : d_(0.0) {}

    // This constructor is used for testing; it is not required by the
    // concept VALUE.
    explicit Value(double d) : d_(d) {}

    // This access function is used for testing; it is not required by
    // the concept VALUE.
    double val() const { return d_; }

    // VALUES must be destructible
    ~Value() {}

    // VALUES must be LessThanComparable
    bool operator<(Value const& other) const { return d_ < other.d_; }

    // The private stuff below is all implementation detail, and not
    // required by the concept VALUE.
  private:
    double d_;
  };

}  // namespace testdetsetrefvector

using namespace testdetsetrefvector;
typedef edm::DetSetVector<Value> dsv_type;
typedef dsv_type::detset detset;

TEST_CASE("DetSetRefVector", "[DetSetRefVector]") {
  SECTION("DetSetRefVector construction") {
    dsv_type c;
    detset d3;
    Value v1(1.1);
    Value v2(2.2);
    d3.id = edm::det_id_type(3);
    d3.data.push_back(v1);
    d3.data.push_back(v2);
    c.insert(d3);
    detset d1;
    Value v1a(4.1);
    Value v2a(3.2);
    d1.id = edm::det_id_type(1);
    d1.data.push_back(v1a);
    d1.data.push_back(v2a);
    c.insert(d1);
    c.post_insert();

    edm::TestHandle<dsv_type> pc2(&c, edm::ProductID(1, 1));

    {
      std::vector<edm::det_id_type> ids;
      ids.push_back(1);
      ids.push_back(3);

      edm::DetSetRefVector<Value> refVector(pc2, ids);
      REQUIRE(refVector.size() == ids.size());

      dsv_type::const_iterator dsvItr = c.begin();
      for (edm::DetSetRefVector<Value>::const_iterator it = refVector.begin(), itEnd = refVector.end(); it != itEnd;
           ++it, ++dsvItr) {
        REQUIRE(it->id == dsvItr->id);
        REQUIRE(it->data.size() == dsvItr->data.size());
      }
    }

    {
      std::vector<edm::det_id_type> ids;
      ids.push_back(3);

      edm::DetSetRefVector<Value> refVector(pc2, ids);
      REQUIRE(refVector.size() == ids.size());

      edm::DetSetRefVector<Value>::const_iterator itRef = refVector.begin();
      for (std::vector<edm::det_id_type>::const_iterator itId = ids.begin(), itIdEnd = ids.end(); itId != itIdEnd;
           ++itRef, ++itId) {
        REQUIRE(itRef->id == *itId);
        REQUIRE(itRef->id == c.find(*itId)->id);
        REQUIRE(itRef->data.size() == c.find(*itId)->data.size());
      }
    }
  }

  SECTION("DetSetRefVector find") {
    dsv_type c;
    detset d3;
    Value v1(1.1);
    Value v2(2.2);
    d3.id = edm::det_id_type(3);
    d3.data.push_back(v1);
    d3.data.push_back(v2);
    c.insert(d3);
    detset d1;
    Value v1a(4.1);
    Value v2a(3.2);
    d1.id = edm::det_id_type(1);
    d1.data.push_back(v1a);
    d1.data.push_back(v2a);
    c.insert(d1);
    c.post_insert();

    edm::TestHandle<dsv_type> pc2(&c, edm::ProductID(1, 1));

    {
      std::vector<edm::det_id_type> ids;
      ids.push_back(1);
      ids.push_back(3);

      edm::DetSetRefVector<Value> refVector(pc2, ids);

      REQUIRE(refVector.find(1)->id == c.find(1)->id);
      REQUIRE(refVector.find(3)->id == c.find(3)->id);
      REQUIRE(refVector.find(4) == refVector.end());
    }
  }
}
