#include <cppunit/extensions/HelperMacros.h>

#include <DD4hep/Filter.h>
#include <memory>
#include <string_view>
#include <vector>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace dd4hep;
using namespace dd4hep::dd;
using namespace std;
using namespace std::literals;

class testFilter : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testFilter);
  CPPUNIT_TEST(checkFilter);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkFilter();

private:
  vector<unique_ptr<Filter>> filters_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testFilter);

void testFilter::setUp() {
  vector<string_view> selections = {"//MB2P.*",
                                    "//MB2P.*/MB2SuperLayerPhi",
                                    "//MB2P.*/MB2SuperLayerPhi/MB2SLPhiLayer_.*Cells.*",
                                    "//MB2P.*/MB2SuperLayerPhi/MB2SLPhiLayer_.*Cells.*/MBSLPhiGas",
                                    "//MB2P.*/MB2SuperLayerZ",
                                    "//MB2P.*/MB2SuperLayerZ/MB2SLZLayer_.*Cells",
                                    "//MB2P.*/MB2SuperLayerZ/MB2SLZLayer_.*Cells/MB2SLZGas"};
  Filter* currentFilter = nullptr;

  for (const auto& i : selections) {
    vector<string_view> toks = split(i, "/");
    auto const& filter = find_if(begin(filters_), end(filters_), [&](auto const& f) {
      auto const& k = find_if(begin(f->skeys), end(f->skeys), [&](auto const& p) { return toks.front() == p; });
      if (k != end(f->skeys)) {
        currentFilter = f.get();
        return true;
      }
      return false;
    });
    if (filter == end(filters_)) {
      bool isRegex = dd4hep::dd::isRegex(toks.front());
      filters_.emplace_back(make_unique<Filter>());
      filters_.back()->isRegex.emplace_back(isRegex);
      filters_.back()->hasNamespace.emplace_back(dd4hep::dd::hasNamespace(toks.front()));
      if (isRegex) {
        filters_.back()->index.emplace_back(filters_.back()->keys.size());
        filters_.back()->keys.emplace_back(std::regex(std::begin(toks.front()), std::end(toks.front())));
      } else {
        filters_.back()->index.emplace_back(filters_.back()->skeys.size());
      }
      filters_.back()->skeys.emplace_back(toks.front());
      filters_.back()->up = nullptr;
      filters_.back()->spec = nullptr;
      currentFilter = filters_.back().get();
    }
    // all next levels
    for (size_t pos = 1; pos < toks.size(); ++pos) {
      if (currentFilter->next != nullptr) {
        currentFilter = currentFilter->next.get();
        auto const& l = find_if(
            begin(currentFilter->skeys), end(currentFilter->skeys), [&](auto const& p) { return toks[pos] == p; });
        if (l == end(currentFilter->skeys)) {
          bool isRegex = dd4hep::dd::isRegex(toks[pos]);
          currentFilter->isRegex.emplace_back(isRegex);
          currentFilter->hasNamespace.emplace_back(dd4hep::dd::hasNamespace(toks[pos]));
          if (isRegex) {
            currentFilter->index.emplace_back(currentFilter->keys.size());
            currentFilter->keys.emplace_back(std::regex(std::begin(toks[pos]), std::end(toks[pos])));
          } else {
            currentFilter->index.emplace_back(currentFilter->skeys.size());
          }
          currentFilter->skeys.emplace_back(toks[pos]);
        }
      } else {
        auto filter = std::make_unique<Filter>();
        bool isRegex = dd4hep::dd::isRegex(toks[pos]);
        filter->isRegex.emplace_back(isRegex);
        filter->hasNamespace.emplace_back(dd4hep::dd::hasNamespace(toks[pos]));
        if (isRegex) {
          filter->index.emplace_back(filters_.back()->keys.size());
          filter->keys.emplace_back(std::regex(toks[pos].begin(), toks[pos].end()));
        } else {
          filter->index.emplace_back(filters_.back()->skeys.size());
        }
        filter->skeys.emplace_back(toks[pos]);
        filter->next = nullptr;
        filter->up = currentFilter;

        currentFilter->next = std::move(filter);
      }
    }
  }
}

void testFilter::checkFilter() {
  string_view name = "MB2P.*"sv;
  CPPUNIT_ASSERT(std::regex_match(name.begin(), name.end(), filters_.front()->keys.front()) == 1);
  CPPUNIT_ASSERT(filters_.front()->up == nullptr);
  CPPUNIT_ASSERT(filters_.front()->next != nullptr);
  CPPUNIT_ASSERT(filters_.size() == 1);

  Filter* current = nullptr;
  for (auto const& i : filters_) {
    current = i.get();
    do {
      for (auto const& i : current->skeys) {
        std::cout << i << ", ";
      }
      std::cout << "\n";
      current = current->next.get();
    } while (current != nullptr);
  }
}
