#include <catch2/catch_all.hpp>
#include <string>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

using edm::ESInputTag;
using namespace std::string_literals;

namespace {
  void require_empty(ESInputTag const& tag) {
    REQUIRE(tag.module().empty());
    REQUIRE(tag.data().empty());
  };

  void require_labels(ESInputTag const& tag, std::string const& module_label, std::string const& data_label) {
    REQUIRE(tag.module() == module_label);
    REQUIRE(tag.data() == data_label);
  };
}  // namespace

TEST_CASE("ESInputTag behavior", "[ESInputTag]") {
  SECTION("emptyTags") {
    ESInputTag const empty1{};
    ESInputTag const empty2{""};
    ESInputTag const empty3{":"};
    ESInputTag const empty4{"", ""};

    require_empty(empty1);
    require_empty(empty2);
    require_empty(empty3);
    require_empty(empty4);

    // Equivalence
    REQUIRE(empty1 == empty2);
    REQUIRE(empty1 == empty3);
    REQUIRE(empty1 == empty4);
  }

  SECTION("twoStringConstructor") {
    ESInputTag const tag{"ML", "DL"};
    REQUIRE(tag.module() == "ML");
    REQUIRE(tag.data() == "DL");
  }

  SECTION("encodedTags") {
    ESInputTag const moduleOnlywToken{"ML:"};
    ESInputTag const dataOnlywToken{":DL"};
    ESInputTag const bothFields{"ML:DL"};

    require_labels(moduleOnlywToken, "ML", "");
    require_labels(dataOnlywToken, "", "DL");
    require_labels(bothFields, "ML", "DL");

    // Too many colons
    REQUIRE_THROWS_AS((ESInputTag{"ML:DL:"}), cms::Exception);
    REQUIRE_THROWS_AS((ESInputTag{"ML"}), cms::Exception);
  }

  SECTION("mixedConstructors") {
    // No module label
    REQUIRE((ESInputTag{"", "DL"}) == (ESInputTag{":DL"}));

    // No data label
    REQUIRE((ESInputTag{"ML", ""}) == (ESInputTag{"ML:"}));

    // With module label
    REQUIRE((ESInputTag{"ML", "DL"}) == (ESInputTag{"ML:DL"}));
  }
}
