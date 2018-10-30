#include <cppunit/extensions/HelperMacros.h>
#include <string>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

class testESInputTag: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testESInputTag);
  CPPUNIT_TEST(emptyTags);
  CPPUNIT_TEST(twoStringConstructor);
  CPPUNIT_TEST(encodedTags);
  CPPUNIT_TEST(mixedConstructors);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void emptyTags();
  void twoStringConstructor();
  void encodedTags();
  void mixedConstructors();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testESInputTag);

using edm::ESInputTag;
using namespace std::string_literals;

void testESInputTag::emptyTags()
{
  auto require_empty = [](ESInputTag const& tag) {
    CPPUNIT_ASSERT(tag.module().empty());
    CPPUNIT_ASSERT(tag.data().empty());
  };

  ESInputTag const empty1{};
  ESInputTag const empty2{""};
  ESInputTag const empty3{":"};
  ESInputTag const empty4{"", ""};

  require_empty(empty1);
  require_empty(empty2);
  require_empty(empty3);
  require_empty(empty4);

  // Equivalence
  CPPUNIT_ASSERT_EQUAL(empty1, empty2);
  CPPUNIT_ASSERT_EQUAL(empty1, empty3);
  CPPUNIT_ASSERT_EQUAL(empty1, empty4);
}

void testESInputTag::twoStringConstructor()
{
  ESInputTag const tag{"ML", "DL"};
  CPPUNIT_ASSERT_EQUAL(tag.module(), "ML"s);
  CPPUNIT_ASSERT_EQUAL(tag.data(), "DL"s);
}

void testESInputTag::encodedTags()
{
  auto require_labels = [](ESInputTag const& tag,
                           std::string const& module_label,
                           std::string const& data_label) {
    CPPUNIT_ASSERT_EQUAL(tag.module(), module_label);
    CPPUNIT_ASSERT_EQUAL(tag.data(), data_label);
  };

  ESInputTag const moduleOnly{"ML"};
  ESInputTag const moduleOnlywToken{"ML:"};
  ESInputTag const dataOnlywToken{":DL"};
  ESInputTag const bothFields{"ML:DL"};

  require_labels(moduleOnly, "ML", "");
  require_labels(moduleOnlywToken, "ML", "");
  require_labels(dataOnlywToken, "", "DL");
  require_labels(bothFields, "ML", "DL");

  // Too many colons
  CPPUNIT_ASSERT_THROW((ESInputTag{"ML:DL:"}), cms::Exception);
}

void testESInputTag::mixedConstructors()
{
  // No module label
  CPPUNIT_ASSERT_EQUAL((ESInputTag{"", "DL"}), (ESInputTag{":DL"}));

  // No data label
  CPPUNIT_ASSERT_EQUAL((ESInputTag{"ML", ""}), (ESInputTag{"ML:"}));

  // With module label
  CPPUNIT_ASSERT_EQUAL((ESInputTag{"ML", "DL"}), (ESInputTag{"ML:DL"}));
}
