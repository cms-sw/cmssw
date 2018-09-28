#include <cppunit/extensions/HelperMacros.h>
#include <string>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

class testESInputTag: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testESInputTag);
  CPPUNIT_TEST(emptyTags);
  CPPUNIT_TEST(oneStringConstructor);
  CPPUNIT_TEST(twoStringConstructor);
  CPPUNIT_TEST(encodedTags);
  CPPUNIT_TEST(mixedConstructors);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void emptyTags();
  void oneStringConstructor();
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
  ESInputTag const empty3{"", ""};
  ESInputTag const empty4{"", ESInputTag::Encoded};
  ESInputTag const empty5{":", ESInputTag::Encoded};

  require_empty(empty1);
  require_empty(empty2);
  require_empty(empty3);
  require_empty(empty4);
  require_empty(empty5);

  // Equivalence
  CPPUNIT_ASSERT_EQUAL(empty1, empty2);
  CPPUNIT_ASSERT_EQUAL(empty1, empty3);
  CPPUNIT_ASSERT_EQUAL(empty1, empty4);
  CPPUNIT_ASSERT_EQUAL(empty1, empty5);
}

void testESInputTag::oneStringConstructor()
{
  ESInputTag const tag{"DL"};
  CPPUNIT_ASSERT(tag.module().empty());
  CPPUNIT_ASSERT_EQUAL(tag.data(), "DL"s);

  // Cannot have colons for the one-argument constructor
  CPPUNIT_ASSERT_THROW(ESInputTag{":DL"}, cms::Exception);
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

  ESInputTag const moduleOnly{"ML", ESInputTag::Encoded};
  ESInputTag const moduleOnlywToken{"ML:", ESInputTag::Encoded};
  ESInputTag const dataOnlywToken{":DL", ESInputTag::Encoded};
  ESInputTag const bothFields{"ML:DL", ESInputTag::Encoded};

  require_labels(moduleOnly, "ML", "");
  require_labels(moduleOnlywToken, "ML", "");
  require_labels(dataOnlywToken, "", "DL");
  require_labels(bothFields, "ML", "DL");

  // Too many colons
  CPPUNIT_ASSERT_THROW((ESInputTag{"ML:DL:", ESInputTag::Encoded}), cms::Exception);
}

void testESInputTag::mixedConstructors()
{
  // No module label
  ESInputTag const data_only_label{"DL"};
  CPPUNIT_ASSERT_EQUAL(data_only_label, (ESInputTag{"", "DL"}));
  CPPUNIT_ASSERT_EQUAL(data_only_label, (ESInputTag{":DL", ESInputTag::Encoded}));

  // No data label
  CPPUNIT_ASSERT_EQUAL((ESInputTag{"ML", ""}), (ESInputTag{"ML:", ESInputTag::Encoded}));

  // With module label
  CPPUNIT_ASSERT_EQUAL((ESInputTag{"ML", "DL"}), (ESInputTag{"ML:DL", ESInputTag::Encoded}));
}
