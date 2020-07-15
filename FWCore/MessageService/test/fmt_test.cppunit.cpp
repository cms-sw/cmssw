/*----------------------------------------------------------------------

Test program for fmt external.

 ----------------------------------------------------------------------*/

#include <fmt/format.h>
#include <cppunit/extensions/HelperMacros.h>

class test_fmt_external : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_fmt_external);

  CPPUNIT_TEST(test_fmt);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void test_fmt();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(test_fmt_external);

//using std::cerr;
//using std::endl;

struct date {
  int year, month, day;
};

template <>
struct fmt::formatter<date> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const date& d, FormatContext& ctx) {
    return format_to(ctx.out(), "{}-{}-{}", d.year, d.month, d.day);
  }
};

template <typename... Args>
auto capture(const Args&... args) {
  return std::make_tuple(args...);
}

auto print_message = [](const auto&... args) { fmt::print(args...); };

void test_fmt_external::test_fmt()

{
  std::string s = fmt::format("The date is {}", date{2012, 12, 9});
  std::string s_check = "The date is 2012-12-9";
  CPPUNIT_ASSERT(s_check == s);
  auto args = capture("{} {}", 42, "foo");
  std::apply(print_message, args);
  fmt::memory_buffer buf;
  format_to(buf, "{}", 42);  // replaces itoa(42, buffer, 10)
  fmt::print(buf.data());
  format_to(buf, "{:x}", 42);  // replaces itoa(42, buffer, 16)
  fmt::print(to_string(buf));
}
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
