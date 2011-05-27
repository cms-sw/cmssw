#include "EventFilter/StorageManager/interface/XHTMLMonitor.h"
#include "EventFilter/StorageManager/interface/XHTMLMaker.h"

#include <string>
#include <iostream>
#include "boost/thread.hpp"

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

using namespace stor;

class testXHTMLmaker : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testXHTMLmaker);
  CPPUNIT_TEST(initialize_make_make_terminate);
  CPPUNIT_TEST(initialize_make_terminate_repeat);
  
  // It seems the Xerces-C++ is not thread safe, at least in the
  // manner in which we are using it. Activating the test below causes,
  // in a reproducible manner, a segmentation violation.
  // CPPUNIT_TEST(run_many_threads);

  CPPUNIT_TEST_SUITE_END();

public:
  testXHTMLmaker();
  void initialize_make_make_terminate();
  void initialize_make_terminate_repeat();
  void run_many_threads();

private:

  void make_a_page
  (
    XHTMLMaker& maker,
    const std::string& title,
    const std::string& h_text,
    const std::string& url,
    const std::string& url_text,
    const std::string& p_text
  );
  
  void initialize_make_terminate
  (
    const std::string& title,
    const std::string& h_text,
    const std::string& url,
    const std::string& url_text,
    const std::string& p_text,
    std::string& out
  );

  void make_one_make_two_write_one_write_two
  (
    std::string& one,
    std::string& two
  );
  
  void run_many_threads(unsigned num_thread_pairs);
  
  const std::string fixed_1;
  const std::string fixed_2;
};

testXHTMLmaker::testXHTMLmaker() :
fixed_1(
  "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\" ?>\n"
  "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n"
  "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n\n"
  "  <head>\n"
  "    <title>Fancy Page</title>\n"
  "    <link href=\"xhtmlmaker_t.css\" rel=\"stylesheet\"/>\n"
  "    <script src=\"xhtmlmaker_t.js\" type=\"text/javascript\"> </script>\n"
  "  </head>\n\n"
    "  <body>\n"
  "    <h1>Kinda Fancy Page</h1>\n"
  "    <p>Hi there\n"
  "      <a href=\"http://www.google.com\" id=\"my_id\">Fancy Link</a>\n"
  "    </p>\n"
  "  </body>\n\n"
  "</html>\n"),
fixed_2(
  "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\" ?>\n"
  "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n"
  "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n\n"
  "  <head>\n"
  "    <title>Another Page</title>\n"
  "    <link href=\"xhtmlmaker_t.css\" rel=\"stylesheet\"/>\n"
  "    <script src=\"xhtmlmaker_t.js\" type=\"text/javascript\"> </script>\n"
  "  </head>\n\n"
  "  <body>\n"
  "    <h1>Not So Fancy Page</h1>\n"
  "    <p>Get lost\n"
  "      <a href=\"http://www.cern.ch\" id=\"my_id\">No Fun Here</a>\n"
  "    </p>\n"
  "  </body>\n\n"
  "</html>\n")
{}

void testXHTMLmaker::make_a_page
(
  XHTMLMaker& maker,
  const std::string& title,
  const std::string& h_text,
  const std::string& url,
  const std::string& url_text,
  const std::string& p_text
)
{
  XHTMLMaker::Node* body = maker.start( title );
  
  XHTMLMaker::AttrMap lmap;
  lmap[ "rel" ] = "stylesheet";
  lmap[ "href" ] = "xhtmlmaker_t.css";
  maker.addNode( "link", maker.getHead(), lmap );
  
  XHTMLMaker::AttrMap smap;
  smap[ "type" ] = "text/javascript";
  smap[ "src" ] = "xhtmlmaker_t.js";
  XHTMLMaker::Node* s = maker.addNode( "script", maker.getHead(), smap );
  maker.addText( s, " " );
  
  
  XHTMLMaker::Node* h1 = maker.addNode( "h1", body );
  maker.addText( h1, h_text );
  
  XHTMLMaker::AttrMap pmap;
  XHTMLMaker::Node* p = maker.addNode( "p", body );
  maker.addText( p, p_text );
  
  XHTMLMaker::AttrMap m;
  m[ "href" ] = url;
  m[ "id" ] = "my_id";
  XHTMLMaker::Node* link =
    maker.addNode( "a", p, m );
  maker.addText( link, url_text );
}


void testXHTMLmaker::initialize_make_terminate
(
  const std::string& title,
  const std::string& h_text,
  const std::string& url,
  const std::string& url_text,
  const std::string& p_text,
  std::string& out
)
{
  XHTMLMonitor monitor;

  XHTMLMaker maker;

  make_a_page( maker,
	       title,
	       h_text,
	       url,
	       url_text,
	       p_text );

  maker.out( out );
}

void testXHTMLmaker::make_one_make_two_write_one_write_two
(
  std::string& one,
  std::string& two
)
{
  XHTMLMonitor monitor;

  XHTMLMaker maker_one;

  make_a_page( maker_one,
	       "Fancy Page",
	       "Kinda Fancy Page",
	       "http://www.google.com",
	       "Fancy Link",
	       "Hi there" );

  XHTMLMaker maker_two;

  make_a_page( maker_two,
	       "Another Page",
	       "Not So Fancy Page",
	       "http://www.cern.ch",
	       "No Fun Here",
	       "Get lost" );

  maker_one.out( one );

  maker_two.out( two );
}

void testXHTMLmaker::initialize_make_make_terminate()
{
  std::string one;
  std::string two;
  make_one_make_two_write_one_write_two( one, two );
  CPPUNIT_ASSERT( one == fixed_1 );
  CPPUNIT_ASSERT( two == fixed_2 );
}

void testXHTMLmaker::initialize_make_terminate_repeat()
{
  std::string one;
  initialize_make_terminate( "Fancy Page",
			     "Kinda Fancy Page",
			     "http://www.google.com",
			     "Fancy Link",
			     "Hi there",
			     one );
  CPPUNIT_ASSERT( one == fixed_1 );

  std::string two;
  initialize_make_terminate( "Another Page",
			     "Not So Fancy Page",
			     "http://www.cern.ch",
			     "No Fun Here",
			     "Get lost",
			     two );
  CPPUNIT_ASSERT( two == fixed_2 );
}

void testXHTMLmaker::run_many_threads(unsigned num_thread_pairs)
{
  boost::thread_group threads;
  for (unsigned i = 0; i != num_thread_pairs; ++i)
  {
    threads.add_thread(new boost::thread(
        boost::bind( &testXHTMLmaker::initialize_make_make_terminate, this )));
    threads.add_thread(new boost::thread(
        boost::bind( &testXHTMLmaker::initialize_make_terminate_repeat, this )));
  }
  
  threads.join_all();
}

void testXHTMLmaker::run_many_threads()
{
  run_many_threads(1000);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testXHTMLmaker);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
