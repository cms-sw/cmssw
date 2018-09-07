#include "DetectorDescription/DDCMS/interface/DDSingleton.h"

#include <iostream>
#include <string>
#include <chrono>

using namespace std;
using namespace cms;

struct StrSingleton : public DDSingleton<string, StrSingleton> {};
struct StrSingletonTwo : public DDSingleton<string, StrSingletonTwo>
{
  StrSingletonTwo() : DDSingleton(1) {}
  static std::unique_ptr<string> init() { return std::make_unique<string>( "Initialized" ); }
};

class TestClass
{
public:
  TestClass() : m_strVal("Hello World")
  {}

  void test1( int i )
  {
    StrSingleton::getInstance()->assign( std::to_string( i ));
  }

  void test2( int i )
  {
    m_str->assign( std::to_string( i ));
  }

  void test3( int i )
  {
    m_strVal.assign( std::to_string( i ));
  }
  
  void test4( int i )
  {
    *m_str = std::to_string( i );
  }
  
  void test5( int i )
  {
    m_str2->assign( std::to_string( i ));
  }
  
  void test6( int i )
  {
    *m_str2 = std::to_string( i );
  }

private:
  string m_strVal;
  StrSingleton m_str;
  StrSingleton m_str2;
};

class BenchmarkGrd
{
public:
  BenchmarkGrd( const std::string &name )
    : m_start( std::chrono::high_resolution_clock::now()),
      m_name( name )
  {}
  
  ~BenchmarkGrd()
  {
    std::chrono::duration< double, std::milli > diff = std::chrono::high_resolution_clock::now() - m_start;
    std::cout << "Benchmark '" << m_name << "' took " << diff.count() << " millis\n";
  }
  
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
  std::string m_name;
};

#ifdef BENCHMARK_ON
#   define BENCHMARK_START(X) {BenchmarkGrd(#X)
#   define BENCHMARK_END }
#else
#   define BENCHMARK_START(X)
#   define BENCHMARK_END
#endif

int main( int argc, char *argv[])
{
  TestClass tst;
  
  BENCHMARK_START( test1 );
  for( int i = 0; i < 1000000; ++i )
    tst.test1( i );
  BENCHMARK_END;
  
  BENCHMARK_START( test2 );
  for( int i = 0; i < 1000000; ++i )
    tst.test2( i );
  BENCHMARK_END;
  
  BENCHMARK_START( test3 );
  for( int i = 0; i < 1000000; ++i )
    tst.test3( i );
  BENCHMARK_END;
  
  BENCHMARK_START( test4 );
  for( int i = 0; i < 1000000; ++i )
    tst.test4( i );
  BENCHMARK_END;
  
  BENCHMARK_START( test5 );
  for( int i = 0; i < 1000000; ++i )
    tst.test5( i );
  BENCHMARK_END;
  
  BENCHMARK_START( test6 );
  for( int i = 0; i < 1000000; ++i )
    tst.test6( i );
  BENCHMARK_END;
  
  return 0;
}
