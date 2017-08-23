// -*- C++ -*-
//
// Package:    TestCompareDDSpecsDumpFiles
// Class:      TestCompareDDSpecsDumpFiles
// 
/**\class TestCompareDDSpecsDumpFiles TestCompareDDSpecsDumpFiles.cc test/TestCompareDDSpecsDumpFiles/src/TestCompareDDSpecsDumpFiles.cc

 Description: Compares two SpecPars dump files 

 Implementation:
     Read two files with a certain format and compare each line.
**/
//
// Original Author:  Ianna Osborne
//         Created:  Thu Dec 2, 2010
//
//

#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class TestCompareDDSpecsDumpFiles
  : public edm::one::EDAnalyzer<>
{
public:
  explicit TestCompareDDSpecsDumpFiles( const edm::ParameterSet& );
  ~TestCompareDDSpecsDumpFiles( void ) override;
  
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

private:
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

  std::string clean( const std::string& in );
  std::string merge( const std::list<std::string>& list );
  std::string fillAndSort( const tokenizer::iterator& start, const tokenizer::iterator& end, std::list<std::string>& list );
  std::string preFill( const tokenizer::iterator& it, std::list<std::string>& list );

  std::string fname1_;
  std::string fname2_;
  double tol_;
  std::ifstream f1_;
  std::ifstream f2_;
};

TestCompareDDSpecsDumpFiles::TestCompareDDSpecsDumpFiles( const edm::ParameterSet& ps ) 
  : fname1_( ps.getParameter<std::string>( "dumpFile1" )),
    fname2_( ps.getParameter<std::string>( "dumpFile2" )),
    tol_( ps.getUntrackedParameter<double>( "tolerance", 0.000001 )),
    f1_( fname1_.c_str(), std::ios::in ),
    f2_( fname2_.c_str(), std::ios::in )
{
  if( !f1_ || !f2_ )
  {
    throw cms::Exception( "MissingFileDDTest" ) << fname1_ << " and/or " << fname2_ << " do not exist.";
  }
}

TestCompareDDSpecsDumpFiles::~TestCompareDDSpecsDumpFiles( void )
{
  f1_.close();
  f2_.close();
}

std::string
TestCompareDDSpecsDumpFiles::clean( const std::string& in )
{
  std::string str( in );
  boost::trim( str );
  size_t found1;
  found1 = str.find( "/" );
  if( found1 != std::string::npos )
  {
    size_t found2;
    found2 = str.find( " " );
    if( found2 != std::string::npos )
    {
      str.erase( found1, found2 );
    }
	  
    boost::trim( str );
  }

  return str;
}

std::string
TestCompareDDSpecsDumpFiles::preFill( const tokenizer::iterator& it, std::list<std::string>& list )
{
  boost::char_separator<char> space( " " );

  tokenizer firstString( *it, space );
  tokenizer::iterator fit = firstString.begin();
  std::string str( *it );
  str.erase( 0, ( *fit ).size());
  boost::trim( str );
  list.emplace_back( clean( str ));

  return *fit;
}

std::string
TestCompareDDSpecsDumpFiles::merge( const std::list<std::string>& list )
{
  std::string str( "" );
  for(const auto & it : list)
  {
    str.append( it );
    str.append("|");
  }

  return str;
}

std::string
TestCompareDDSpecsDumpFiles::fillAndSort( const tokenizer::iterator& start, const tokenizer::iterator& end, std::list<std::string>& list )
{
  for( tokenizer::iterator it = start; it != end; ++it )
  {
    list.emplace_back( clean( *it ));
  }     
  list.sort();

  return merge( list );
}

void
TestCompareDDSpecsDumpFiles::analyze( const edm::Event&, const edm::EventSetup& )
{  
  std::string l1, l2;
  boost::char_separator<char> sep( "|" );

  int line = 0;
  
  while( !f1_.eof() && !f2_.eof() )
  {
    getline( f1_, l1 );
    getline( f2_, l2 );

    if( l1.empty() && l2.empty())
      continue;
    
    tokenizer tokens1( l1, sep );
    std::list< std::string > items1;
    tokenizer::iterator tok_iter1 = tokens1.begin();
    std::string firstStr1 = preFill( tok_iter1, items1 );
    ++tok_iter1;
    
    tokenizer tokens2( l2, sep );
    std::list< std::string > items2;
    tokenizer::iterator tok_iter2 = tokens2.begin();
    std::string firstStr2 = preFill( tok_iter2, items2 );
    ++tok_iter2;

    edm::LogInfo( "TestCompareDDSpecsDumpFiles" )
      << "#" << ++line
      << " Comparing " << firstStr1 << " " << firstStr1.size()
      << " with " << firstStr2 << " " << firstStr2.size() << " : ";
    
    if( firstStr1 != firstStr2 )
    {
      edm::LogError( "TestCompareDDSpecsDumpFiles" ) << ">>>>>> Cannot compare lines!!!!" << "\n";
    }

    // If the lines do not match, they may need sorting.
    if( l1 != l2 )
    {
      // The first cleaned token is already in the list.
      std::string sl1 = fillAndSort( tok_iter1, tokens1.end(), items1 );
      std::string sl2 = fillAndSort( tok_iter2, tokens2.end(), items2 );

      // Compare sorted lines.
      if( sl1 != sl2 )
      {
	edm::LogError( "TestCompareDDSpecsDumpFiles" )
	  << "#" << line << " Lines don't match.\n"
	  << "["<< l1 <<"]\n"
	  << "["<< l2 <<"]\n";
	
	// Remove common tokens.
	tokenizer sl1tokens( sl1, sep );
	for( tokenizer::iterator it = sl1tokens.begin(); it != sl1tokens.end(); ++it )
	{
	  std::string str( *it );
	  str.append( "|" );
	  size_t found;
	  found = sl2.find( str );
	  if( found == std::string::npos )
	  {
	    str.erase( remove( str.begin(), str.end(), '|' ), str.end());
	    edm::LogError( "TestCompareDDSpecsDumpFiles" ) << "<<<<<===== " << str << "\n";
	  }	  
	  else
	  {
	    sl2.erase( found, (*it).size());
	  }
	}
	// Print what's left.
	tokenizer sl2tokens( sl2, sep );
	for( tokenizer::iterator it = sl2tokens.begin(); it != sl2tokens.end(); ++it )
	  edm::LogError( "TestCompareDDSpecsDumpFiles" ) << "=====>>>>> " << *it << "\n";
      }
      else
	edm::LogInfo( "TestCompareDDSpecsDumpFiles" ) << " OK." << "\n";
    }
    else
      edm::LogInfo( "TestCompareDDSpecsDumpFiles" ) << " OK." << "\n";
  }
}

DEFINE_FWK_MODULE( TestCompareDDSpecsDumpFiles );
