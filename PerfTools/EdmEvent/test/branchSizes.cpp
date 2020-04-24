/** measure branch sizes
 *
 * author Luca Lista
 *
 */

#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <utility>
#include <cassert>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <TStyle.h>
#include <TObjArray.h>
#include <TBranch.h>
#include <TH1.h>
#include <TCanvas.h>
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include <utility>

#include "TBufferFile.h"

using namespace std;

static const char * const kHelpOpt = "help";
static const char * const kHelpCommandOpt = "help,h";
static const char * const kDataFileOpt = "data-file";
static const char * const kDataFileCommandOpt = "data-file,d";
static const char * const kAutoLoadOpt ="auto-loader";
static const char * const kAutoLoadCommandOpt ="auto-loader,a";
static const char * const kPlotOpt ="plot";
static const char * const kPlotCommandOpt ="plot,p";
static const char * const kSavePlotOpt ="save-plot";
static const char * const kSavePlotCommandOpt ="save-plot,s";
static const char * const kPlotTopOpt ="plot-top";
static const char * const kPlotTopCommandOpt ="plot-top,t";
static const char * const kVerboseOpt = "verbose";
static const char * const kVerboseCommandOpt = "verbose,v";
static const char * const kAlphabeticOrderOpt ="alphabetic-order";
static const char * const kAlphabeticOrderCommandOpt ="alphabetic-order,A";

typedef pair<size_t, size_t> size_type;

typedef pair<string, size_type> BranchRecord;

typedef vector<BranchRecord> BranchVector;

ostream & operator << ( ostream & out, const size_type & s ) {
  out << s.first << '/' << s.second << " bytes";
  if ( s.second != 0 ) 
    out << " (compr: " << double( s.first ) / double ( s.second ) << ")";
  return out;
}

size_type & operator+=( size_type & s1, const size_type & s2 ) {
  s1.first += s2.first;
  s1.second += s2.second;
  return s1;
}

size_type GetTotalSize( TBranch *, bool verbose );

size_type GetBasketSize( TBranch *, bool verbose );

size_type GetBasketSize( TObjArray * branches, bool verbose ) {
  size_type result = make_pair( 0, 0 );
  size_t n = branches->GetEntries();
  for( size_t i = 0; i < n; ++ i ) {
    TBranch * b = dynamic_cast<TBranch*>( branches->At( i ) );
    assert( b != 0 );
    result += GetBasketSize( b, verbose );
  }
  return result;
}

size_type GetBasketSize( TBranch * b, bool verbose ) {
  size_type result = make_pair( 0, 0 );
  if ( b != 0 ) {
    if ( b->GetZipBytes() > 0 ) {
      result = make_pair( b->GetTotBytes(), b->GetZipBytes() );
    } else {
      result = make_pair( b->GetTotalSize(), b->GetTotalSize() );
    }
    if ( verbose )
      cout << " branch: " << b->GetName() << ", size:" << result.first << "/" << result.second << endl;
    result += GetBasketSize( b->GetListOfBranches(), verbose );
  }
  return result;
}

size_type GetTotalSize( TBranch * br, bool verbose ) {
  TBufferFile buf( TBuffer::kWrite, 10000 );
  TBranch::Class()->WriteBuffer( buf, br );
  size_type size = GetBasketSize( br, verbose );
  if ( br->GetZipBytes() > 0 )
    size.first += buf.Length();
  if ( verbose )
    cout << ">>> total branch size: " << br->GetName() << ":" << size.first << "/" << size.second << endl;  
  return size;
}

size_type GetTotalSize( TObjArray * branches, bool verbose ) {
  size_type result = make_pair( 0, 0 );
  size_t n = branches->GetEntries();
  for( size_t i = 0; i < n; ++ i ) {
    result += GetTotalSize( dynamic_cast<TBranch*>( branches->At( i ) ), verbose );
  }
  return result;
}

size_type GetTotalSize( TTree *t ) {
  size_t total = t->GetTotBytes();
  TBufferFile b(TBuffer::kWrite, 10000);
  TTree::Class()->WriteBuffer(b, t);
  total += b.Length();
  return make_pair( total, t->GetZipBytes() );
} 

size_type GetTotalBranchSize( TTree *t, bool verbose ) {
  return GetTotalSize( t->GetListOfBranches(), verbose );
} 

struct sortByCompressedSize {
  bool operator()( const BranchRecord & t1, const BranchRecord & t2 ) const {
    size_t s1 = t1.second.second, s2 = t2.second.second;
    if ( s1 == 0 && s2 == 0 ) {
      s1 = t1.second.first; s2 = t2.second.first;
    }
    return s1 > s2;
  }
};

struct sortByName {
  bool operator()( const BranchRecord & t1, const BranchRecord & t2 ) const {
    return t1.first < t2.first;
  }
};

int main( int argc, char * argv[] ) {
  using namespace boost::program_options;
  using namespace std;

  string programName( argv[ 0 ] );
  string descString( programName );
  descString += " [options] ";
  descString += "data_file \nAllowed options";
  options_description desc( descString );

  desc.add_options()
    ( kHelpCommandOpt, "produce help message" )
    ( kAutoLoadCommandOpt, "automatic library loading (avoid root warnings)" )
    ( kDataFileCommandOpt, value<string>(), "data file" )
    ( kAlphabeticOrderCommandOpt, "sort by alphabetic order (default: sort by size)" )
    ( kPlotCommandOpt, value<string>(), "produce a summary plot" )
    ( kPlotTopCommandOpt, value<int>(), "plot only the <arg> top size branches" )
    ( kSavePlotCommandOpt, value<string>(), "save plot into root file <arg>" )
    ( kVerboseCommandOpt, "verbose printout" );

  positional_options_description p;

  p.add( kDataFileOpt, -1 );

  variables_map vm;
  try {
    store( command_line_parser(argc,argv).options(desc).positional(p).run(), vm );
    notify( vm );
  } catch( const error& ) {
    return 7000;
  }

  if( vm.count( kHelpOpt ) ) {
    cout << desc <<std::endl;
    return 0;
  }

  if( ! vm.count( kDataFileOpt ) ) {
    string shortDesc("ConfigFileNotFound");
    cerr << programName << ": no data file given" << endl;
    return 7001;
  }

  gROOT->SetBatch();
  
  if( vm.count( kAutoLoadOpt ) != 0 ) {
    gSystem->Load( "libFWCoreFWLite" );
    FWLiteEnabler::enable();
  }

  string fileName = vm[kDataFileOpt].as<string>();
  TFile file( fileName.c_str() );
  if( ! file.IsOpen() ) {
    cerr << programName << ": unable to open data file " << fileName << endl;
    return 7002;
  }

  TObject * o = file.Get( "Events" );
  if ( o == 0 ) {
    cerr << programName << ": no object \"Events\" found in file: " << fileName << endl;
    return 7003;
  }

  TTree * events = dynamic_cast<TTree*>( o );
  if ( events == 0 ) {
    cerr << programName << ": object \"Events\" is not a TTree in file: " << fileName << endl;
    return 7004;
  }

  TObjArray * branches = events->GetListOfBranches();
  if ( branches == 0 ) {
    cerr << programName << ": tree \"Events\" in file " << fileName 
	 << " contains no branches" << endl;
    return 7004;
  }

  bool verbose = vm.count( kVerboseOpt ) > 0;

  BranchVector v;
  const size_t n =  branches->GetEntries();
  cout << fileName << " has " << n << " branches" << endl;
  for( size_t i = 0; i < n; ++i ) {
    TBranch * b = dynamic_cast<TBranch*>( branches->At( i ) );
    assert( b != 0 );
    string name( b->GetName() );
    if ( name == "EventAux" ) continue;
    size_type s = GetTotalSize( b, verbose );
    v.push_back( make_pair( b->GetName(), s ) );
  }
  if ( vm.count( kAlphabeticOrderOpt ) ) {
    sort( v.begin(), v.end(), sortByName() );
  } else {
    sort( v.begin(), v.end(), sortByCompressedSize() );
  }
  bool plot = ( vm.count( kPlotOpt ) > 0 );
  bool save = ( vm.count( kSavePlotOpt ) > 0 );
  int top = n;
  if( vm.count( kPlotTopOpt ) > 0 ) top = vm[ kPlotTopOpt ].as<int>();
  TH1F uncompressed( "uncompressed", "branch sizes", top, -0.5, - 0.5 + top );
  TH1F compressed( "compressed", "branch sizes", top, -0.5, - 0.5 + top );
  int x = 0;
  TAxis * cxAxis = compressed.GetXaxis();
  TAxis * uxAxis = uncompressed.GetXaxis();

  for( BranchVector::const_iterator b = v.begin(); b != v.end(); ++ b ) {
    const string & name = b->first;
    size_type size = b->second;
    cout << size << " " << name << endl;
    if ( x < top ) {
      cxAxis->SetBinLabel( x + 1, name.c_str() );
      uxAxis->SetBinLabel( x + 1, name.c_str() );
      compressed.Fill( x, size.second );
      uncompressed.Fill( x, size.first );
      x++;
    }
  }
  //  size_type branchSize = GetTotalBranchSize( events );
  //  cout << "total branches size: " << branchSize.first << " bytes (uncompressed), " 
  //       << branchSize.second << " bytes (compressed)"<< endl;
  size_type totalSize = GetTotalSize( events );
  cout << "total tree size: " << totalSize.first << " bytes (uncompressed), " 
       << totalSize.second << " bytes (compressed)"<< endl;
  double mn = DBL_MAX;
  for( int i = 1; i <= top; ++i ) {
    double cm = compressed.GetMinimum( i ), um = uncompressed.GetMinimum( i );
    if ( cm > 0 && cm < mn ) mn = cm;
    if ( um > 0 && um < mn ) mn = um;
  }
  mn *= 0.8;
  double mx = max( compressed.GetMaximum(), uncompressed.GetMaximum() );
  mx *= 1.2;
  uncompressed.SetMinimum( mn );
  uncompressed.SetMaximum( mx );
  compressed.SetMinimum( mn );
  //  compressed.SetMaximum( mx );
  cxAxis->SetLabelOffset( -0.32 );
  cxAxis->LabelsOption( "v" );
  cxAxis->SetLabelSize( 0.03 );
  uxAxis->SetLabelOffset( -0.32 );
  uxAxis->LabelsOption( "v" );
  uxAxis->SetLabelSize( 0.03 );
  compressed.GetYaxis()->SetTitle( "Bytes" );
  compressed.SetFillColor( kBlue );
  compressed.SetLineWidth( 2 );
  uncompressed.GetYaxis()->SetTitle( "Bytes" );
  uncompressed.SetFillColor( kRed );
  uncompressed.SetLineWidth( 2 );
  if( plot ) {
    string plotName = vm[kPlotOpt].as<string>();
    gROOT->SetStyle( "Plain" );
    gStyle->SetOptStat( kFALSE );
    gStyle->SetOptLogy();
    TCanvas c;
    uncompressed.Draw();
    compressed.Draw( "same" );
    c.SaveAs( plotName.c_str() );
  }
  if ( save ) {
    string fileName = vm[kSavePlotOpt].as<string>();
    TFile f( fileName.c_str(), "RECREATE" );
    compressed.Write();
    uncompressed.Write();
    f.Close();
  }
  return 0;
}
