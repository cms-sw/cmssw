/** \file PerfTools/EdmEvent/interface/EdmEventSize.cc
 *
 *  \author Vincenzo Innocente
 */
#include "PerfTools/EdmEvent/interface/EdmEventSize.h"
#include <valarray>
#include <functional>
#include <algorithm>
#include <boost/bind.hpp>
#include <ostream>
#include <limits>
#include <assert.h>

#include "Rtypes.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TObjArray.h"
#include "TBranch.h"
#include "TH1.h"
#include "TCanvas.h"
#include "Riostream.h"

#include "TBufferFile.h"

namespace {

  enum Indices {kUncompressed,kCompressed};

  typedef std::valarray<Long64_t> size_type; 

  size_type getBasketSize( TBranch *);
  
  size_type getBasketSize( TObjArray * branches) {
    size_type result(static_cast<Long64_t>(0),2);
    size_t n = branches->GetEntries();
    for( size_t i = 0; i < n; ++ i ) {
      TBranch * b = dynamic_cast<TBranch*>( branches->At( i ) );
      assert( b != 0 );
      result += getBasketSize(b);
    }
    return result;
  }
  
  size_type getBasketSize( TBranch * b) {
    size_type result(static_cast<Long64_t>(0),2);
    if ( b != 0 ) {
      if ( b->GetZipBytes() > 0 ) {
	result[kUncompressed]  = b->GetTotBytes();  result[kCompressed] = b->GetZipBytes();
      } else {
	result[kUncompressed] = b->GetTotalSize(); result[kCompressed] = b->GetTotalSize();
      }
      result += getBasketSize( b->GetListOfBranches() );
    }
    return result;
  }


  size_type getTotalSize( TBranch * br) {
    TBufferFile buf( TBuffer::kWrite, 10000 );
    TBranch::Class()->WriteBuffer( buf, br );
    size_type size = getBasketSize(br);
    if ( br->GetZipBytes() > 0 )
      size[kUncompressed] += buf.Length();
    return size;
  }
}

namespace perftools {

  EdmEventSize::EdmEventSize() : 
    m_nEvents(0) {}
  
  EdmEventSize::EdmEventSize(std::string const & fileName, std::string const & treeName ) : 
    m_nEvents(0) {
    parseFile(fileName);
  }
  
  void EdmEventSize::parseFile(std::string const & fileName, std::string const & treeName) {
    m_fileName = fileName;
    m_branches.clear();

    TFile * file = TFile::Open( fileName.c_str() );
    if( file==0  || ( !(*file).IsOpen() ) )
      throw Error( "unable to open data file " + fileName, 7002);
    
    TObject * o = file->Get(treeName.c_str() );
    if ( o == 0 )
      throw Error("no object \"" + treeName + "\" found in file: " + fileName, 7003);
    
    TTree * events = dynamic_cast<TTree*> (o);
    if ( events == 0 )
      throw Error("object \"" + treeName + "\" is not a TTree in file: " + fileName, 7004);
    
    m_nEvents = events->GetEntries();
    if ( m_nEvents == 0 )
      throw Error("tree \"" + treeName + "\" in file " + fileName + " contains no Events", 7005);


    TObjArray * branches = events->GetListOfBranches();
    if ( branches == 0 )
      throw Error("tree \"" + treeName+ "\" in file " + fileName + " contains no branches", 7006);
    
    const size_t n =  branches->GetEntries();
    m_branches.reserve(n);
    for( size_t i = 0; i < n; ++i ) {
      TBranch * b = dynamic_cast<TBranch*>( branches->At( i ) );
      if (b==0) continue;
      std::string const name( b->GetName() );
      if ( name == "EventAux" ) continue;
      size_type s = getTotalSize(b);
      m_branches.push_back( BranchRecord(name, double(s[kCompressed])/double(m_nEvents), double(s[kUncompressed])/double(m_nEvents)) );
    }
    std::sort(m_branches.begin(),m_branches.end(), 
	      boost::bind(std::greater<double>(),
			  boost::bind(&BranchRecord::compr_size,_1),
			  boost::bind(&BranchRecord::compr_size,_2))
	      );

  }
  
  void EdmEventSize::sortAlpha() {
    std::sort(m_branches.begin(),m_branches.end(), 
	      boost::bind(std::less<std::string>(),
			  boost::bind(&BranchRecord::name,_1),
			  boost::bind(&BranchRecord::name,_2))
	      );

  }

  namespace detail {
    // format as product:label (type)
    void shorterName(EdmEventSize::BranchRecord & br) {
      size_t b = br.fullName.find('_');
      size_t e = br.fullName.rfind('_');
      if (b==e) br.name=br.fullName;
      else {
	// remove type and process
	br.name = br.fullName.substr(b+1,e-b-1);
	// change label separator in :
	e = br.name.rfind('_');
	if (e!=std::string::npos) br.name.replace(e,1,":");
	// add the type name
	br.name.append(" ("+br.fullName.substr(0,b)+")");
      }
    }

  }
  
  void EdmEventSize::formatNames() {
    std::for_each(m_branches.begin(),m_branches.end(),
		  &detail::shorterName);
  }



  namespace detail {

    void dump(std::ostream& co, EdmEventSize::BranchRecord const & br) {
      co << br.name << " " <<  br.uncompr_size <<  " " << br.compr_size << "\n"; 
    }
  }

  
  void EdmEventSize::dump(std::ostream & co, bool header) const {
    if (header) {
      co << "File " << m_fileName << " Events " << m_nEvents << "\n";
      co <<"Branch Name | Average Uncompressed Size (Bytes/Event) | Average Compressed Size (Bytes/Event) \n";
    }
    std::for_each(m_branches.begin(),m_branches.end(),
		  boost::bind(detail::dump,boost::ref(co),_1));
  }

  namespace detail {

    struct Hist {

      explicit Hist(int itop) : 
	top(itop),
	uncompressed( "uncompressed", "branch sizes", top, -0.5, - 0.5 + top ),
	compressed( "compressed", "branch sizes", top, -0.5, - 0.5 + top ),
	cxAxis(compressed.GetXaxis()),
	uxAxis(uncompressed.GetXaxis()),
	x(0) {}
      
      void fill(EdmEventSize::BranchRecord const & br) {
	if ( x < top ) {
	  cxAxis->SetBinLabel( x + 1, br.name.c_str() );
	  uxAxis->SetBinLabel( x + 1, br.name.c_str() );
	  compressed.Fill( x, br.compr_size );
	  uncompressed.Fill( x, br.uncompr_size );
	  x++;
	}
      }

      void finalize() {
	double mn = std::numeric_limits<double>::max();
	for( int i = 1; i <= top; ++i ) {
	  double cm = compressed.GetMinimum( i ), um = uncompressed.GetMinimum( i );
	  if ( cm > 0 && cm < mn ) mn = cm;
	  if ( um > 0 && um < mn ) mn = um;
	}
	mn *= 0.8;
	double mx = std::max( compressed.GetMaximum(), uncompressed.GetMaximum() );
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
	
      }
      
      int top;
      TH1F uncompressed;
      TH1F compressed;
      TAxis * cxAxis;
      TAxis * uxAxis;
      
      int x;
    };
  
  }
  
  void EdmEventSize::produceHistos(std::string const & plot, std::string const & file, int top) const {
    if (top==0) top = m_branches.size();
    detail::Hist h(top);
    std::for_each(m_branches.begin(),m_branches.end(),
		  boost::bind(&detail::Hist::fill,boost::ref(h),_1));
    h.finalize();
    if( !plot.empty() ) {
      gROOT->SetStyle( "Plain" );
      gStyle->SetOptStat( kFALSE );
      gStyle->SetOptLogy();
      TCanvas c;
      h.uncompressed.Draw();
      h.compressed.Draw( "same" );
      c.SaveAs( plot.c_str() );
    }
    if ( !file.empty() ) {
      TFile f( file.c_str(), "RECREATE" );
      h.compressed.Write();
      h.uncompressed.Write();
      f.Close();
    }

  }

}
