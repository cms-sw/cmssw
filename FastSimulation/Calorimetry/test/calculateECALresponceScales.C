#include<iomanip> // provides setprecision
#include<iostream>
#include<sstream>
#include<string>

// ROOT
#include<TChain.h>
#include<TFile.h>
#include<TH3F.h>
#include<TMath.h>
#include<TProfile.h>
#include<TROOT.h>

using namespace std;

std::map<int,std::vector<float>> getVectorsFromTree( TChain& tree, const TH3F& h, float minR ) {
  /* map.first:  global bin in TH3F h, belonging to e,eta,-1
   * map.second: vector of r in this bin (with this e,eta)
   */

  std::map<int,std::vector<float>> vectors;

  float r,e,eta;
  tree.SetBranchAddress("r",&r);
  tree.SetBranchAddress("e",&e);
  tree.SetBranchAddress("eta",&eta);
  for( int i=0; i<tree.GetEntries(); i++ ) {
    tree.GetEntry(i);
    if( r < minR ) continue;
    int bin = h.FindFixBin( e, eta, -1 );

    // check if this bin already exists in map
    if( !vectors.count( bin ) ) {
      vectors[bin] = std::vector<float>();
    }

    vectors.at(bin).push_back( r );
  }

  for( auto& m : vectors ) {
    std::sort( m.second.begin(), m.second.end() );
  }

  return vectors;
}

void fillEmptyBinsWithUnity( TH3F& h3 ) {
  for( int x=0; x<h3.GetNbinsX()+2; x++ ) {
    for( int y=0; y<h3.GetNbinsY()+2; y++ ) {
      for( int z=0; z<h3.GetNbinsZ()+2; z++ ) {
        if( h3.GetBinContent(x,y,z) < 1e-5 ) {
          h3.SetBinContent(x,y,z,1.);
        }
      }
    }
  }
}

class KKFactorsFactory {
  public:

  KKFactorsFactory( const TH3F& h3_, const string& fileNameFast, const string& fileNameFull, const string& treeName ):
      h3(h3_),
      chFast(treeName.c_str()),
      chFull(treeName.c_str())
    {
      chFast.AddFile( fileNameFast.c_str() );
      chFull.AddFile( fileNameFull.c_str() );
    }

  void calculate(){
    float minR = h3.GetZaxis()->GetXmin();

    auto fastvectors = getVectorsFromTree( chFast, h3, minR );
    auto fullvectors = getVectorsFromTree( chFull, h3, minR );

    for( auto m : fastvectors ) {

      if( ! fullvectors.count( m.first ) ) {
        std::cout << "no compatible vector found for fullsim" << std::endl;
        continue;
      }

      auto fastvector = m.second;
      auto fullvector = fullvectors.at( m.first );

      TProfile profile("profile", "title", h3.GetZaxis()->GetNbins(), h3.GetZaxis()->GetXmin(), h3.GetZaxis()->GetXmax(), "s" );
      for( unsigned i=0; i<fastvector.size(); i++ ) {
        auto fa = fastvector[i];
        auto jRel = 1.*i/fastvector.size()*fullvector.size();
        int j = (int) jRel;
        auto fu = fullvector[j]*(jRel-j)+fullvector[j+1]*(j-jRel+1);
        if( fa ) profile.Fill( fa, fu/fa );
      }

      int xbin, ybin, zbin;
      h3.GetBinXYZ( m.first, xbin, ybin, zbin );

      for( int i=0; i<profile.GetNbinsX()+2; i++ ) {
        h3.SetBinContent( xbin, ybin, i, profile.GetBinContent(i) );
      }

    }

    fillEmptyBinsWithUnity( h3 );

  }

  TH3F GetH3() const { return h3; }

  private:
  TH3F h3;
  TChain chFast;
  TChain chFull;
};



