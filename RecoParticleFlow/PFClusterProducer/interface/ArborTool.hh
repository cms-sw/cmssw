#ifndef ARBORTOOL_H_
#define ARBORTOOL_H_

#include "TVector3.h"
#include "TMatrixF.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

namespace arbor{

  class QuickUnion{
  std::vector<unsigned> _id;
  std::vector<unsigned> _size;
  int _count;

  public:
    QuickUnion(const unsigned NBranches) {
      _count = NBranches;
      _id.resize(NBranches);
      _size.resize(NBranches);
      for( unsigned i = 0; i < NBranches; ++i ) {
	_id[i] = i;
	_size[i] = 1;
      }
    }
    
    int count() const { return _count; }
    
    unsigned find(unsigned p) {
      while( p != _id[p] ) {
	_id[p] = _id[_id[p]];
	p = _id[p];
      }
      return p;
    }
    
    bool connected(unsigned p, unsigned q) { return find(p) == find(q); }
    
    void unite(unsigned p, unsigned q) {
      unsigned rootP = find(p);
      unsigned rootQ = find(q);
      _id[p] = q;
      
      if(_size[rootP] < _size[rootQ] ) { 
	_id[rootP] = rootQ; _size[rootQ] += _size[rootP]; 
      } else { 
	_id[rootQ] = rootP; _size[rootP] += _size[rootQ]; 
      }
      --_count;
    }
  };

  typedef std::vector< std::vector<int> > branchcoll;
  typedef std::vector<int> branch;
  typedef std::vector< std::pair<int, int> > linkcoll;
  
  int BarrelFlag( TVector3 inputPos );
  
  int DepthFlag( TVector3 inputPos );
  
  TVector3 CalVertex( TVector3 Pos1, TVector3 Dir1, TVector3 Pos2, TVector3 Dir2 );
  
  int TPCPosition( TVector3 inputPos );		//Used to tag MCParticle position, if generated inside TPC & Dead outside
  
  float DisSeedSurface( TVector3 SeedPos );	//for a given position, calculate the distance to Calo surface ( ECAL )
  
  float DisTPCBoundary( TVector3 Pos );
  
  std::vector<bool>
  MatrixSummarize(const std::unordered_multimap<unsigned,unsigned>&, 
		  const unsigned  NBranches);

  TMatrixF MatrixSummarize( TMatrixF inputMatrix );
  
  std::vector<unsigned> SortMeasure( const std::vector<float>& Measure, int ControlOrderFlag );
  
  float DistanceChargedParticleToCluster(TVector3 CPRefPos, TVector3 CPRefMom, TVector3 CluPosition);
  
  branchcoll ArborBranchMerge(branchcoll inputbranches, TMatrixF inputMatrix);
}

#endif //
