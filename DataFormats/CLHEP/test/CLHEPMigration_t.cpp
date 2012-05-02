#include "DataFormats/CLHEP/interface//AlgebraicObjects.h"
#include<cassert>
#include<iostream>

namespace {
  namespace cmst {
    void tm() {
      
      CLHEP::HepMatrix m(3,4);
      double k=0.;
      for (int i=0;i!=3;++i)
	for (int j=0;j!=4;++j) {
	  k+=1;
	  m(i+1,j+1)=k;
	}

      ROOT::Math::SMatrix<double,3,4, typename ROOT::Math::MatRepStd<double,3,4> > sm = asSMatrix<3,4>(m);
      for (int i=0;i!=3;++i)
	for (int j=0;j!=4;++j) {
	  assert(m(i+1,j+1)==sm(i,j));
	}
      CLHEP::HepMatrix hm = asHepMatrix(sm);
      for (int i=0;i!=3;++i)
	for (int j=0;j!=4;++j)
	  assert(hm(i+1,j+1)==sm(i,j));
      
    }
    
    void ts() {
      
      CLHEP::HepSymMatrix m(4);
      double k=0.;
      for (int i=0;i!=4;++i)
	for (int j=i;j!=4;++j) {
	k+=1;
	m(i+1,j+1)=k;
	}

      ROOT::Math::SMatrix<double,4,4, typename ROOT::Math::MatRepSym<double,4> > sm = asSMatrix<4>(m);
      for (int i=0;i!=4;++i)
	for (int j=0;j!=4;++j)
	  assert(m(i+1,j+1)==sm(i,j));
      
      CLHEP::HepSymMatrix hm = asHepMatrix(sm);
      for (int i=0;i!=4;++i)
	for (int j=0;j!=4;++j)
	  assert(hm(i+1,j+1)==sm(i,j));
    }
  }
}


int main() {
  cmst::tm();
  cmst::ts();
}
