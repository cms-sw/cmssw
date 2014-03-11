#include "DataFormats/Math/interface/ProjectMatrix.h"
#include "DataFormats/Math/interface/MatRepSparse.h"



int main() {
 
  typedef double T;
  typedef  ROOT::Math::SMatrix<T,5,5,ROOT::Math::MatRepSym<T,5> > SMat55;
  typedef  ROOT::Math::SMatrix<T,2,2,ROOT::Math::MatRepSym<T,2> > SMatDD;
  typedef  ROOT::Math::SMatrix<T,5,5 > SMatNN;
  typedef  ROOT::Math::SMatrix<T,5,2 > SMatND;
  typedef  ROOT::Math::SMatrix<T,2,5 > SMatDN;

  typedef  ROOT::Math::SMatrix<T,5,5,MatRepSparse<T,5,5,1> > SMat00;
  typedef  ROOT::Math::SMatrix<T,5,5,MatRepSparse<T,5,5,3> > SMat34;

  std::cout << "size func " << sizeof(std::function<int(int)>) << std::endl;
  std::cout << "sizes " << sizeof(SMat55) << " " << sizeof(SMat00) << " " << sizeof(SMat34) << std::endl;
  
  //  SMat00 m00; m00.fRep = MatRepSparse<T,5,5,1>([](int i){ return i==0 ? 0 : -1;});
  SMat00 m00; m00.fRep.f = [](int i){ return i==0 ? 0 : -1;};
  m00(0,0)=3.2;
  SMat34 m34; m34.fRep.f = [](int i){ return i==(5*3+3) ? 0 : ( i==(5*4+4) ? 1 : (  (i==(5*3+4) || i==(5*4+3) ) ? 2 : -1 ) ) ;  };
  m34(3,3) = 1.2;
  m34(4,4) = 0.2;
  m34(3,4) = -0.4;

  SMat55 sum;
  std::cout << m00 << std::endl;
  std::cout << m34 << std::endl;
  ROOT::Math::AssignSym::Evaluate(sum,m00+m34);
  std::cout << sum << std::endl;

  double v[3] = {1., -0.5, 2.};
  SMatDD S(v,3);

  std::cout << S << std::endl;
  

  {
    SMatDN H; H(0,3)=1; H(1,4)=1;
    SMatND K = ROOT::Math::Transpose(H) * S;  
  
    SMatNN V = K*H; 
    
    std::cout << K << std::endl;
    std::cout << V << std::endl;

  }
  {

    ProjectMatrix<double,5,2> H; H.index[0]=3; H.index[1]=4;
    SMatND K = H.project(S);  
  
    SMatNN V = H.project(K); 
    
    std::cout << K << std::endl;
    std::cout << V << std::endl;

  }




  return 0;
  
}

