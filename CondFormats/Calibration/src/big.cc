#include "CondFormats/Calibration/interface/big.h"
#include <iostream>
//fill big
void 
big::fill( size_t tVectorSize,size_t thVectorSize,size_t sVectorSize,
	   const std::string& atitle){
  for(size_t i=0;i<tVectorSize;++i){
    big::bigEntry b;
    b.fill(i,1.0);
    tVector_.push_back(b);
  }
  for(size_t i=0;i<thVectorSize;++i){
    big::bigHeader h;
    h.fill(atitle);
    thVector_.push_back(h);
  }
  for(size_t i=0;i<sVectorSize;++i){
    big::bigStore s;
    s.fill(atitle);
    sVector_.push_back(s);
  }
}

//fill bigEntry
void
big::bigEntry::fill(int r,float seed){
  runnum=r; alpha=seed; cotalpha=seed; beta=seed; cotbeta=seed;
  costrk[0]=seed*0.1;costrk[1]=seed*0.2;costrk[2]=seed*0.3;
  qavg=seed ; symax=seed ; dyone=seed; syone=seed;sxmax=seed;
  dxone=seed;sxone=seed;dytwo=seed;sytwo=seed;dxtwo=seed;sxtwo=seed;
  qmin=seed;
  for (int i=0; i<parIDX::LEN1; ++i){
    for (int j=0; j<parIDX::LEN2; ++j){
      for (int k=0; k<parIDX::LEN3; ++k){
	par[parIDX::indexOf(i,j,k)]=seed;
      }
    }
  }
  for (int i=0; i<ytempIDX::LEN1; ++i){
    for (int j=0; j<ytempIDX::LEN2; ++j){
      ytemp[ytempIDX::indexOf(i,j)]=seed;
    }
  }
  for (int i=0; i<xtempIDX::LEN1; ++i){
    for (int j=0; j<xtempIDX::LEN2; ++j){
      xtemp[xtempIDX::indexOf(i,j)]=seed;
    }
  }
  for (int i=0; i<avgIDX::LEN1; ++i){
    for (int j=0; j<avgIDX::LEN2; ++j){
      for (int k=0; k<avgIDX::LEN3; ++k){
	avg[avgIDX::indexOf(i,j,k)]=seed;
      }
    }
  }
  for (int i=0; i<aqflIDX::LEN1; ++i){
    for (int j=0; j<aqflIDX::LEN2; ++j){
      for (int k=0; k<aqflIDX::LEN3; ++k){
	aqfl[aqflIDX::indexOf(i,j,k)]=seed;
      }
    }
  }
  for (int i=0; i<chi2IDX::LEN1; ++i){
    for (int j=0; j<chi2IDX::LEN2; ++j){
      for (int k=0; k<chi2IDX::LEN3; ++k){
	chi2[chi2IDX::indexOf(i,j,k)]=seed;
      }
    }
  }
  for (int i=0; i<spareIDX::LEN1; ++i){
    for (int j=0; j<spareIDX::LEN2; ++j){
	spare[spareIDX::indexOf(i,j)]=seed;
    }
  }
}

//fill bigHeader
void
big::bigHeader::fill( const std::string& atitle){
  title=std::string("atitle");
  ID=0; 
  NBy=1; 
  NByx=2; 
  NBxx=3; 
  NFy=4; 
  NFyx=5;
  NFxx=6;
  vbias=0.1; 
  temperature=0.2;
  fluence=0.3;
  qscale=0.4;
  s50=0.5;
  templ_version=1;
}
//fill bigStore
void
big::bigStore::fill( const std::string& atitle ){
  head.fill(atitle);
  for (int i=0; i<entbyIDX::LEN1; ++i){
    bigEntry b;
    b.fill(i,0.5*i);
    entby[entbyIDX::indexOf(i)]=b;//or use push_back as prefer
  }
  std::cout<<"length of entbx 1 "<<entbxIDX::LEN1<<std::endl;
  std::cout<<"length of entbx 2 "<<entbxIDX::LEN2<<std::endl;
  std::cout<<"total size of entbx "<<entbxIDX::SIZE<<std::endl;
  for (int i=0; i<entbxIDX::LEN1; ++i){
    for (int j=0; j<entbxIDX::LEN2; ++j){
      bigEntry c;
      c.fill(i*j,0.3*j);
      entbx[entbxIDX::indexOf(i,j)]=c;//or use push_back as prefer
    }
  }
  for (int i=0; i<entfyIDX::LEN1; ++i){
    bigEntry f;
    f.fill(i,0.4*i);
    entfy[entfyIDX::indexOf(i)]=f;//or use push_back as prefer
  }
  for (int i=0; i<entfxIDX::LEN1; ++i){
    for (int j=0; j<entfxIDX::LEN2; ++j){
      bigEntry f;
      f.fill(i*j,0.25*j);
      entfx[entfxIDX::indexOf(i,j)]=f;//or use push_back as prefer
    }
  }
}
