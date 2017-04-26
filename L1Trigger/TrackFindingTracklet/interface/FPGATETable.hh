#ifndef FPGATETABLE_H
#define FPGATETABLE_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class FPGATETable{

public:

  FPGATETable() {
   
  }

  ~FPGATETable() {

  }


  void init(int stubpt1bits,
	    int stubpt2bits,
	    int deltaphibits,
	    int deltarbits,
	    int z1bits,
	    int z2bits,
	    int r1bits,
	    int r2bits,
	    double deltaphi,  //width of vm in phi
	    double deltaphioffset,  //offset of l2 vm with respect l1
            double zmin1,
	    double zmax1,
            double zmin2,
	    double zmax2,
            double rmin1,
	    double rmax1,
            double rmin2,
	    double rmax2,
	    int i1,
	    int i2,
	    int j1,
	    int j2
	    ) {

    //cout << "In FPGATETable::init()"<<endl;

    z1_=zmin1;
    z2_=zmin2;
    
    i1_=i1;
    i2_=i2;
    j1_=j1;
    j2_=j2;


    phitablebits_=stubpt1bits+stubpt2bits+deltaphibits+deltarbits;
    ztablebits_=z1bits+z2bits+r1bits+r2bits;

    //int firstprint=true;

    phitableentries_=1<<phitablebits_;
    ztableentries_=1<<ztablebits_;

    //cout << "init 2 phitablebits_ : "<<phitablebits_<<endl;
    //cout << "init 2 phitableentries_ : "<<phitableentries_<<endl;


    for (int i=0;i<phitableentries_;i++){
      int istubpt1=i>>(stubpt2bits+deltaphibits+deltarbits);
      int istubpt2=(i>>(deltaphibits+deltarbits))&((1<<stubpt2bits)-1);
      int ideltaphi=(i>>(deltarbits))&((1<<deltaphibits)-1);
      int ideltar=i&((1<<deltarbits)-1);
      //now ideltaphi and idelta r are actually signed integers. Convert
      int irshift = 8*sizeof(int)-deltarbits;
      int iphishift = 8*sizeof(int)-deltaphibits;
      ideltaphi = (ideltaphi<<iphishift)>>iphishift;
      ideltar   = (ideltar<<irshift)>>irshift;

      assert(istubpt1>=0&&istubpt1<8);
      assert(istubpt2>=0&&istubpt2<8);


      /*
      bool print=false;
      if (ideltaphi==11&&ideltar==0&&rmin1<30.0&&firstprint) {
	print=true;
	firstprint=false;
      }
      */

      double deltaphiavg=deltaphioffset+2*(deltaphi*ideltaphi)/(1<<deltaphibits);
      double deltar=rmin2-rmin1+2.0*(ideltar)*(rmax1-rmin1)/(1<<deltarbits);
      double Delta=sqrt(deltar*deltar+2*rmax1*rmax2*(1-cos(deltaphiavg)));
      double rinv=-2*sin(deltaphiavg)/Delta; //HACK this sign should be fixed elsewhere!!!

      double ptstubinv1 = -99;
      double ptstubinv2 = -99;

      int stubL1 = 0;                                                         
      if(fabs(rmin1 - rmeanL1) <= 3 ) stubL1 = 1;                             
      if(fabs(rmin1 - rmeanL2) <= 3 ) stubL1 = 2;                             
      if(fabs(rmin1 - rmeanL3) <= 3 ) stubL1 = 3;                             
      if(fabs(rmin1 - rmeanL4) <= 3 ) stubL1 = 4;                             
      if(fabs(rmin1 - rmeanL5) <= 3 ) stubL1 = 5;                             
      if(fabs(rmin1 - rmeanL6) <= 3 ) stubL1 = 6;                             
                                                                              
      int stubL2 = 0;                                                         
      if(fabs(rmin2 - rmeanL1) <= 3 ) stubL2 = 1;                             
      if(fabs(rmin2 - rmeanL2) <= 3 ) stubL2 = 2;                             
      if(fabs(rmin2 - rmeanL3) <= 3 ) stubL2 = 3;                             
      if(fabs(rmin2 - rmeanL4) <= 3 ) stubL2 = 4;                             
      if(fabs(rmin2 - rmeanL5) <= 3 ) stubL2 = 5;                             
      if(fabs(rmin2 - rmeanL6) <= 3 ) stubL2 = 6;      


      // Standard encoding 
      if(!enstubbend){
      if (istubpt1==0) ptstubinv1=0.40;
      if (istubpt1==1) ptstubinv1=0.25;
      if (istubpt1==2) ptstubinv1=0.15;
      if (istubpt1==3) ptstubinv1=0.05;
      if (istubpt1==4) ptstubinv1=-0.05;
      if (istubpt1==5) ptstubinv1=-0.15;
      if (istubpt1==6) ptstubinv1=-0.25;
      if (istubpt1==7) ptstubinv1=-0.40;

      if (istubpt2==0) ptstubinv2=0.40;
      if (istubpt2==1) ptstubinv2=0.25;
      if (istubpt2==2) ptstubinv2=0.15;
      if (istubpt2==3) ptstubinv2=0.05;
      if (istubpt2==4) ptstubinv2=-0.05;
      if (istubpt2==5) ptstubinv2=-0.15;
      if (istubpt2==6) ptstubinv2=-0.25;
      if (istubpt2==7) ptstubinv2=-0.40;
      }

      float KL1 =1./6.;                                                       
      float KL2 =1./6.;                                                       
      float KL3 =1./6.;                                                       
      float KL4 =1./6.;                                                       
      float KL5 =1./6.;                                                       
      float KL6 =1./6.;                                                       

      if(enstubbend){
      if(stubL1 == 1 ) ptstubinv1 = (istubpt1)*KL1 -0.66666;                  
      if(stubL1 == 2 ) ptstubinv1 = (istubpt1)*KL2 -0.66666;                  
      if(stubL1 == 3 ) ptstubinv1 = istubpt1*KL3 -0.666666;                   
      if(stubL1 == 4 ) ptstubinv1 = istubpt1*KL4 -0.666666;                   
      if(stubL1 == 5 ) ptstubinv1 = istubpt1*KL5 -0.666666;                   
      if(stubL1 == 6 ) ptstubinv1 = istubpt1*KL6 -0.666666;                   
                                                                              
      if(stubL2 == 1 ) ptstubinv2 = istubpt2*KL1 -0.66666;                    
      if(stubL2 == 2 ) ptstubinv2 = istubpt2*KL2 -0.66666;                    
      if(stubL2 == 3 ) ptstubinv2 = istubpt2*KL3 -0.66666;                    
      if(stubL2 == 4 ) ptstubinv2 = istubpt2*KL4 -0.66666;                    
      if(stubL2 == 5 ) ptstubinv2 = istubpt2*KL5 -0.66666;                    
      if(stubL2 == 6 ) ptstubinv2 = istubpt2*KL6 -0.66666;               
      }




      double pttracklet=0.3*3.8/(rinv*100); 

      //if (i==8187) {	
      //cout << "pttracklet: "<<i<<" "<<pttracklet<<endl;
      //}

      bool pass1=fabs(ptstubinv1-1.0/pttracklet)<teptconsistency;
      bool pass2=fabs(ptstubinv2-1.0/pttracklet)<teptconsistency;
      bool pass=pass1&&pass2;      


      //if (i==8187) {
      //	cout << "pttracklet: "<<i<<" "<<pttracklet<<" "<<pass<<endl;
      //      }


      double deltaphimin=2*(deltaphi*ideltaphi)/(1<<deltaphibits);
      deltaphimin+=deltaphioffset;
      deltaphimin=fabs(deltaphimin)-deltaphi/(1<<deltaphibits);
      if (deltaphimin<0.0) deltaphimin=0.0;

      //double deltarmax=rmin2-rmin1+0.5*((rmax1-rmin1)+(rmax2-rmin2))*
      //fabs(ideltar)/(1<<deltarbits);
      double deltarmax=rmin2-rmin1+2.0*(ideltar)*(rmax1-rmin1)/(1<<deltarbits);
     
      double Deltamax=sqrt(deltarmax*deltarmax+
			   2*rmax1*rmax2*(1-cos(deltaphimin)));
      double rinvmin=2*sin(deltaphimin)/Deltamax;

      tablephi_.push_back((fabs(rinvmin)<0.0057)&&pass);


      /*
      if (print) cout << "deltaphimin deltaphi deltaphioffset ideltaphi rinvmin:"
		      << deltaphimin <<" "
		      << deltaphi<<" "
		      <<deltaphioffset <<" "
		      <<ideltaphi <<" "
		      <<rinvmin<<endl;

      if (i<16) cout << "i ideltaphi ideltar : "
         << i <<" "<<ideltaphi
         <<" "<<ideltar<<" "<<tablephi_[i]<<endl;
      */

    }

    //cout << "init 3 phitablebits_ : "<<phitablebits_<<endl;


    for (int i=0;i<ztableentries_;i++){
      int iz1=i>>(z2bits+r1bits+r2bits);
      int iz2=(i>>(r1bits+r2bits))&((1<<z2bits)-1);
      int ir1=(i>>(r2bits))&((1<<r1bits)-1);
      int ir2=i&((1<<r2bits)-1);
      //now ir1 and ir2 are actually signed integers. Convert
      int ir1shift = 8*sizeof(int)-r1bits;
      int ir2shift = 8*sizeof(int)-r2bits;
      ir1 = (ir1<<ir1shift)>>ir1shift;
      ir2 = (ir2<<ir2shift)>>ir2shift;

      //bool printz=(j1_==3)&(j2_==3)&(i==0);

      double z1[2];
      double z2[2];
      double r1[2];
      double r2[2];

      z1[0]=zmin1+iz1*(zmax1-zmin1)/(1<<z1bits);
      z1[1]=zmin1+(iz1+1)*(zmax1-zmin1)/(1<<z1bits);

      z2[0]=zmin2+iz2*(zmax2-zmin2)/(1<<z2bits);
      z2[1]=zmin2+(iz2+1)*(zmax2-zmin2)/(1<<z2bits);

      double rave1 = 0.5*(rmin1+rmax1);
      double rave2 = 0.5*(rmin2+rmax2);

      r1[0]=rave1+ir1*(rmax1-rmin1)/(1<<r1bits);
      r1[1]=rave1+(ir1+1)*(rmax1-rmin1)/(1<<r1bits);

      r2[0]=rave2+ir2*(rmax2-rmin2)/(1<<r2bits);
      r2[1]=rave2+(ir2+1)*(rmax2-rmin2)/(1<<r2bits);

      bool below=false;
      bool center=false;
      bool above=false;


      for(int iiz1=0;iiz1<2;iiz1++){
	for(int iiz2=0;iiz2<2;iiz2++){
	  for(int iir1=0;iir1<2;iir1++){
	    for(int iir2=0;iir2<2;iir2++){
	      double z0=z1[iiz1]+(z1[iiz1]-z2[iiz2])*r1[iir1]/(r2[iir2]-r1[iir1]);

	      //if (i==3480&&r1[0]<40.0) {
	      //cout << "ZTABLE "<<z0<<endl;
	      //}

	      /*
	      if (printz) {
		cout << "z1 z2 r1 r2 z0 "
		     <<z1[iz1]<<" "
		     <<z2[iz2]<<" "
		     <<r1[ir1]<<" "
		     <<r2[ir2]<<" "
		     <<z0<<endl;
	      }
	      */
	      if (fabs(z0)<15.0) center=true;
	      if (z0<-15.0) below=true;
	      if (z0>15.0) above=true;
	    }
	  }
	}
      }


      /*
      if (i==3480&&r1[0]<40.0) {
	cout << "ZTABLE : z1 "<<z1[0]<<" "<<z1[1]
	     << " z2 "<<z2[0]<<" "<<z2[1]<<" "<<(center|(below&above))<<endl;
	cout << "ZTABLE : r1 "<<r1[0]<<" "<<r1[1]
	     << " r2 "<<r2[0]<<" "<<r2[1]<<" "<<(center|(below&above))<<endl;
	cout << "ZTABLE : rmin1 rmax1 r1bits "<<rmin1<<" "<<rmax1
	     << " "<<r1bits<<endl;
	cout << "ZTABLE : iz1 iz2 ir1 ir2 "<<iz1<<" "<<iz2
	     << " "<<ir1<<" "<<ir2<<endl;
      }
      */

      tablez_.push_back(center|(below&above));
      
      //cout << "i iz1 iz2 ir1 ir2 : "
      //   << i<<" "<<iz1<<" "<<iz2<<" "<<ir1<<" "<<ir2<<" "<<tablez_[i]<<endl;
    }

    //cout << "init 4 phitablebits_ : "<<phitablebits_<<endl;
    //cout << "init 2 phitableentries_ : "<<phitableentries_<<endl;


  }
	    

  void writephi(std::string fname) {

    ofstream out(fname.c_str());

    //cout << "writephi 2 phitableentries_ : "<<phitableentries_<<endl;

    for (int i=0;i<phitableentries_;i++){
      FPGAWord entry;
      //cout << "phitablebits_ : "<<phitablebits_<<endl;
      entry.set(i,phitablebits_);
      //out << entry.str()<<" "<<tablephi_[i]<<endl;
      out <<tablephi_[i]<<endl;
    }
    out.close();
  
  }


  void writez(std::string fname) {

    ofstream out(fname.c_str());

    for (int i=0;i<ztableentries_;i++){
      FPGAWord entry;
      //cout << "ztablebits_ : "<<ztablebits_<<endl;
      entry.set(i,ztablebits_);
      out << entry.str()<<" "<<tablez_[i]<<endl;
    }
    out.close();
      
  }

  bool phicheck(int address) const {
    //assert(i1==i1_);
    //assert(i2==i2_);
    //assert(j1==j1_);
    //assert(j2==j2_);
    assert(address>=0);
    assert(address<phitableentries_);
    return tablephi_[address];
  }

  bool zcheck(int address) const {
    assert(address>=0);
    assert(address<ztableentries_);
    return tablez_[address];
  }

  double z1() {
    return z1_;
  }

  double z2() {
    return z2_;
  }

private:

  vector<bool> tablephi_;
  vector<bool> tablez_;

  int phitableentries_;
  int ztableentries_;
  int phitablebits_;
  int ztablebits_;
  
  //int stubpt1bits_;
  //int stubpt2bits_;
  //int deltaphibits_;
  //int deltarbits_;
  //int z1bits_;
  //int z2bits_;
  //int r1bits_;
  //int r2bits_;
  
  int i1_;
  int i2_;
  int j1_;
  int j2_;

  double z1_;
  double z2_;
  
};



#endif



