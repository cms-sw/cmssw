#include <stdio.h>
#include <math.h>


class Conv{
  
 public:
  
  Conv(){}
  
  //square wave fcn convoluted with 1/(t+2.5)
  float elec(float t,float vs){
    float f;
    if (t<=vs){
      f=log(t+2.5)-log(2.5);
    }
    else{
      f=log(t+2.5)-log(t-50+2.5);
    }
    return f;
  }//end elec
  
  
  //calculate single electron distribution in 6.25 ns steps
  void mkbins(float vs){
    int i,k;
    float t;
    for(i=0;i<120;i++) conve[i]=0.0;
    for(i=0;i<120;i++){
      for(k=0;k<16;k++){
	t=(6.25*i)+(k*0.625);
	conve[i]=conve[i]+elec(t,vs);
      }
    }
  } //end mkbins
  

  //convolution function
  void convolution(float *xleft_a, float *xleft_b, float *min_left, float *xright_a, float *xright_b, float *min_right, float *pTime){
    //void(convolution){  
    
    float max, cross0,cross2,min_l,min_r,sum_x=0.0,sumx2=0.;
    float sum_y_left=0.0,sum_y_right=0.0,sum_xy_left=0.0,sum_xy_right=0.0;
    float a_left=0.0,a_right=0.0,b_left=0.0,b_right=0.0,chi2_left=0.0,chi2_right=0.0,chi_left=0.0,chi_right=0.0;
    float aleft=0.0,aright=0.0,bleft=0.0,bright=0.0;
    int i,j,k,l,imax=0;
    
    for(l=0;l<3;l++){
      for(i=0;i<119;i++)conv[l][i]=0.0;
      for(j=0;j<119;j++){
	for(k=0;k<119;k++){
	  if(j+k<119)conv[l][j+k]=conv[l][j+k]+convd[l][j]*conve[k];
	}
      }
    }
    max=0;
    min_l=9999999999999999.0;
    min_r=9999999999999999.0;
    for(i=0;i<119;i++){
      if(conv[1][i]>max){ 
	max=conv[1][i];
	imax=i;
      }
    }
    
    //find the max peak time from 3 timebins when line intersects x axis a+b*x=0 -> x=-a/b 
    float time1=-999.0, time2=-999.0, time3=-999.0;
    float data1=-999.0, data2=-999.0, data3=-999.0;
    float peakTime=0.0;
    
    time1=imax-1;
    time2=imax;
    time3=imax+1;
    
    data1=conv[1][imax-1];
    data2=conv[1][imax];
    data3=conv[1][imax+1];
    
    peakTime=(0.5)*((time1*time1*(data3-data2)+time2*time2*(data1-data3)+time3*time3*(data2-data1))/(time1*(data3-data2)+time2*(data1-data3)+time3*(data2-data1)))*6.25;
    
    for(l=0;l<3;l++){
      for(i=0;i<119;i++)conv[l][i]=conv[l][i]/max;
    }
    
    int nobs = 0;
    for (int j=0; j<119; j++){
      if (conv[1][j]>0.6) nobs++;
    }
    
    for(i=0;i<119;i++){
      cross0=0.0;
      cross2=0.0;
      
      if(conv[1][i]>0.6){
	cross0=conv[0][i]/(conv[0][i]+conv[1][i]+conv[2][i]);
	cross2=conv[2][i]/(conv[0][i]+conv[1][i]+conv[2][i]);
	
	sum_x += i;
	sum_y_left += cross0;
	sum_y_right += cross2;
	sumx2 += i*i;
	sum_xy_left += i*cross0;
	sum_xy_right += i*cross2;
      }
    }  
    
    //LMS fitting straight line y=a+b*x
    
    bleft  = ((nobs*sum_xy_left) - (sum_x * sum_y_left))/((nobs*sumx2) - (sum_x*sum_x));
    bright = ((nobs*sum_xy_right) - (sum_x * sum_y_right))/((nobs*sumx2) - (sum_x*sum_x));
    
    aleft  = ((sum_y_left*sumx2)-(sum_x*sum_xy_left))/((nobs*sumx2)-(sum_x*sum_x));
    aright = ((sum_y_right*sumx2)-(sum_x*sum_xy_right))/((nobs*sumx2)-(sum_x*sum_x));
    
    for(i=0;i<119;i++ ){
      chi2_left  += (cross0 -(aleft+(bleft*i)))*(cross0 -(aleft+(bleft*i)));
      chi2_right += (cross2 -(aright+(bright*i)))*(cross2 -(aright+(bright*i)));
    }	
    
    if(chi_left<min_l){
      min_l=chi_left;
      bleft=bleft;
      aleft=aleft;
    }
    if(chi_right<min_r){
      min_r=chi_right;
      bright=bright;
      aright=aright;
    }
    
    
    //Now calculating parameters in ns to compensate for drift in peak time  
    b_left  = bleft/6.25;
    b_right = bright/6.25;
    
    a_left  = aleft  + (bleft*peakTime/6.25);
    a_right = aright + (bright*peakTime/6.25);
    
    *xleft_a   = a_left; 
    *xleft_b   = b_left;
    *min_left  = chi2_left;
    *xright_a  = a_right;
    *xright_b  = b_right;
    *min_right = chi2_right;
    *pTime     = peakTime;
    
  } //end CONVOLUTION  
  
  ~Conv(){}


  float convd[3][120];
  float nconvd[3][120];
  float conve[120];
  float conv[3][120];
 
private:
  
  

} ;

