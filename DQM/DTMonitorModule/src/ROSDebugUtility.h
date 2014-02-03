/*
 * \file ROSDebugUtility.h
 * File to generate the plots used in the debug of ROS parameters
 * $Date: 2007/10/9 
 * \author Jorge Molina (CIEMAT)
 */

std::string itoa(int current);
double ResetCount_unfolded_comp =0;
double ResetCount_unfolded_comp2 =0;
double ResetCount_unfolded_comp3 =0;
double ResetCount_unfolded_comp4 =0;
double ResetCount_unfolded_comp5 =0;
double ResetCount_unfolded_comp6 =0;
double ResetCount_unfolded_comp7 =0;
double ResetCount_unfolded_comp8 =0;
double ResetCount_unfolded_comp9 =0;
double ResetCount_unfolded_comp10 =0;
double ResetCount_unfolded_comp11 =0;
double ResetCount_unfolded_comp12 =0;
double ResetCount_unfolded=0;
int cont=0,cont2=0,cont3=0,cont4=0,cont5=0,cont6=0;
int cont7=0,cont8=0,cont9=0,cont10=0,cont11=0,cont12=0,cycle = 1495;
float trigger_counter1=0,trigger_counter2=0,trigger_counter3=0,trigger_counter4=0,trigger_counter5=0,trigger_counter6=0;
float trigger_counter7=0,trigger_counter8=0,trigger_counter9=0,trigger_counter10=0,trigger_counter11=0,trigger_counter12=0;
int freq1,freq2,freq3,freq4,freq5,freq6,freq7,freq8,freq9,freq10,freq11,freq12;
long double first_evt=0;
long double first_evt2=0;
long double first_evt3=0;
long double first_evt4=0;
long double first_evt5=0;
long double first_evt6=0;
long double first_evt7=0;
long double first_evt8=0;
long double first_evt9=0;
long double first_evt10=0;
long double first_evt11=0;
long double first_evt12=0;
float peso=1;

inline void ROSWords_t(double& ResetCount_unfolded,int ROS_number,int ROSDebug_BcntResCnt,int nevents)
{

// synchronize with evt #
// if (neventsROS25 ==1) ResetCount_unfolded_comp.first=ROS_number;

// Processing first ROS
 if (ROS_number == 1){
  double ResetCount = ROSDebug_BcntResCnt*0.0000891;
 
  if (ResetCount_unfolded_comp <= (ResetCount)) {
   ResetCount_unfolded = ResetCount + cycle*cont;
   }
  else { cont = cont + 1;
   ResetCount_unfolded = ResetCount + cycle*cont;
   }  
  ResetCount_unfolded_comp = ResetCount; 
   }
    
// second ROS  
 else if (ROS_number == 2){
  double ResetCount2 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp2 <= (ResetCount2)) {
   ResetCount_unfolded = ResetCount2 + cycle*cont2;
   }
  else { cont2 = cont2 + 1;
   ResetCount_unfolded = ResetCount2 + cycle*cont2;
    }
  ResetCount_unfolded_comp2 = ResetCount2;    
   }

// third ROS  
 else if (ROS_number == 3){
  double ResetCount3 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp3 <= (ResetCount3)) {
   ResetCount_unfolded = ResetCount3 + cycle*cont3;
   }
  else { cont3 = cont3 + 1;
   ResetCount_unfolded = ResetCount3 + cycle*cont3;
    }
  ResetCount_unfolded_comp3 = ResetCount3;  
   }
  

// 4th ROS  
 else if (ROS_number == 4){
  double ResetCount4 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp4 <= (ResetCount4)) {
   ResetCount_unfolded = ResetCount4 + cycle*cont4;
   }
  else { cont4 = cont4 + 1;
   ResetCount_unfolded = ResetCount4 + cycle*cont4;
    }
  ResetCount_unfolded_comp = ResetCount4;  
    }

// 5th ROS  
 else if (ROS_number == 5){
  double ResetCount5 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp5 <= (ResetCount5)) {
   ResetCount_unfolded = ResetCount5 + cycle*cont5;
   }
  else { cont5 = cont5 + 1;
   ResetCount_unfolded = ResetCount5 + cycle*cont5;
    }
  ResetCount_unfolded_comp5 = ResetCount5;  
   }

// 6th ROS  
 else if (ROS_number == 6){
  double ResetCount6 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp6 <= (ResetCount6)) {
   ResetCount_unfolded = ResetCount6 + cycle*cont6;
   }
  else { cont6 = cont6 + 1;
   ResetCount_unfolded = ResetCount6 + cycle*cont6;
    }
  ResetCount_unfolded_comp6 = ResetCount6;  
   }
       
// 7th ROS  
 else if (ROS_number == 7){
  double ResetCount7 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp7 <= (ResetCount7)) {
   ResetCount_unfolded = ResetCount7 + cycle*cont7;
   }
  else { cont7 = cont7 + 1;
   ResetCount_unfolded = ResetCount7 + cycle*cont7;
    }
   ResetCount_unfolded_comp7 = ResetCount7;
   }

// 8th ROS  
 else if (ROS_number == 8){
  double ResetCount8 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp8 <= (ResetCount8)) {
   ResetCount_unfolded = ResetCount8 + cycle*cont8;
   }
  else { cont8 = cont8 + 1;
   ResetCount_unfolded = ResetCount8 + cycle*cont8;
    }
  ResetCount_unfolded_comp8 = ResetCount8;
   }

// 9th ROS  
 else if (ROS_number == 9){
  double ResetCount9 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp9 <= (ResetCount9)) {
   ResetCount_unfolded = ResetCount9 + cycle*cont9;
   }
  else { cont9 = cont9 + 1;
   ResetCount_unfolded = ResetCount9 + cycle*cont9;
    }
  ResetCount_unfolded_comp9 = ResetCount9;  
   }

// 10th ROS  
 else if (ROS_number == 10){
  double ResetCount10 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp10 <= (ResetCount10)) {
   ResetCount_unfolded = ResetCount10 + cycle*cont10;
   }
  else { cont10 = cont10 + 1;
   ResetCount_unfolded = ResetCount10 + cycle*cont10;
    }
  ResetCount_unfolded_comp10 = ResetCount10;  
   }

// 11th ROS  
 else if (ROS_number == 11){
  double ResetCount11 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp11 <= (ResetCount11)) {
   ResetCount_unfolded = ResetCount11 + cycle*cont11;
   }
  else { cont11 = cont11 + 1;
   ResetCount_unfolded = ResetCount11 + cycle*cont11;
    }
  ResetCount_unfolded_comp11 = ResetCount11;  
   }
  
  // 12th ROS  
 else if (ROS_number == 12){
  double ResetCount12 = ROSDebug_BcntResCnt*0.0000891;
  if (ResetCount_unfolded_comp12 <= (ResetCount12)) {
   ResetCount_unfolded = ResetCount12 + cycle*cont12;
   }
  else { cont12 = cont12 + 1;
   ResetCount_unfolded = ResetCount12 + cycle*cont12;
    }
  ResetCount_unfolded_comp12 = ResetCount12;  
   }  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



void ROS_L1A_Frequency(int ROS_number,int ROSDebug_BcntResCnt,
			int neventsROS25,double& frequency,float& trigger_counter)
{

 trigger_counter = 0.;
 
// first ROS
 if (ROS_number == 1)
 {  
  long double second_evt = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt - first_evt)<(1/(0.0000891*peso))){        
     trigger_counter1 += 1;
     trigger_counter = trigger_counter1;
     }
     else{ //window change
     frequency = trigger_counter1;
     first_evt = second_evt;
     trigger_counter1 = 0;} 
   }
  else{
    long double second_evt_neg = second_evt + max_bx;      

    if ((second_evt_neg - first_evt)<(1/(0.0000891*peso))){
     trigger_counter1 +=1;
     trigger_counter = trigger_counter1;
     frequency = first_evt*0.0000891;
     }
     else {
     first_evt = second_evt;
     trigger_counter1 = 0;}
  }
  return;
 }



// second ROS
 else if (ROS_number == 2){ 
  long double second_evt2 = ROSDebug_BcntResCnt;
		   
   if (neventsROS25==1) first_evt2 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt2 - first_evt2)<(1/(0.0000891*peso))){        
     trigger_counter2 += 1;
     trigger_counter = trigger_counter2;
     }
     else{ //window change
     frequency = trigger_counter2;
     first_evt2 = second_evt2;
     trigger_counter2 = 0;} 
   }
  else{
    long double second_evt_neg2 = second_evt2 + max_bx;   

    if ((second_evt_neg2 - first_evt2)<(1/(0.0000891*peso))){
     trigger_counter2 +=1;
     trigger_counter = trigger_counter2;
     frequency = first_evt2*0.0000891;
     }
     else {
     first_evt2 = second_evt2;
     trigger_counter2 = 0;}
  }
 }     
 
// third ROS
 else if (ROS_number == 3){
  
  long double second_evt3 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt3 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt3 - first_evt3)<(1/(0.0000891*peso))){        
     trigger_counter3 += 1;
     trigger_counter = trigger_counter3;
     }
     else{ //window change
     frequency = trigger_counter3;
     first_evt3 = second_evt3;
     trigger_counter3 = 0;} 
   }
  else{
    long double second_evt_neg3 = second_evt3 + max_bx; 

    if ((second_evt_neg3 - first_evt3)<(1/(0.0000891*peso))){
     trigger_counter3 +=1;
     trigger_counter = trigger_counter3;
     frequency = first_evt3*0.0000891;
     }
     else {
     first_evt3 = second_evt3;
     trigger_counter3 = 0;}
  }
 }

 
// 4th ROS
 else if (ROS_number == 4){
  
  long double second_evt4 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt4 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt4 - first_evt4)<(1/(0.0000891*peso))){        
     trigger_counter4 += 1;
     trigger_counter = trigger_counter4;
     }
     else{ //window change
     frequency = trigger_counter4;
     first_evt4 = second_evt4;
     trigger_counter4 = 0;} 
   }
  else{
    long double second_evt_neg4 = second_evt4 + max_bx;      

    if ((second_evt_neg4 - first_evt4)<(1/(0.0000891*peso))){
     trigger_counter4 +=1;
     trigger_counter = trigger_counter4;
     frequency = first_evt4*0.0000891;
     }
     else {
     first_evt4 = second_evt4;
     trigger_counter4 = 0;}
  }
 }
		   
// 5th ROS
 else if (ROS_number == 5){
  
  long double second_evt5 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt5 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt5 - first_evt5)<(1/(0.0000891*peso))){        
     trigger_counter5 += 1;
     trigger_counter = trigger_counter5;
     }
     else{ //window change
     frequency = trigger_counter5;
     first_evt5 = second_evt5;
     trigger_counter5 = 0;} 
   }
  else{
    long double second_evt_neg5 = second_evt5 + max_bx;      

    if ((second_evt_neg5 - first_evt5)<(1/(0.0000891*peso))){
     trigger_counter5 +=1;
     trigger_counter = trigger_counter5;
     frequency = first_evt5*0.0000891;
     }
     else {
     first_evt5 = second_evt5;
     trigger_counter5 = 0;}
  }
 }
		   
// 6th ROS
 else if (ROS_number == 6){
  
  long double second_evt6 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt6 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt6 - first_evt6)<(1/(0.0000891*peso))){        
     trigger_counter6 += 1;
     trigger_counter = trigger_counter6;
     }
     else{ //window change
     frequency = trigger_counter6;
     first_evt6 = second_evt6;
     trigger_counter6 = 0;} 
   }
  else{
    long double second_evt_neg6 = second_evt6 + max_bx;      

    if ((second_evt_neg6 - first_evt6)<(1/(0.0000891*peso))){
     trigger_counter6 +=1;
     trigger_counter = trigger_counter6;
     frequency = first_evt6*0.0000891;
     }
     else {
     first_evt6 = second_evt6;
     trigger_counter6 = 0;}
  }
 }
		   

// 7th ROS
 else if (ROS_number == 7){
  
  long double second_evt7 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt7 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt7 - first_evt7)<(1/(0.0000891*peso))){        
     trigger_counter7 += 1;
     trigger_counter = trigger_counter7;
     }
     else{ //window change
     frequency = trigger_counter7;
     first_evt7 = second_evt7;
     trigger_counter7 = 0;} 
   }
  else{
    long double second_evt_neg7 = second_evt7 + max_bx;      

    if ((second_evt_neg7 - first_evt7)<(1/(0.0000891*peso))){
     trigger_counter7 +=1;
     trigger_counter = trigger_counter7;
     frequency = first_evt7*0.0000891;
     }
     else {
     first_evt7 = second_evt7;
     trigger_counter7 = 0;}
  }
 }

// 8th ROS
 else if (ROS_number == 8){
  
  long double second_evt8 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt8 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt8 - first_evt8)<(1/(0.0000891*peso))){        
     trigger_counter8 += 1;
     trigger_counter = trigger_counter8;
     }
     else{ //window change
     frequency = trigger_counter8;
     first_evt8 = second_evt8;
     trigger_counter8 = 0;} 
   }
  else{
    long double second_evt_neg8 = second_evt8 + max_bx;      

    if ((second_evt_neg8 - first_evt8)<(1/(0.0000891*peso))){
     trigger_counter8 +=1;
     trigger_counter = trigger_counter8;
     frequency = first_evt8*0.0000891;
     }
     else {
     first_evt8 = second_evt8;
     trigger_counter8 = 0;}
  }
 }

// 9th ROS
 else if (ROS_number == 9){
  
  long double second_evt9 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt9 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt9 - first_evt9)<(1/(0.0000891*peso))){        
     trigger_counter9 += 1;
     trigger_counter = trigger_counter9;
     }
     else{ //window change
     frequency = trigger_counter9;
     first_evt9 = second_evt9;
     trigger_counter9 = 0;} 
   }
  else{
    long double second_evt_neg9 = second_evt9 + max_bx;      

    if ((second_evt_neg9 - first_evt9)<(1/(0.0000891*peso))){
     trigger_counter9 +=1;
     trigger_counter = trigger_counter9;
     frequency = first_evt9*0.0000891;
     }
     else {
     first_evt9 = second_evt9;
     trigger_counter9 = 0;}
  }
 }


// 10th ROS
 else if (ROS_number == 10){
  
  long double second_evt10 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt10 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt10 - first_evt10)<(1/(0.0000891*peso))){        
     trigger_counter10 += 1;
     trigger_counter = trigger_counter10;
     }
     else{ //window change
     frequency = trigger_counter10;
     first_evt10 = second_evt10;
     trigger_counter10 = 0;} 
   }
  else{
    long double second_evt_neg10 = second_evt10 + max_bx;      

    if ((second_evt_neg10 - first_evt10)<(1/(0.0000891*peso))){
     trigger_counter10 +=1;
     trigger_counter = trigger_counter10;
     frequency = first_evt10*0.0000891;
     }
     else {
     first_evt10 = second_evt10;
     trigger_counter10 = 0;}
  }
 }

 
// 11th ROS
 else if (ROS_number == 11){
  
  long double second_evt11 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt11 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt11 - first_evt11)<(1/(0.0000891*peso))){        
     trigger_counter11 += 1;
     trigger_counter = trigger_counter11;
     }
     else{ //window change
     frequency = trigger_counter11;
     first_evt11 = second_evt11;
     trigger_counter11 = 0;} 
   }
  else{
    long double second_evt_neg11 = second_evt11 + max_bx;      

    if ((second_evt_neg11 - first_evt11)<(1/(0.0000891*peso))){
     trigger_counter11 +=1;
     trigger_counter = trigger_counter11;
     frequency = first_evt11*0.0000891;
     }
     else {
     first_evt11 = second_evt11;
     trigger_counter11 = 0;}
  }
 }

 
// 12th ROS
 else if (ROS_number == 12){
  
  long double second_evt12 = ROSDebug_BcntResCnt;
   if (neventsROS25==1) first_evt12 = ROSDebug_BcntResCnt;
		   
  if (ROSDebug_BcntResCnt<16777215) {

    if ((second_evt12 - first_evt12)<(1/(0.0000891*peso))){        
     trigger_counter12 += 1;
     trigger_counter = trigger_counter12;
     }
     else{ //window change
     frequency = trigger_counter12;
     first_evt12 = second_evt12;
     trigger_counter12 = 0;} 
   }
  else{
    long double second_evt_neg12 = second_evt12 + max_bx;      

    if ((second_evt_neg12 - first_evt12)<(1/(0.0000891*peso))){
     trigger_counter12 +=1;
     trigger_counter = trigger_counter12;
     frequency = first_evt12*0.0000891;
     }
     else {
     first_evt12 = second_evt12;
     trigger_counter12 = 0;}
  }
 }
}
