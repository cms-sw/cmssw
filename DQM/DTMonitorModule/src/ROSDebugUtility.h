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
int cont7=0,cont8=0,cont9=0,cont10=0,cont11=0,cont12=0;
int cycle = 1495;

inline void ROSWords_t(double& ResetCount_unfolded,int ROS_number,int ROSDebug_BcntResCnt,int nevents)
{

// synchronize with evt #
// if (neventsROS25 ==1) ResetCount_unfolded_comp.first=ROS_number;

// Processing first ROS
 if (ROS_number == 1){
  double ResetCount = ROSDebug_BcntResCnt*0.0000891;
 
  if (ResetCount_unfolded_comp <= (ResetCount)) {
  cont = cont;  
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
  cont2 = cont2;  
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
  cont3 = cont3;  
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
  cont4 = cont4;  
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
  cont5 = cont5;  
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
  cont6 = cont6;  
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
  cont7 = cont7;  
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
  cont8 = cont8;  
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
  cont9 = cont9;  
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
  cont10 = cont10;  
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
  cont11 = cont11;  
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
  cont12 = cont12;  
   ResetCount_unfolded = ResetCount12 + cycle*cont12;
   }
  else { cont12 = cont12 + 1;
   ResetCount_unfolded = ResetCount12 + cycle*cont12;
    }
  ResetCount_unfolded_comp12 = ResetCount12;  
   }  
}
