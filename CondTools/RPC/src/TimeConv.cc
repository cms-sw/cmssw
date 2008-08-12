 /* 
 *  See header file for a description of this class.
 *
 *  $Date: 2008/02/1 15:35:49 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "TimeConv.h"
#include <iostream>
#include <math.h>


int TtoUT(coral::TimeStamp time)
{
  //  seconds since 1/1/1970
  int yea = time.year();
  int mon = time.month();
  int day = time.day();
  int hou = time.hour();
  int min = time.minute();
  int sec = time.second();
  int yes = (yea-1970)*31536000;
  int cony = (yea-1972)%4;
  if (cony == 0) yes = yes + (yea-1972)/4*86400; 
  else yes = yes + ((yea-1972)/4 - 1)*86400;
  int conm = (mon - 1);
  int mos = 0;
  switch (conm) {
  case 1 : 
    mos = 31*86400;
    break;
  case 2 : 
    mos = 59*86400;
    break;
  case 3 : 
    mos = 90*86400;
    break;
  case 4 : 
    mos = 120*86400;
    break;
  case 5 : 
    mos = 151*86400;
    break;
  case 6 : 
    mos = 181*86400;
    break;
  case 7 : 
    mos = 212*86400;
    break;
  case 8 : 
    mos = 243*86400;
    break;
  case 9 : 
    mos = 273*86400;
    break;
  case 10 : 
    mos = 304*86400;
    break;
  case 11 : 
    mos = 334*86400;
    break;
}
  int das = (day - 1)*86400;
  int hos = hou*3600;
  int mis = min*60;
  int utime = yes + mos + das + hos + mis + sec;
  return utime;
}




coral::TimeStamp UTtoT(int utime) 
{
  int yea = static_cast<int>(trunc(utime/31536000) + 1970);
  int yes = (yea-1970)*31536000;
  int cony = (yea-1972)%4;
  if (cony == 0) yes = yes + (yea-1972)/4*86400; 
  else yes = yes +  static_cast<int>(trunc((yea-1972)/4))*86400;
  int day = static_cast<int>(trunc((utime - yes)/86400));
  int rest = static_cast<int>(utime - yes - day*86400);
  int mon = 0;
  // BISESTILE YEAR
  if (cony == 0) {
   day = day + 1; 
   if (day < 32){
      mon = 1;
      day = day - 0;
    }
    if (day >= 32 && day < 61){
      mon = 2;
      day = day - 31;
    }
    if (day >= 61 && day < 92){
      mon = 3;
      day = day - 60;
    }
    if (day >= 92 && day < 122){
      mon = 4;
      day = day - 91;
    }
    if (day >= 122 && day < 153){
      mon = 5;
      day = day - 121;
    }
    if (day >= 153 && day < 183){
      mon = 6;
      day = day - 152;
    }
    if (day >= 183 && day < 214){
      mon = 7;
      day = day - 182;
    }
    if (day >= 214 && day < 245){
      mon = 8;
      day = day - 213;
    }
    if (day >= 245 && day < 275){
      mon = 9;
      day = day - 244;
    }
    if (day >= 275 && day < 306){
      mon = 10;
      day = day - 274;
    }
    if (day >= 306 && day < 336){
      mon = 11;
      day = day - 305;
    }
    if (day >= 336){
      mon = 12;
      day = day - 335;
    }
  }
  // NOT BISESTILE YEAR
  else {
    if (day < 32){
      mon = 1;   
      day = day - 0;
    }
    if (day >= 32 && day < 60){
      mon = 2;
      day = day - 31;
    }
    if (day >= 60 && day < 91){
      mon = 3;
      day = day - 59;
    }
    if (day >= 91 && day < 121){
      mon = 4;
      day = day - 90;
    }
    if (day >= 121 && day < 152){
      mon = 5;
      day = day - 120;
    }
    if (day >= 152 && day < 182){
      mon = 6;
      day = day - 151;
    }
    if (day >= 182 && day < 213){
      mon = 7;
      day = day - 181;
    }
    if (day >= 213 && day < 244){
      mon = 8;
      day = day - 212;
    }
    if (day >= 244 && day < 274){
      mon = 9;
      day = day - 243;
    }
    if (day >= 274 && day < 305){
      mon = 10;
      day = day - 273;
    }
    if (day >= 305 && day < 335){
      mon = 11;
      day = day - 304;
    }
    if (day >= 335){
      mon = 12;
      day = day - 334;
    }
  }
  
  int hou = static_cast<int>(trunc(rest/3600)); 
  rest = rest - hou*3600;
  int min = static_cast<int>(trunc(rest/60));
  rest = rest - min*60;
  int sec = rest; 
  int nan = 0;

  //  std::cout <<">> Processing since: "<<day<<"/"<<mon<<"/"<<yea<<" "<<hou<<":"<<min<<"."<<sec<< std::endl;

  coral::TimeStamp Tthr;  

  Tthr = coral::TimeStamp(yea, mon, day, hou, min, sec, nan);
  return Tthr;
}
