#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <ctype.h>
 #include <stdlib.h>

using namespace std;

int main(int argc, char *argv[])
{
  ifstream file1(argv[1]);
  ifstream file2(argv[2]);
  float pedThresh = atof(argv[3]);
  cout<<"diff between "<<argv[1]<<" and  "<<argv[2]<<" w/ threshhold: "<<pedThresh<<endl;

 std::string line1, line2;

  int iEta1, iEta2;
  int iPhi1, iPhi2;
  int  depth1, depth2;
  //TString sdName;
  string sdName1, sdName2;
  int  detId1, detId2;
  float value1[8], value2[8];

  float avg_ped1, avg_ped2, avg_ped_diff;
  float avg_pedwidth, avg_pedwidth2, avg_pedwidth_diff;

 while (getline(file1, line1) && getline(file2, line2))
   {
     if(!line1.size() || line1[0]=='#')    continue;
     std::istringstream linestream1(line1);
     std::istringstream linestream2(line2);
     //cout<<line1<<endl;
     //cout<<line2<<endl;

     //specifically for gains:
     linestream1 >> iEta1 >> iPhi1 >> depth1 >> sdName1 >> value1[0] >>value1[1]>>value1[2]>>value1[3] >> value1[4] 
>>value1[5]>>value1[6]>>value1[7]>>hex >> detId1;
     linestream2 >> iEta2 >> iPhi2 >> depth2 >> sdName2 >> value2[0] >>value2[1]>>value2[2]>>value2[3] >> value2[4] 
>>value2[5]>>value2[6]>>value2[7]>>hex >> detId2;
     float diff[8];
     
     avg_ped1 =  0.25*(value1[0]+value1[1]+value1[2]+value1[3]);
     avg_ped2 =  0.25*(value2[0]+value2[1]+value2[2]+value2[3]);
     avg_ped_diff = avg_ped1-avg_ped2;

     diff[0] = value1[0]-value2[0];
     diff[1] = value1[1]-value2[1];
     diff[2] = value1[2]-value2[2];
     diff[3] = value1[3]-value2[3];
     diff[4] = value1[4]-value2[4];
     diff[5] = value1[5]-value2[5];
     diff[6] = value1[6]-value2[6];
     diff[7] = value1[7]-value2[7];

     /*
     if (fabs(diff[0])>pedThresh || fabs(diff[1])>pedThresh || fabs(diff[2])>pedThresh || fabs(diff[3])>pedThresh || 
fabs(diff[4])>pedThresh || fabs(diff[5])>pedThresh || fabs(diff[6])>pedThresh || fabs(diff[7])>pedThresh)
       cout<<dec<<iEta1<<"  "<<iPhi1<<"  "<<sdName1<<"  "<<diff[0]<<"  "<<diff[1]<<"  "<<diff[2]<<"  "<<diff[3]<<"  "<<diff[4]<<"  "<<diff[5]<<"  "<<diff[6]<<"  "<<diff[7]<<"   "<<hex<< uppercase<<detId1<<endl;
     */

     /*     
     if (fabs(avg_ped_diff)>pedThresh)
cout<<dec<<iEta1<<"  "<<iPhi1<<"  "<<sdName1<<"  "<<diff[0]<<"  "<<diff[1]<<"  "<<diff[2]<<"  "<<diff[3]<<"  "<<diff[4]<<"  "<<diff[5]<<"  "<<diff[6]<<"  "<<diff[7]<<"   "<<hex<< uppercase<<detId1<<endl;
     */
     if (fabs(avg_ped_diff)>pedThresh)
       cout<<"("<<dec<<iEta1<<"  "<<iPhi1<<"  "<<depth1<<"  "<<sdName1<<") "<< hex<< uppercase<<detId1<< "\t average of 4 cap-ids changed by:  "<<avg_ped_diff<<endl;
     

//     cTree -> Fill();

   }


}


