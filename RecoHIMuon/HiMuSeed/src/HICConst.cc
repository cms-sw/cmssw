#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include <iostream>
namespace cms {
void HICConst::setVertex(double a){zvert=a;}
HICConst::HICConst()
{

// layer selection for propagation.

numbar=8;
numbarlost=7;
nofirstpixel=1;

numfrw=9;
nummxt=5;
nminus_gen=0;
nplus_gen=1;

numbargroup[0] = 8;
numbargroup[1] = 7;
numbargroup[2] = 1;

numfrwgroup[0] = 1;  // 8
numfrwgroup[1] = 1;
numfrwgroup[2] = 1;
numfrwgroup[3] = 1;
numfrwgroup[4] = 1;
numfrwgroup[5] = 1;
numfrwgroup[6] = 1;
numfrwgroup[7] = 1;
numfrwgroup[8] = 1;
numfrwgroup[9] = 1;

// last layer = 12
barlay[0][0][0]=0;
barlay[0][0][1]=1;
barlay[0][0][2]=2;
barlay[0][0][3]=8;
barlay[0][0][4]=9;
barlay[0][0][5]=10;
barlay[0][0][6]=11;
barlay[0][0][7]=12;

barlay[0][1][0]=0;
barlay[0][1][1]=1;
barlay[0][1][2]=2;
barlay[0][1][3]=8;
barlay[0][1][4]=9;
barlay[0][1][5]=10;
barlay[0][1][6]=-1;
barlay[0][1][7]=12;

barlay[0][2][0]=0;
barlay[0][2][1]=1;
barlay[0][2][2]=2;
barlay[0][2][3]=8;
barlay[0][2][4]=9;
barlay[0][2][5]=-1;
barlay[0][2][6]=11;
barlay[0][2][7]=12;

barlay[0][3][0]=0;
barlay[0][3][1]=1;
barlay[0][3][2]=2;
barlay[0][3][3]=8;
barlay[0][3][4]=-1;
barlay[0][3][5]=10;
barlay[0][3][6]=11;
barlay[0][3][7]=12;

barlay[0][4][0]=0;
barlay[0][4][1]=1;
barlay[0][4][2]=2;
barlay[0][4][3]=-1;
barlay[0][4][4]=9;
barlay[0][4][5]=10;
barlay[0][4][6]=11;
barlay[0][4][7]=12;

barlay[0][5][0]=0;
barlay[0][5][1]=1;
barlay[0][5][2]=-1;
barlay[0][5][3]=8;
barlay[0][5][4]=9;
barlay[0][5][5]=10;
barlay[0][5][6]=11;
barlay[0][5][7]=12;

barlay[0][6][0]=0;
barlay[0][6][1]=-1;
barlay[0][6][2]=2;
barlay[0][6][3]=8;
barlay[0][6][4]=9;
barlay[0][6][5]=10;
barlay[0][6][6]=11;
barlay[0][6][7]=12;

barlay[0][7][0]=-1;
barlay[0][7][1]=1;
barlay[0][7][2]=2;
barlay[0][7][3]=8;
barlay[0][7][4]=9;
barlay[0][7][5]=10;
barlay[0][7][6]=11;
barlay[0][7][7]=12;

//
// Last layer 11
//
barlay[1][0][0]=0;
barlay[1][0][1]=1;
barlay[1][0][2]=2;
barlay[1][0][3]=8;
barlay[1][0][4]=9;
barlay[1][0][5]=10;
barlay[1][0][6]=11;
barlay[1][0][7]=-1;

barlay[1][1][0]=0;
barlay[1][1][1]=1;
barlay[1][1][2]=2;
barlay[1][1][3]=8;
barlay[1][1][4]=9;
barlay[1][1][5]=-1;
barlay[1][1][6]=11;
barlay[1][1][7]=-1;

barlay[1][2][0]=0;
barlay[1][2][1]=1;
barlay[1][2][2]=2;
barlay[1][2][3]=8;
barlay[1][2][4]=-1;
barlay[1][2][5]=10;
barlay[1][2][6]=11;
barlay[1][2][7]=-1;

barlay[1][3][0]=0;
barlay[1][3][1]=1;
barlay[1][3][2]=2;
barlay[1][3][3]=-1;
barlay[1][3][4]=9;
barlay[1][3][5]=10;
barlay[1][3][6]=11;
barlay[1][3][7]=-1;

barlay[1][4][0]=0;
barlay[1][4][1]=1;
barlay[1][4][2]=-1;
barlay[1][4][3]=8;
barlay[1][4][4]=9;
barlay[1][4][5]=10;
barlay[1][4][6]=11;
barlay[1][4][7]=-1;

barlay[1][5][0]=0;
barlay[1][5][1]=-1;
barlay[1][5][2]=2;
barlay[1][5][3]=8;
barlay[1][5][4]=9;
barlay[1][5][5]=10;
barlay[1][5][6]=11;
barlay[1][5][7]=-1;

barlay[1][6][0]=-1;
barlay[1][6][1]=1;
barlay[1][6][2]=2;
barlay[1][6][3]=8;
barlay[1][6][4]=9;
barlay[1][6][5]=10;
barlay[1][6][6]=11;
barlay[1][6][7]=-1;

//
// Last layer = 10
//

barlay[2][0][0]=0;
barlay[2][0][1]=1;
barlay[2][0][2]=2;
barlay[2][0][3]=8;
barlay[2][0][4]=9;
barlay[2][0][5]=10;
barlay[2][0][6]=-1;
barlay[2][0][7]=-1;

//cout<<" HICConst::barlay "<<barlay[6][5]<<endl;

frwlay[0][0][0]=0;
frwlay[0][0][1]=1;
frwlay[0][0][2]=5;
frwlay[0][0][3]=6;
frwlay[0][0][4]=7;
frwlay[0][0][5]=8;
frwlay[0][0][6]=9;
frwlay[0][0][7]=10;
frwlay[0][0][8]=11;
frwlay[0][0][9]=12;
frwlay[0][0][10]=13;

// Lost layer if 13 layer is the last

frwlay[0][1][0]=0;
frwlay[0][1][1]=1;
frwlay[0][1][2]=5;
frwlay[0][1][3]=6;
frwlay[0][1][4]=7;
frwlay[0][1][5]=8;
frwlay[0][1][6]=9;
frwlay[0][1][7]=10;
frwlay[0][1][8]=11;
frwlay[0][1][9]=-1;
frwlay[0][1][10]=13;


frwlay[0][2][0]=0;
frwlay[0][2][1]=1;
frwlay[0][2][2]=5;
frwlay[0][2][3]=6;
frwlay[0][2][4]=7;
frwlay[0][2][5]=8;
frwlay[0][2][6]=9;
frwlay[0][2][7]=10;
frwlay[0][2][8]=-1;
frwlay[0][2][9]=12;
frwlay[0][2][10]=13;


frwlay[0][3][0]=0;
frwlay[0][3][1]=1;
frwlay[0][3][2]=5;
frwlay[0][3][3]=6;
frwlay[0][3][4]=7;
frwlay[0][3][5]=8;
frwlay[0][3][6]=9;
frwlay[0][3][7]=-1;
frwlay[0][3][8]=11;
frwlay[0][3][9]=12;
frwlay[0][3][10]=13;


frwlay[0][4][0]=0;
frwlay[0][4][1]=1;
frwlay[0][4][2]=5;
frwlay[0][4][3]=6;
frwlay[0][4][4]=7;
frwlay[0][4][5]=8;
frwlay[0][4][6]=-1;
frwlay[0][4][7]=10;
frwlay[0][4][8]=11;
frwlay[0][4][9]=12;
frwlay[0][4][10]=13;


frwlay[0][5][0]=0;
frwlay[0][5][1]=1;
frwlay[0][5][2]=5;
frwlay[0][5][3]=6;
frwlay[0][5][4]=7;
frwlay[0][5][5]=-1;
frwlay[0][5][6]=9;
frwlay[0][5][7]=10;
frwlay[0][5][8]=11;
frwlay[0][5][9]=12;
frwlay[0][5][10]=13;


frwlay[0][6][0]=0;
frwlay[0][6][1]=1;
frwlay[0][6][2]=5;
frwlay[0][6][3]=6;
frwlay[0][6][4]=-1;
frwlay[0][6][5]=8;
frwlay[0][6][6]=9;
frwlay[0][6][7]=10;
frwlay[0][6][8]=11;
frwlay[0][6][9]=12;
frwlay[0][6][10]=13;

frwlay[0][7][0]=0;
frwlay[0][7][1]=1;
frwlay[0][7][2]=5;
frwlay[0][7][3]=-1;
frwlay[0][7][4]=7;
frwlay[0][7][5]=8;
frwlay[0][7][6]=9;
frwlay[0][7][7]=10;
frwlay[0][7][8]=11;
frwlay[0][7][9]=12;
frwlay[0][7][10]=13;


frwlay[0][8][0]=0;
frwlay[0][8][1]=1;
frwlay[0][8][2]=-1;
frwlay[0][8][3]=6;
frwlay[0][8][4]=7;
frwlay[0][8][5]=8;
frwlay[0][8][6]=9;
frwlay[0][8][7]=10;
frwlay[0][8][8]=11;
frwlay[0][8][9]=12;
frwlay[0][8][10]=13;


frwlay[0][9][0]=0;
frwlay[0][9][1]=-1;
frwlay[0][9][2]=5;
frwlay[0][9][3]=6;
frwlay[0][9][4]=7;
frwlay[0][9][5]=8;
frwlay[0][9][6]=9;
frwlay[0][9][7]=10;
frwlay[0][9][8]=11;
frwlay[0][9][9]=12;
frwlay[0][9][10]=13;

frwlay[0][10][0]=-1;
frwlay[0][10][1]=1;
frwlay[0][10][2]=5;
frwlay[0][10][3]=6;
frwlay[0][10][4]=7;
frwlay[0][10][5]=8;
frwlay[0][10][6]=9;
frwlay[0][10][7]=10;
frwlay[0][10][8]=11;
frwlay[0][10][9]=12;
frwlay[0][10][10]=13;

//----------------------------


frwlay[1][0][0]=0;
frwlay[1][0][1]=1;
frwlay[1][0][2]=5;
frwlay[1][0][3]=6;
frwlay[1][0][4]=7;
frwlay[1][0][5]=8;
frwlay[1][0][6]=9;
frwlay[1][0][7]=10;
frwlay[1][0][8]=11;
frwlay[1][0][9]=12;
frwlay[1][0][10]=-1;

frwlay[2][0][0]=0;
frwlay[2][0][1]=1;
frwlay[2][0][2]=5;
frwlay[2][0][3]=6;
frwlay[2][0][4]=7;
frwlay[2][0][5]=8;
frwlay[2][0][6]=9;
frwlay[2][0][7]=10;
frwlay[2][0][8]=11;
frwlay[2][0][9]=-1;
frwlay[2][0][10]=-1;

frwlay[3][0][0]=0;
frwlay[3][0][1]=1;
frwlay[3][0][2]=5;
frwlay[3][0][3]=6;
frwlay[3][0][4]=7;
frwlay[3][0][5]=8;
frwlay[3][0][6]=9;
frwlay[3][0][7]=10;
frwlay[3][0][8]=-1;
frwlay[3][0][9]=-1;
frwlay[3][0][10]=-1;

frwlay[4][0][0]=0;
frwlay[4][0][1]=1;
frwlay[4][0][2]=5;
frwlay[4][0][3]=6;
frwlay[4][0][4]=7;
frwlay[4][0][5]=8;
frwlay[4][0][6]=9;
frwlay[4][0][7]=-1;
frwlay[4][0][8]=-1;
frwlay[4][0][9]=-1;
frwlay[4][0][10]=-1;


frwlay[5][0][0]=0;
frwlay[5][0][1]=1;
frwlay[5][0][2]=5;
frwlay[5][0][3]=6;
frwlay[5][0][4]=7;
frwlay[5][0][5]=8;
frwlay[5][0][6]=-1;
frwlay[5][0][7]=-1;
frwlay[5][0][8]=-1;
frwlay[5][0][9]=-1;
frwlay[5][0][10]=-1;

frwlay[6][0][0]=0;
frwlay[6][0][1]=1;
frwlay[6][0][2]=5;
frwlay[6][0][3]=6;
frwlay[6][0][4]=7;
frwlay[6][0][5]=-1;
frwlay[6][0][6]=-1;
frwlay[6][0][7]=-1;
frwlay[6][0][8]=-1;
frwlay[6][0][9]=-1;
frwlay[6][0][10]=-1;

frwlay[7][0][0]=0;
frwlay[7][0][1]=1;
frwlay[7][0][2]=5;
frwlay[7][0][3]=6;
frwlay[7][0][4]=-1;
frwlay[7][0][5]=-1;
frwlay[7][0][6]=-1;
frwlay[7][0][7]=-1;
frwlay[7][0][8]=-1;
frwlay[7][0][9]=-1;
frwlay[7][0][10]=-1;

frwlay[8][0][0]=0;
frwlay[8][0][1]=1;
frwlay[8][0][2]=5;
frwlay[8][0][3]=-1;
frwlay[8][0][4]=-1;
frwlay[8][0][5]=-1;
frwlay[8][0][6]=-1;
frwlay[8][0][7]=-1;
frwlay[8][0][8]=-1;
frwlay[8][0][9]=-1;
frwlay[8][0][10]=-1;


mxtlay[0]=8; mxtlay[1]=9; mxtlay[2]=10; mxtlay[3]=11; mxtlay[4]=12;

phias[0]=150.;
phias[1]=290.;
phias[2]=290.;
phias[3]=290.;
phias[4]=290.;
phias[5]=290.;
phias[6]=290.;
phias[7]=310.;
phias[8]=285.;
phias[9]=285.;
phias[10]=310.;
phias[11]=310.;
phias[12]=295.;
phias[13]=295.;
phias[14]=293.;
phias[15]=0.24;
phias[16]=0.24;
phias[17]=0.27;
phias[18]=0.27;
phias[19]=0.27;
phias[20]=0.09;
phias[21]=0.24;
phias[22]=0.24;
phias[23]=0.27;
phias[24]=0.27;
phias[25]=0.27;
// phias[26]=0.09; ORCA
phias[26]=0.09;
phias[27]=3.;
//--------------------------------------------------------------------
phibs[0]=100.;
phibs[1]=160.;
phibs[2]=160.;
phibs[3]=160.;
phibs[4]=160.;
phibs[5]=160.;
phibs[6]=160.;
phibs[7]=160.;
phibs[8]=150.;
phibs[9]=150.;
phibs[10]=160.;
phibs[11]=160.;
phibs[12]=160.;
phibs[13]=160.;
phibs[14]=154.;
phibs[15]=0.4;
phibs[16]=0.4;
phibs[17]=0.44;
phibs[18]=0.44;
phibs[19]=0.44;
phibs[20]=0.5;
phibs[21]=0.4;
phibs[22]=0.4;
phibs[23]=0.44;
phibs[24]=0.44;
phibs[25]=0.44;
// phibs[26]=0.4; ORCA
phibs[26]= 0.45; 
phibs[27]=55.;
//--------------------------------------------------------------------
phiai[0]=600.;
phiai[1]=385.;
phiai[2]=385.;
phiai[3]=385.;
phiai[4]=385.;
phiai[5]=385.;
phiai[6]=385.;
phiai[7]=370.;
phiai[8]=380.;
phiai[9]=380.;
phiai[10]=370.;
phiai[11]=370.;
phiai[12]=370.;
phiai[13]=370.;
phiai[14]=360.;
phiai[15]=0.6;
phiai[16]=0.6;
phiai[17]=0.8;
phiai[18]=0.8;
phiai[19]=0.8;
phiai[20]=0.9;
phiai[21]=0.6;
phiai[22]=0.6;
phiai[23]=0.8;
phiai[24]=0.8;
phiai[25]=0.8;
// phiai[26]=0.4; ORCA
phiai[26]=0.6;
phiai[27]=430.;
//--------------------------------------------------------------------
phibi[0]=310.;
phibi[1]=170.;
phibi[2]=170.;
phibi[3]=170.;
phibi[4]=170.;
phibi[5]=170.;
phibi[6]=170.;
phibi[7]=170.;
phibi[8]=190.;
phibi[9]=190.;
phibi[10]=170.;
phibi[11]=170.;
phibi[12]=170.;
phibi[13]=170.;
phibi[14]=183.;
phibi[15]=0.6;
phibi[16]=0.6;
phibi[17]=0.8;
phibi[18]=0.8;
phibi[19]=0.8;
phibi[20]=1.0;
phibi[21]=0.6;
phibi[22]=0.6;
phibi[23]=0.8;
phibi[24]=0.8;
phibi[25]=0.8;
// phibi[26]=0.7; ORCA
phibi[26]=1.;
phibi[27]=300.;
//--------------------------------------------------------------------
			 
newparam[0]=-0.67927; newparam[1]=0.53635; newparam[2]=-0.00457;
newparamgt40[0]=0.38813; newparamgt40[1]=0.41003; newparamgt40[2]=-0.0019956;
forwparam[0]=0.0307;forwparam[1]=3.475;

// size of window in phi-z.
// Barrel, phiwinbar (last layer in barrel)
 
// +++++++++++ Last layer = 12 

phiwinbar[12][12][11] = 0.1;
phiwinbar[12][12][10] = 0.14;
phiwinbar[12][12][9] = 0.14;
phiwinbar[12][12][8] = 0.14;
phiwinbar[12][12][7] = 0.14;
phiwinbar[12][12][6] = 0.14;
phiwinbar[12][12][5] = 0.14;
phiwinbar[12][12][4] = 0.14;
phiwinbar[12][12][3] = 0.14;
phiwinbar[12][12][2] = 0.14;
phiwinbar[12][12][1] = 0.14;
phiwinbar[12][12][0] = 0.14;

phiwinbar[12][11][10] = 0.0011; //0.0009
phiwinbar[12][11][9] = 0.0018;
phiwinbar[12][11][8] = 0.0027;
phiwinbar[12][11][7] = 0.0009;
phiwinbar[12][11][6] = 0.0009;
phiwinbar[12][11][5] = 0.0009;
phiwinbar[12][11][4] = 0.0009;
phiwinbar[12][11][3] = 0.0009;
phiwinbar[12][11][2] = 0.01;
phiwinbar[12][11][1] = 0.003;
phiwinbar[12][11][0] = 0.0015;

phiwinbar[12][10][9] = 0.0011; //0.0009
phiwinbar[12][10][8] = 0.0018;
phiwinbar[12][10][7] = 0.0027;
phiwinbar[12][10][6] = 0.0009;
phiwinbar[12][10][5] = 0.0009;
phiwinbar[12][10][4] = 0.0009;
phiwinbar[12][10][3] = 0.0009;
phiwinbar[12][10][2] = 0.01;
phiwinbar[12][10][1] = 0.003;
phiwinbar[12][10][0] = 0.0015;

phiwinbar[12][9][8] = 0.0014; //0.0009
phiwinbar[12][9][7] = 0.0018;
phiwinbar[12][9][6] = 0.0027;
phiwinbar[12][9][5] = 0.0009;
phiwinbar[12][9][4] = 0.0009;
phiwinbar[12][9][3] = 0.0009;
phiwinbar[12][9][2] = 0.06;
phiwinbar[12][9][1] = 0.003;
phiwinbar[12][9][0] = 0.0015;

phiwinbar[12][8][7] = 0.0011; //0.0009
phiwinbar[12][8][6] = 0.0018;
phiwinbar[12][8][5] = 0.0027;
phiwinbar[12][8][4] = 0.0009;
phiwinbar[12][8][3] = 0.0009;
phiwinbar[12][8][2] = 0.06;
phiwinbar[12][8][1] = 0.003;
phiwinbar[12][8][0] = 0.0015;

phiwinbar[12][7][6] = 0.0009;
phiwinbar[12][7][5] = 0.0018;
phiwinbar[12][7][4] = 0.00027;
phiwinbar[12][7][3] = 0.028;
phiwinbar[12][7][2] = 0.035;
phiwinbar[12][7][1] = 0.042;
phiwinbar[12][7][0] = 0.049;

phiwinbar[12][6][5] = 0.0009;
phiwinbar[12][6][4] = 0.0018;
phiwinbar[12][6][3] = 0.0027;
phiwinbar[12][6][2] = 0.028;
phiwinbar[12][6][1] = 0.035;
phiwinbar[12][6][0] = 0.042;

phiwinbar[12][5][4] = 0.0009;
phiwinbar[12][5][3] = 0.0018;
phiwinbar[12][5][2] = 0.027;
phiwinbar[12][5][1] = 0.028;
phiwinbar[12][5][0] = 0.035;

phiwinbar[12][4][3] = 0.0009;
phiwinbar[12][4][2] = 0.0018;
phiwinbar[12][4][1] = 0.0027;
phiwinbar[12][4][0] = 0.028;

phiwinbar[12][3][2] = 0.005;
phiwinbar[12][3][1] = 0.009;
phiwinbar[12][3][0] = 0.021;

phiwinbar[12][2][1] = 0.005;
phiwinbar[12][2][0] = 0.006;

phiwinbar[12][1][0] = 0.005;

// +++++++++++ Last layer = 11 

phiwinbar[11][11][10] = 0.1;
phiwinbar[11][11][9] = 0.1;
phiwinbar[11][11][8] = 0.1;
phiwinbar[11][11][7] = 0.1;
phiwinbar[11][11][6] = 0.1;
phiwinbar[11][11][5] = 0.1;
phiwinbar[11][11][4] = 0.1;
phiwinbar[11][11][3] = 0.1;
phiwinbar[11][11][2] = 0.1;
phiwinbar[11][11][1] = 0.1;
phiwinbar[11][11][0] = 0.1;

phiwinbar[11][10][9] = 0.0009;
phiwinbar[11][10][8] = 0.0018;
phiwinbar[11][10][7] = 0.0027;
phiwinbar[11][10][6] = 0.0009;
phiwinbar[11][10][5] = 0.0009;
phiwinbar[11][10][4] = 0.0009;
phiwinbar[11][10][3] = 0.0009;
phiwinbar[11][10][2] = 0.06;
phiwinbar[11][10][1] = 0.003;
phiwinbar[11][10][0] = 0.0015;

phiwinbar[11][9][8] = 0.0011;
phiwinbar[11][9][7] = 0.0018;
phiwinbar[11][9][6] = 0.0027;
phiwinbar[11][9][5] = 0.0009;
phiwinbar[11][9][4] = 0.0009;
phiwinbar[11][9][3] = 0.0009;
phiwinbar[11][9][2] = 0.06;
phiwinbar[11][9][1] = 0.003;
phiwinbar[11][9][0] = 0.0015;

phiwinbar[11][8][7] = 0.0009;
phiwinbar[11][8][6] = 0.0018;
phiwinbar[11][8][5] = 0.0027;
phiwinbar[11][8][4] = 0.0009;
phiwinbar[11][8][3] = 0.0009;
phiwinbar[11][8][2] = 0.06;
phiwinbar[11][8][1] = 0.01;
phiwinbar[11][8][0] = 0.01;

phiwinbar[11][7][6] = 0.0009;
phiwinbar[11][7][5] = 0.0018;
phiwinbar[11][7][4] = 0.00027;
phiwinbar[11][7][3] = 0.028;
phiwinbar[11][7][2] = 0.035;
phiwinbar[11][7][1] = 0.042;
phiwinbar[11][7][0] = 0.049;

phiwinbar[11][6][5] = 0.0009;
phiwinbar[11][6][4] = 0.0018;
phiwinbar[11][6][3] = 0.0027;
phiwinbar[11][6][2] = 0.028;
phiwinbar[11][6][1] = 0.035;
phiwinbar[11][6][0] = 0.042;

phiwinbar[11][5][4] = 0.0009;
phiwinbar[11][5][3] = 0.0018;
phiwinbar[11][5][2] = 0.0027;
phiwinbar[11][5][1] = 0.028;
phiwinbar[11][5][0] = 0.035;

phiwinbar[11][4][3] = 0.0009;
phiwinbar[11][4][2] = 0.0018;
phiwinbar[11][4][1] = 0.0027;
phiwinbar[11][4][0] = 0.028;

phiwinbar[11][3][2] = 0.005;
phiwinbar[11][3][1] = 0.009;
phiwinbar[11][3][0] = 0.021;

phiwinbar[11][2][1] = 0.005;
phiwinbar[11][2][0] = 0.006;

phiwinbar[11][1][0] = 0.005;

// +++++++++++ Last layer = 10 

phiwinbar[10][10][9] = 0.1;
phiwinbar[10][10][8] = 0.1;
phiwinbar[10][10][7] = 0.1;
phiwinbar[10][10][6] = 0.1;
phiwinbar[10][10][5] = 0.1;
phiwinbar[10][10][4] = 0.1;
phiwinbar[10][10][3] = 0.1;
phiwinbar[10][10][2] = 0.1;
phiwinbar[10][10][1] = 0.1;
phiwinbar[10][10][0] = 0.1;

phiwinbar[10][9][8] = 0.0009;
phiwinbar[10][9][7] = 0.0018;
phiwinbar[10][9][6] = 0.0027;
phiwinbar[10][9][5] = 0.0009;
phiwinbar[10][9][4] = 0.0009;
phiwinbar[10][9][3] = 0.0009;
phiwinbar[10][9][2] = 0.06;
phiwinbar[10][9][1] = 0.003;
phiwinbar[10][9][0] = 0.0015;

phiwinbar[10][8][7] = 0.0009;
phiwinbar[10][8][6] = 0.0018;
phiwinbar[10][8][5] = 0.0027;
phiwinbar[10][8][4] = 0.0009;
phiwinbar[10][8][3] = 0.0009;
phiwinbar[10][8][2] = 0.06;
phiwinbar[10][8][1] = 0.003;
phiwinbar[10][8][0] = 0.0015;

phiwinbar[10][7][6] = 0.0009;
phiwinbar[10][7][5] = 0.0018;
phiwinbar[10][7][4] = 0.00027;
phiwinbar[10][7][3] = 0.028;
phiwinbar[10][7][2] = 0.035;
phiwinbar[10][7][1] = 0.042;
phiwinbar[10][7][0] = 0.049;

phiwinbar[10][6][5] = 0.0009;
phiwinbar[10][6][4] = 0.0018;
phiwinbar[10][6][3] = 0.0027;
phiwinbar[10][6][2] = 0.028;
phiwinbar[10][6][1] = 0.035;
phiwinbar[10][6][0] = 0.042;

phiwinbar[10][5][4] = 0.0009;
phiwinbar[10][5][3] = 0.0018;
phiwinbar[10][5][2] = 0.0027;
phiwinbar[10][5][1] = 0.028;
phiwinbar[10][5][0] = 0.035;

phiwinbar[10][4][3] = 0.0009;
phiwinbar[10][4][2] = 0.0018;
phiwinbar[10][4][1] = 0.0027;
phiwinbar[10][4][0] = 0.028;

phiwinbar[10][3][2] = 0.005;
phiwinbar[10][3][1] = 0.009;
phiwinbar[10][3][0] = 0.021;

phiwinbar[10][2][1] = 0.005;
phiwinbar[10][2][0] = 0.006;

phiwinbar[10][1][0] = 0.005;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// size of update cut in phi-z for the trajectories.
// Barrel (last layer in barrel) phicutbar
// +++++++++++ Last layer = 12 

phicutbar[12][12][11] = 0.0002;
phicutbar[12][12][10] = 0.0004;
phicutbar[12][12][9] = 0.0008;
phicutbar[12][12][8] = 0.0008;
phicutbar[12][12][7] = 0.0008;
phicutbar[12][12][6] = 0.0008;
phicutbar[12][12][5] = 0.0008;
phicutbar[12][12][4] = 0.0008;
phicutbar[12][12][3] = 0.0008;
phicutbar[12][12][2] = 0.0008;
phicutbar[12][12][1] = 0.0008;
phicutbar[12][12][0] = 0.0008;

phicutbar[12][11][10] = 0.0004;
phicutbar[12][11][9] = 0.0009;
phicutbar[12][11][8] = 0.0018;
phicutbar[12][11][7] = 0.0009;
phicutbar[12][11][6] = 0.0009;
phicutbar[12][11][5] = 0.0009;
phicutbar[12][11][4] = 0.0009;
phicutbar[12][11][3] = 0.0009;
phicutbar[12][11][2] = 0.005;
phicutbar[12][11][1] = 0.003;
phicutbar[12][11][0] = 0.0015;

phicutbar[12][10][9] = 0.0011;
phicutbar[12][10][8] = 0.0018;
phicutbar[12][10][7] = 0.0027;
phicutbar[12][10][6] = 0.0009;
phicutbar[12][10][5] = 0.0009;
phicutbar[12][10][4] = 0.0009;
phicutbar[12][10][3] = 0.0009;
phicutbar[12][10][2] = 0.005;
phicutbar[12][10][1] = 0.003;
phicutbar[12][10][0] = 0.0015;

phicutbar[12][9][8] = 0.0014;
phicutbar[12][9][7] = 0.0018;
phicutbar[12][9][6] = 0.0027;
phicutbar[12][9][5] = 0.0009;
phicutbar[12][9][4] = 0.0009;
phicutbar[12][9][3] = 0.0009;
phicutbar[12][9][2] = 0.01;
phicutbar[12][9][1] = 0.01;
phicutbar[12][9][0] = 0.01;

phicutbar[12][8][7] = 0.0009;
phicutbar[12][8][6] = 0.0018;
phicutbar[12][8][5] = 0.0027;
phicutbar[12][8][4] = 0.0009;
phicutbar[12][8][3] = 0.0009;
phicutbar[12][8][2] = 0.004;
phicutbar[12][8][1] = 0.005;
phicutbar[12][8][0] = 0.005;

phicutbar[12][7][6] = 0.0009;
phicutbar[12][7][5] = 0.0018;
phicutbar[12][7][4] = 0.00027;
phicutbar[12][7][3] = 0.028;
phicutbar[12][7][2] = 0.035;
phicutbar[12][7][1] = 0.042;
phicutbar[12][7][0] = 0.049;

phicutbar[12][6][5] = 0.0009;
phicutbar[12][6][4] = 0.0018;
phicutbar[12][6][3] = 0.0027;
phicutbar[12][6][2] = 0.028;
phicutbar[12][6][1] = 0.035;
phicutbar[12][6][0] = 0.042;

phicutbar[12][5][4] = 0.0009;
phicutbar[12][5][3] = 0.0018;
phicutbar[12][5][2] = 0.0027;
phicutbar[12][5][1] = 0.028;
phicutbar[12][5][0] = 0.035;

phicutbar[12][4][3] = 0.0009;
phicutbar[12][4][2] = 0.0018;
phicutbar[12][4][1] = 0.0027;
phicutbar[12][4][0] = 0.028;

phicutbar[12][3][2] = 0.005;
phicutbar[12][3][1] = 0.009;
phicutbar[12][3][0] = 0.021;

phicutbar[12][2][1] = 0.0014;
phicutbar[12][2][0] = 0.002;

phicutbar[12][1][0] = 0.0014;

// +++++++++++ Last layer = 11 

phicutbar[11][11][10] = 0.0002;
phicutbar[11][11][9] = 0.0008;
phicutbar[11][11][8] = 0.0008;
phicutbar[11][11][7] = 0.008;
phicutbar[11][11][6] = 0.008;
phicutbar[11][11][5] = 0.008;
phicutbar[11][11][4] = 0.008;
phicutbar[11][11][3] = 0.008;
phicutbar[11][11][2] = 0.008;
phicutbar[11][11][1] = 0.008;
phicutbar[11][11][0] = 0.008;

phicutbar[11][10][9] = 0.0003;
phicutbar[11][10][8] = 0.0009;
phicutbar[11][10][7] = 0.0018;
phicutbar[11][10][6] = 0.0009;
phicutbar[11][10][5] = 0.0009;
phicutbar[11][10][4] = 0.0009;
phicutbar[11][10][3] = 0.0009;
phicutbar[11][10][2] = 0.005;
phicutbar[11][10][1] = 0.003;
phicutbar[11][10][0] = 0.0015;

phicutbar[11][9][8] = 0.0011;
phicutbar[11][9][7] = 0.0018;
phicutbar[11][9][6] = 0.0027;
phicutbar[11][9][5] = 0.0009;
phicutbar[11][9][4] = 0.0009;
phicutbar[11][9][3] = 0.0009;
phicutbar[11][9][2] = 0.01;
phicutbar[11][9][1] = 0.01;
phicutbar[11][9][0] = 0.01;

phicutbar[11][8][7] = 0.0009;
phicutbar[11][8][6] = 0.0018;
phicutbar[11][8][5] = 0.0027;
phicutbar[11][8][4] = 0.0009;
phicutbar[11][8][3] = 0.0025;
phicutbar[11][8][2] = 0.004;
phicutbar[11][8][1] = 0.005;
phicutbar[11][8][0] = 0.005;

phicutbar[11][7][6] = 0.0009;
phicutbar[11][7][5] = 0.0018;
phicutbar[11][7][4] = 0.00027;
phicutbar[11][7][3] = 0.028;
phicutbar[11][7][2] = 0.035;
phicutbar[11][7][1] = 0.042;
phicutbar[11][7][0] = 0.049;

phicutbar[11][6][5] = 0.0009;
phicutbar[11][6][4] = 0.0018;
phicutbar[11][6][3] = 0.0027;
phicutbar[11][6][2] = 0.028;
phicutbar[11][6][1] = 0.035;
phicutbar[11][6][0] = 0.042;

phicutbar[11][5][4] = 0.0009;
phicutbar[11][5][3] = 0.0018;
phicutbar[11][5][2] = 0.0027;
phicutbar[11][5][1] = 0.028;
phicutbar[11][5][0] = 0.035;

phicutbar[11][4][3] = 0.0009;
phicutbar[11][4][2] = 0.0018;
phicutbar[11][4][1] = 0.0027;
phicutbar[11][4][0] = 0.028;

phicutbar[11][3][2] = 0.005;
phicutbar[11][3][1] = 0.009;
phicutbar[11][3][0] = 0.021;

phicutbar[11][2][1] = 0.0014;
phicutbar[11][2][0] = 0.002;

phicutbar[11][1][0] = 0.0014;

// +++++++++++ Last layer = 10 

phicutbar[10][10][9] = 0.0002;
phicutbar[10][10][8] = 0.0005;
phicutbar[10][10][7] = 0.005;
phicutbar[10][10][6] = 0.005;
phicutbar[10][10][5] = 0.005;
phicutbar[10][10][4] = 0.005;
phicutbar[10][10][3] = 0.005;
phicutbar[10][10][2] = 0.005;
phicutbar[10][10][1] = 0.003;
phicutbar[10][10][0] = 0.0015;

phicutbar[10][9][8] = 0.0008;
phicutbar[10][9][7] = 0.0009;
phicutbar[10][9][6] = 0.0027;
phicutbar[10][9][5] = 0.0009;
phicutbar[10][9][4] = 0.0009;
phicutbar[10][9][3] = 0.0009;
phicutbar[10][9][2] = 0.01;
phicutbar[10][9][1] = 0.01;
phicutbar[10][9][0] = 0.01;

phicutbar[10][8][7] = 0.0009;
phicutbar[10][8][6] = 0.0018;
phicutbar[10][8][5] = 0.0027;
phicutbar[10][8][4] = 0.0009;
phicutbar[10][8][3] = 0.0009;
phicutbar[10][8][2] = 0.004;
phicutbar[10][8][1] = 0.0011;
phicutbar[10][8][0] = 0.0011;

phicutbar[10][7][6] = 0.0009;
phicutbar[10][7][5] = 0.0018;
phicutbar[10][7][4] = 0.00027;
phicutbar[10][7][3] = 0.028;
phicutbar[10][7][2] = 0.035;
phicutbar[10][7][1] = 0.042;
phicutbar[10][7][0] = 0.049;

phicutbar[10][6][5] = 0.0009;
phicutbar[10][6][4] = 0.0018;
phicutbar[10][6][3] = 0.0027;
phicutbar[10][6][2] = 0.028;
phicutbar[10][6][1] = 0.035;
phicutbar[10][6][0] = 0.042;

phicutbar[10][5][4] = 0.0009;
phicutbar[10][5][3] = 0.0018;
phicutbar[10][5][2] = 0.0027;
phicutbar[10][5][1] = 0.028;
phicutbar[10][5][0] = 0.035;

phicutbar[10][4][3] = 0.0009;
phicutbar[10][4][2] = 0.0018;
phicutbar[10][4][1] = 0.0027;
phicutbar[10][4][0] = 0.028;

phicutbar[10][3][2] = 0.005;
phicutbar[10][3][1] = 0.009;
phicutbar[10][3][0] = 0.021;

phicutbar[10][2][1] = 0.0014;
phicutbar[10][2][0] = 0.002;

phicutbar[10][1][0] = 0.0014;

// Roads in barrel if the trajectory starts in forward
// phicutfbb

// ++++++++++++Last layer = 13

phicutfbb[13][12][11] = 0.0009;
phicutfbb[13][12][10] = 0.0018;
phicutfbb[13][12][9] = 0.0027;
phicutfbb[13][12][8] = 0.0027;
phicutfbb[13][12][7] = 0.0027;
phicutfbb[13][12][6] = 0.0027;
phicutfbb[13][12][5] = 0.0027;
phicutfbb[13][12][4] = 0.0027;
phicutfbb[13][12][3] = 0.0027;
phicutfbb[13][12][2] = 0.0027;
phicutfbb[13][12][1] = 0.005;
phicutfbb[13][12][0] = 0.005;

phicutfbb[13][11][10] = 0.0009;
phicutfbb[13][11][9] = 0.0018;
phicutfbb[13][11][8] = 0.0027;
phicutfbb[13][11][7] = 0.0027;
phicutfbb[13][11][6] = 0.0027;
phicutfbb[13][11][5] = 0.0027;
phicutfbb[13][11][4] = 0.0027;
phicutfbb[13][11][3] = 0.0027;
phicutfbb[13][11][2] = 0.0027;
phicutfbb[13][11][1] = 0.005;
phicutfbb[13][11][0] = 0.005;

phicutfbb[13][10][9] = 0.0009;
phicutfbb[13][10][8] = 0.0018;
phicutfbb[13][10][7] = 0.0027;
phicutfbb[13][10][6] = 0.0027;
phicutfbb[13][10][5] = 0.0027;
phicutfbb[13][10][4] = 0.0027;
phicutfbb[13][10][3] = 0.0027;
phicutfbb[13][10][2] = 0.0027;
phicutfbb[13][10][1] = 0.005;
phicutfbb[13][10][0] = 0.005;

phicutfbb[13][9][8] = 0.0027;
phicutfbb[13][9][7] = 0.0027;
phicutfbb[13][9][6] = 0.0027;
phicutfbb[13][9][5] = 0.0027;
phicutfbb[13][9][4] = 0.0027;
phicutfbb[13][9][3] = 0.0027;
phicutfbb[13][9][2] = 0.0027;
phicutfbb[13][9][1] = 0.005;
phicutfbb[13][9][0] = 0.005;

phicutfbb[13][8][7] = 0.0009;
phicutfbb[13][8][6] = 0.0018;
phicutfbb[13][8][5] = 0.0027;
phicutfbb[13][8][4] = 0.0027;
phicutfbb[13][8][3] = 0.0027;
phicutfbb[13][8][2] = 0.0027;
phicutfbb[13][8][1] = 0.005;
phicutfbb[13][8][0] = 0.005;

phicutfbb[13][7][6] = 0.0009;
phicutfbb[13][7][5] = 0.0018;
phicutfbb[13][7][4] = 0.0027;
phicutfbb[13][7][3] = 0.0027;
phicutfbb[13][7][2] = 0.0027;
phicutfbb[13][7][1] = 0.005;
phicutfbb[13][7][0] = 0.005;

phicutfbb[13][6][5] = 0.0009;
phicutfbb[13][6][4] = 0.0018;
phicutfbb[13][6][3] = 0.0027;
phicutfbb[13][6][2] = 0.0027;
phicutfbb[13][6][1] = 0.005;
phicutfbb[13][6][0] = 0.005;

phicutfbb[13][5][4] = 0.0009;
phicutfbb[13][5][3] = 0.0018;
phicutfbb[13][5][2] = 0.0027;
phicutfbb[13][5][1] = 0.005;
phicutfbb[13][5][0] = 0.005;

phicutfbb[13][4][3] = 0.0027;
phicutfbb[13][4][2] = 0.0027;
phicutfbb[13][4][1] = 0.005;
phicutfbb[13][4][0] = 0.005;

phicutfbb[13][3][2] = 0.005;
phicutfbb[13][3][1] = 0.005;
phicutfbb[13][3][0] = 0.005;

phicutfbb[13][2][1] = 0.005;
phicutfbb[13][2][0] = 0.003;

phicutfbb[13][1][0] = 0.0035;

// +++++++++++ Last layer = 12

phicutfbb[12][12][11] = 0.1;
phicutfbb[12][12][10] = 0.1;
phicutfbb[12][12][9] = 0.1;
phicutfbb[12][12][8] = 0.1;
phicutfbb[12][12][7] = 0.1;
phicutfbb[12][12][6] = 0.1;
phicutfbb[12][12][5] = 0.1;
phicutfbb[12][12][4] = 0.1;
phicutfbb[12][12][3] = 0.1;
phicutfbb[12][12][2] = 0.1;
phicutfbb[12][12][1] = 0.1;
phicutfbb[12][12][0] = 0.1;

phicutfbb[12][11][10] = 0.0009;
phicutfbb[12][11][9] = 0.0018;
phicutfbb[12][11][8] = 0.0027;
phicutfbb[12][11][7] = 0.0027;
phicutfbb[12][11][6] = 0.0027;
phicutfbb[12][11][5] = 0.0027;
phicutfbb[12][11][4] = 0.0027;
phicutfbb[12][11][3] = 0.0027;
phicutfbb[12][11][2] = 0.0027;
phicutfbb[12][11][1] = 0.005;
phicutfbb[12][11][0] = 0.005;


phicutfbb[12][10][9] = 0.0009;
phicutfbb[12][10][8] = 0.0018;
phicutfbb[12][10][7] = 0.0027;
phicutfbb[12][10][6] = 0.0027;
phicutfbb[12][10][5] = 0.0027;
phicutfbb[12][10][4] = 0.0027;
phicutfbb[12][10][3] = 0.0027;
phicutfbb[12][10][2] = 0.0027;
phicutfbb[12][10][1] = 0.005;
phicutfbb[12][10][0] = 0.005;

phicutfbb[12][9][8] = 0.0027;
phicutfbb[12][9][7] = 0.0027;
phicutfbb[12][9][6] = 0.0027;
phicutfbb[12][9][5] = 0.0027;
phicutfbb[12][9][4] = 0.0027;
phicutfbb[12][9][3] = 0.0027;
phicutfbb[12][9][2] = 0.0027;
phicutfbb[12][9][1] = 0.005;
phicutfbb[12][9][0] = 0.005;

phicutfbb[12][8][7] = 0.0009;
phicutfbb[12][8][6] = 0.0018;
phicutfbb[12][8][5] = 0.0027;
phicutfbb[12][8][4] = 0.0027;
phicutfbb[12][8][3] = 0.0027;
phicutfbb[12][8][2] = 0.0027;
phicutfbb[12][8][1] = 0.005;
phicutfbb[12][8][0] = 0.005;

phicutfbb[12][7][6] = 0.0009;
phicutfbb[12][7][5] = 0.0018;
phicutfbb[12][7][4] = 0.0027;
phicutfbb[12][7][3] = 0.0027;
phicutfbb[12][7][2] = 0.0027;
phicutfbb[12][7][1] = 0.005;
phicutfbb[12][7][0] = 0.005;

phicutfbb[12][6][5] = 0.0009;
phicutfbb[12][6][4] = 0.0018;
phicutfbb[12][6][3] = 0.0027;
phicutfbb[12][6][2] = 0.0027;
phicutfbb[12][6][1] = 0.005;
phicutfbb[12][6][0] = 0.005;

phicutfbb[12][5][4] = 0.0009;
phicutfbb[12][5][3] = 0.0018;
phicutfbb[12][5][2] = 0.0027;
phicutfbb[12][5][1] = 0.005;
phicutfbb[12][5][0] = 0.005;

phicutfbb[12][4][3] = 0.0027;
phicutfbb[12][4][2] = 0.0027;
phicutfbb[12][4][1] = 0.005;
phicutfbb[12][4][0] = 0.005;

phicutfbb[12][3][2] = 0.005;
phicutfbb[12][3][1] = 0.005;
phicutfbb[12][3][0] = 0.005;

phicutfbb[12][2][1] = 0.005;
phicutfbb[12][2][0] = 0.005;

phicutfbb[12][1][0] = 0.0035;

// +++++++++++ Last layer = 11

phicutfbb[11][12][11] = 0.1;
phicutfbb[11][11][10] = 0.1;
phicutfbb[11][11][9] = 0.1;
phicutfbb[11][11][8] = 0.1;
phicutfbb[11][11][7] = 0.1;
phicutfbb[11][11][6] = 0.1;
phicutfbb[11][11][5] = 0.1;
phicutfbb[11][11][4] = 0.1;
phicutfbb[11][11][3] = 0.1;
phicutfbb[11][11][2] = 0.1;
phicutfbb[11][11][1] = 0.1;
phicutfbb[11][11][0] = 0.1;


phicutfbb[11][11][10] = 0.1;
phicutfbb[11][11][9] = 0.1;
phicutfbb[11][11][8] = 0.1;
phicutfbb[11][11][7] = 0.1;
phicutfbb[11][11][6] = 0.1;
phicutfbb[11][11][5] = 0.1;
phicutfbb[11][11][4] = 0.1;
phicutfbb[11][11][3] = 0.1;
phicutfbb[11][11][2] = 0.1;
phicutfbb[11][11][1] = 0.1;
phicutfbb[11][11][0] = 0.1;


phicutfbb[11][10][9] = 0.0009;
phicutfbb[11][10][8] = 0.0018;
phicutfbb[11][10][7] = 0.0027;
phicutfbb[11][10][6] = 0.0027;
phicutfbb[11][10][5] = 0.0027;
phicutfbb[11][10][4] = 0.0027;
phicutfbb[11][10][3] = 0.0027;
phicutfbb[11][10][2] = 0.0027;
phicutfbb[11][10][1] = 0.005;
phicutfbb[11][10][0] = 0.005;

phicutfbb[11][9][8] = 0.0027;
phicutfbb[11][9][7] = 0.0027;
phicutfbb[11][9][6] = 0.0027;
phicutfbb[11][9][5] = 0.0027;
phicutfbb[11][9][4] = 0.0027;
phicutfbb[11][9][3] = 0.0027;
phicutfbb[11][9][2] = 0.0027;
phicutfbb[11][9][1] = 0.005;
phicutfbb[11][9][0] = 0.005;

phicutfbb[11][8][7] = 0.0009;
phicutfbb[11][8][6] = 0.0018;
phicutfbb[11][8][5] = 0.0027;
phicutfbb[11][8][4] = 0.0027;
phicutfbb[11][8][3] = 0.0027;
phicutfbb[11][8][2] = 0.0027;
phicutfbb[11][8][1] = 0.005;
phicutfbb[11][8][0] = 0.005;

phicutfbb[11][7][6] = 0.0009;
phicutfbb[11][7][5] = 0.0018;
phicutfbb[11][7][4] = 0.0027;
phicutfbb[11][7][3] = 0.0027;
phicutfbb[11][7][2] = 0.0027;
phicutfbb[11][7][1] = 0.005;
phicutfbb[11][7][0] = 0.005;

phicutfbb[11][6][5] = 0.0009;
phicutfbb[11][6][4] = 0.0018;
phicutfbb[11][6][3] = 0.0027;
phicutfbb[11][6][2] = 0.0027;
phicutfbb[11][6][1] = 0.005;
phicutfbb[11][6][0] = 0.005;

phicutfbb[11][5][4] = 0.0009;
phicutfbb[11][5][3] = 0.0018;
phicutfbb[11][5][2] = 0.0027;
phicutfbb[11][5][1] = 0.005;
phicutfbb[11][5][0] = 0.005;

phicutfbb[11][4][3] = 0.0027;
phicutfbb[11][4][2] = 0.0027;
phicutfbb[11][4][1] = 0.005;
phicutfbb[11][4][0] = 0.005;

phicutfbb[11][3][2] = 0.005;
phicutfbb[11][3][1] = 0.005;
phicutfbb[11][3][0] = 0.005;

phicutfbb[11][2][1] = 0.005;
phicutfbb[11][2][0] = 0.005;

phicutfbb[11][1][0] = 0.0035;

// +++++++++++ Last layer = 10

phicutfbb[10][12][11] = 0.1;
phicutfbb[10][12][10] = 0.1;
phicutfbb[10][12][9] = 0.1;
phicutfbb[10][12][8] = 0.1;
phicutfbb[10][12][7] = 0.1;
phicutfbb[10][12][6] = 0.1;
phicutfbb[10][12][5] = 0.1;
phicutfbb[10][12][4] = 0.1;
phicutfbb[10][12][3] = 0.1;
phicutfbb[10][12][2] = 0.1;
phicutfbb[10][12][1] = 0.1;
phicutfbb[10][12][0] = 0.1;

phicutfbb[10][11][10] = 0.1;
phicutfbb[10][11][9] = 0.1;
phicutfbb[10][11][8] = 0.1;
phicutfbb[10][11][7] = 0.1;
phicutfbb[10][11][6] = 0.1;
phicutfbb[10][11][5] = 0.1;
phicutfbb[10][11][4] = 0.1;
phicutfbb[10][11][3] = 0.1;
phicutfbb[10][11][2] = 0.1;
phicutfbb[10][11][1] = 0.1;
phicutfbb[10][11][0] = 0.1;


phicutfbb[10][10][9] = 0.1;
phicutfbb[10][10][8] = 0.1;
phicutfbb[10][10][7] = 0.1;
phicutfbb[10][10][6] = 0.1;
phicutfbb[10][10][5] = 0.1;
phicutfbb[10][10][4] = 0.1;
phicutfbb[10][10][3] = 0.1;
phicutfbb[10][10][2] = 0.1;
phicutfbb[10][10][1] = 0.1;
phicutfbb[10][10][0] = 0.1;

phicutfbb[10][9][8] = 0.0009;
phicutfbb[10][9][7] = 0.0018;
phicutfbb[10][9][6] = 0.0027;
phicutfbb[10][9][5] = 0.0027;
phicutfbb[10][9][4] = 0.0027;
phicutfbb[10][9][3] = 0.0027;
phicutfbb[10][9][2] = 0.0027;
phicutfbb[10][9][1] = 0.005;
phicutfbb[10][9][0] = 0.005;

phicutfbb[10][8][7] = 0.0009;
phicutfbb[10][8][6] = 0.0018;
phicutfbb[10][8][5] = 0.0027;
phicutfbb[10][8][4] = 0.0027;
phicutfbb[10][8][3] = 0.0027;
phicutfbb[10][8][2] = 0.0027;
phicutfbb[10][8][1] = 0.005;
phicutfbb[10][8][0] = 0.005;

phicutfbb[10][7][6] = 0.0009;
phicutfbb[10][7][5] = 0.0018;
phicutfbb[10][7][4] = 0.0027;
phicutfbb[10][7][3] = 0.0027;
phicutfbb[10][7][2] = 0.0027;
phicutfbb[10][7][1] = 0.005;
phicutfbb[10][7][0] = 0.005;

phicutfbb[10][6][5] = 0.0009;
phicutfbb[10][6][4] = 0.0018;
phicutfbb[10][6][3] = 0.0027;
phicutfbb[10][6][2] = 0.0027;
phicutfbb[10][6][1] = 0.005;
phicutfbb[10][6][0] = 0.005;

phicutfbb[10][5][4] = 0.0009;
phicutfbb[10][5][3] = 0.0018;
phicutfbb[10][5][2] = 0.0027;
phicutfbb[10][5][1] = 0.005;
phicutfbb[10][5][0] = 0.005;

phicutfbb[10][4][3] = 0.0027;
phicutfbb[10][4][2] = 0.0027;
phicutfbb[10][4][1] = 0.005;
phicutfbb[10][4][0] = 0.005;

phicutfbb[10][3][2] = 0.005;
phicutfbb[10][3][1] = 0.005;
phicutfbb[10][3][0] = 0.005;

phicutfbb[10][2][1] = 0.005;
phicutfbb[10][2][0] = 0.005;

phicutfbb[10][1][0] = 0.003;

// +++++++++++ Last layer = 9


phicutfbb[9][12][11] = 0.1;
phicutfbb[9][12][10] = 0.1;
phicutfbb[9][12][9] = 0.1;
phicutfbb[9][12][8] = 0.1;
phicutfbb[9][12][7] = 0.1;
phicutfbb[9][12][6] = 0.1;
phicutfbb[9][12][5] = 0.1;
phicutfbb[9][12][4] = 0.1;
phicutfbb[9][12][3] = 0.1;
phicutfbb[9][12][2] = 0.1;
phicutfbb[9][12][1] = 0.1;
phicutfbb[9][12][0] = 0.1;

phicutfbb[9][11][10] = 0.1;
phicutfbb[9][11][9] = 0.1;
phicutfbb[9][11][8] = 0.1;
phicutfbb[9][11][7] = 0.1;
phicutfbb[9][11][6] = 0.1;
phicutfbb[9][11][5] = 0.1;
phicutfbb[9][11][4] = 0.1;
phicutfbb[9][11][3] = 0.1;
phicutfbb[9][11][2] = 0.1;
phicutfbb[9][11][1] = 0.1;
phicutfbb[9][11][0] = 0.1;


phicutfbb[9][10][9] = 0.1;
phicutfbb[9][10][8] = 0.1;
phicutfbb[9][10][7] = 0.1;
phicutfbb[9][10][6] = 0.1;
phicutfbb[9][10][5] = 0.1;
phicutfbb[9][10][4] = 0.1;
phicutfbb[9][10][3] = 0.1;
phicutfbb[9][10][2] = 0.1;
phicutfbb[9][10][1] = 0.1;
phicutfbb[9][10][0] = 0.1;

phicutfbb[9][9][8] = 0.1;
phicutfbb[9][9][7] = 0.1;
phicutfbb[9][9][6] = 0.1;
phicutfbb[9][9][5] = 0.1;
phicutfbb[9][9][4] = 0.1;
phicutfbb[9][9][3] = 0.1;
phicutfbb[9][9][2] = 0.1;
phicutfbb[9][9][1] = 0.1;
phicutfbb[9][9][0] = 0.1;

phicutfbb[9][8][7] = 0.0009;
phicutfbb[9][8][6] = 0.0018;
phicutfbb[9][8][5] = 0.0027;
phicutfbb[9][8][4] = 0.0027;
phicutfbb[9][8][3] = 0.0027;
phicutfbb[9][8][2] = 0.0027;
phicutfbb[9][8][1] = 0.005;
phicutfbb[9][8][0] = 0.005;

phicutfbb[9][7][6] = 0.0009;
phicutfbb[9][7][5] = 0.0018;
phicutfbb[9][7][4] = 0.0027;
phicutfbb[9][7][3] = 0.0027;
phicutfbb[9][7][2] = 0.0027;
phicutfbb[9][7][1] = 0.005;
phicutfbb[9][7][0] = 0.005;

phicutfbb[9][6][5] = 0.0009;
phicutfbb[9][6][4] = 0.0018;
phicutfbb[9][6][3] = 0.0027;
phicutfbb[9][6][2] = 0.0027;
phicutfbb[9][6][1] = 0.005;
phicutfbb[9][6][0] = 0.005;

phicutfbb[9][5][4] = 0.0009;
phicutfbb[9][5][3] = 0.0018;
phicutfbb[9][5][2] = 0.0027;
phicutfbb[9][5][1] = 0.005;
phicutfbb[9][5][0] = 0.005;

phicutfbb[9][4][3] = 0.0027;
phicutfbb[9][4][2] = 0.0027;
phicutfbb[9][4][1] = 0.005;
phicutfbb[9][4][0] = 0.005;

phicutfbb[9][3][2] = 0.005;
phicutfbb[9][3][1] = 0.005;
phicutfbb[9][3][0] = 0.005;

phicutfbb[9][2][1] = 0.005;
phicutfbb[9][2][0] = 0.005;

phicutfbb[9][1][0] = 0.003;

// +++++++++++ Last layer = 8

phicutfbb[8][12][11] = 0.1;
phicutfbb[8][12][10] = 0.1;
phicutfbb[8][12][9] = 0.1;
phicutfbb[8][12][8] = 0.1;
phicutfbb[8][12][7] = 0.1;
phicutfbb[8][12][6] = 0.1;
phicutfbb[8][12][5] = 0.1;
phicutfbb[8][12][4] = 0.1;
phicutfbb[8][12][3] = 0.1;
phicutfbb[8][12][2] = 0.1;
phicutfbb[8][12][1] = 0.1;
phicutfbb[8][12][0] = 0.1;

phicutfbb[8][11][10] = 0.1;
phicutfbb[8][11][9] = 0.1;
phicutfbb[8][11][8] = 0.1;
phicutfbb[8][11][7] = 0.1;
phicutfbb[8][11][6] = 0.1;
phicutfbb[8][11][5] = 0.1;
phicutfbb[8][11][4] = 0.1;
phicutfbb[8][11][3] = 0.1;
phicutfbb[8][11][2] = 0.1;
phicutfbb[8][11][1] = 0.1;
phicutfbb[8][11][0] = 0.1;


phicutfbb[8][10][9] = 0.1;
phicutfbb[8][10][8] = 0.1;
phicutfbb[8][10][7] = 0.1;
phicutfbb[8][10][6] = 0.1;
phicutfbb[8][10][5] = 0.1;
phicutfbb[8][10][4] = 0.1;
phicutfbb[8][10][3] = 0.1;
phicutfbb[8][10][2] = 0.1;
phicutfbb[8][10][1] = 0.1;
phicutfbb[8][10][0] = 0.1;

phicutfbb[8][9][8] = 0.1;
phicutfbb[8][9][7] = 0.1;
phicutfbb[8][9][6] = 0.1;
phicutfbb[8][9][5] = 0.1;
phicutfbb[8][9][4] = 0.1;
phicutfbb[8][9][3] = 0.1;
phicutfbb[8][9][2] = 0.1;
phicutfbb[8][9][1] = 0.1;
phicutfbb[8][9][0] = 0.1;


phicutfbb[8][8][7] = 0.1;
phicutfbb[8][8][6] = 0.1;
phicutfbb[8][8][5] = 0.1;
phicutfbb[8][8][4] = 0.1;
phicutfbb[8][8][3] = 0.1;
phicutfbb[8][8][2] = 0.1;
phicutfbb[8][8][1] = 0.1;
phicutfbb[8][8][0] = 0.1;

phicutfbb[8][7][6] = 0.0008;
phicutfbb[8][7][5] = 0.0018;
phicutfbb[8][7][4] = 0.0027;
phicutfbb[8][7][3] = 0.0027;
phicutfbb[8][7][2] = 0.0027;
phicutfbb[8][7][1] = 0.005;
phicutfbb[8][7][0] = 0.005;

phicutfbb[8][6][5] = 0.0008;
phicutfbb[8][6][4] = 0.0018;
phicutfbb[8][6][3] = 0.0027;
phicutfbb[8][6][2] = 0.0027;
phicutfbb[8][6][1] = 0.005;
phicutfbb[8][6][0] = 0.005;

phicutfbb[8][5][4] = 0.0008;
phicutfbb[8][5][3] = 0.0018;
phicutfbb[8][5][2] = 0.0027;
phicutfbb[8][5][1] = 0.005;
phicutfbb[8][5][0] = 0.005;

phicutfbb[8][4][3] = 0.0027;
phicutfbb[8][4][2] = 0.0027;
phicutfbb[8][4][1] = 0.005;
phicutfbb[8][4][0] = 0.005;

phicutfbb[8][3][2] = 0.005;
phicutfbb[8][3][1] = 0.005;
phicutfbb[8][3][0] = 0.005;

phicutfbb[8][2][1] = 0.005;
phicutfbb[8][2][0] = 0.005;

phicutfbb[8][1][0] = 0.003;

// +++++++++++ Last layer = 7

phicutfbb[7][12][11] = 0.1;
phicutfbb[7][12][10] = 0.1;
phicutfbb[7][12][9] = 0.1;
phicutfbb[7][12][8] = 0.1;
phicutfbb[7][12][7] = 0.1;
phicutfbb[7][12][6] = 0.1;
phicutfbb[7][12][5] = 0.1;
phicutfbb[7][12][4] = 0.1;
phicutfbb[7][12][3] = 0.1;
phicutfbb[7][12][2] = 0.1;
phicutfbb[7][12][1] = 0.1;
phicutfbb[7][12][0] = 0.1;

phicutfbb[7][11][10] = 0.1;
phicutfbb[7][11][9] = 0.1;
phicutfbb[7][11][8] = 0.1;
phicutfbb[7][11][7] = 0.1;
phicutfbb[7][11][6] = 0.1;
phicutfbb[7][11][5] = 0.1;
phicutfbb[7][11][4] = 0.1;
phicutfbb[7][11][3] = 0.1;
phicutfbb[7][11][2] = 0.1;
phicutfbb[7][11][1] = 0.1;
phicutfbb[7][11][0] = 0.1;


phicutfbb[7][10][9] = 0.1;
phicutfbb[7][10][8] = 0.1;
phicutfbb[7][10][7] = 0.1;
phicutfbb[7][10][6] = 0.1;
phicutfbb[7][10][5] = 0.1;
phicutfbb[7][10][4] = 0.1;
phicutfbb[7][10][3] = 0.1;
phicutfbb[7][10][2] = 0.1;
phicutfbb[7][10][1] = 0.1;
phicutfbb[7][10][0] = 0.1;

phicutfbb[7][9][8] = 0.1;
phicutfbb[7][9][7] = 0.1;
phicutfbb[7][9][6] = 0.1;
phicutfbb[7][9][5] = 0.1;
phicutfbb[7][9][4] = 0.1;
phicutfbb[7][9][3] = 0.1;
phicutfbb[7][9][2] = 0.1;
phicutfbb[7][9][1] = 0.1;
phicutfbb[7][9][0] = 0.1;


phicutfbb[7][8][7] = 0.1;
phicutfbb[7][8][6] = 0.1;
phicutfbb[7][8][5] = 0.1;
phicutfbb[7][8][4] = 0.1;
phicutfbb[7][8][3] = 0.1;
phicutfbb[7][8][2] = 0.1;
phicutfbb[7][8][1] = 0.1;
phicutfbb[7][8][0] = 0.1;


phicutfbb[7][7][6] = 0.1;
phicutfbb[7][7][5] = 0.1;
phicutfbb[7][7][4] = 0.1;
phicutfbb[7][7][3] = 0.1;
phicutfbb[7][7][2] = 0.1;
phicutfbb[7][7][1] = 0.1;
phicutfbb[7][7][0] = 0.1;

phicutfbb[7][6][5] = 0.0008;
phicutfbb[7][6][4] = 0.0018;
phicutfbb[7][6][3] = 0.0027;
phicutfbb[7][6][2] = 0.0027;
phicutfbb[7][6][1] = 0.005;
phicutfbb[7][6][0] = 0.005;

phicutfbb[7][5][4] = 0.0008;
phicutfbb[7][5][3] = 0.0018;
phicutfbb[7][5][2] = 0.0027;
phicutfbb[7][5][1] = 0.005;
phicutfbb[7][5][0] = 0.005;

phicutfbb[7][4][3] = 0.0027;
phicutfbb[7][4][2] = 0.0027;
phicutfbb[7][4][1] = 0.005;
phicutfbb[7][4][0] = 0.005;

phicutfbb[7][3][2] = 0.005;
phicutfbb[7][3][1] = 0.005;
phicutfbb[7][3][0] = 0.005;

phicutfbb[7][2][1] = 0.005;
phicutfbb[7][2][0] = 0.005;

phicutfbb[7][1][0] = 0.003;

// +++++++++++ Last layer = 6

phicutfbb[6][12][11] = 0.1;
phicutfbb[6][12][10] = 0.1;
phicutfbb[6][12][9] = 0.1;
phicutfbb[6][12][8] = 0.1;
phicutfbb[6][12][7] = 0.1;
phicutfbb[6][12][6] = 0.1;
phicutfbb[6][12][5] = 0.1;
phicutfbb[6][12][4] = 0.1;
phicutfbb[6][12][3] = 0.1;
phicutfbb[6][12][2] = 0.1;
phicutfbb[6][12][1] = 0.1;
phicutfbb[6][12][0] = 0.1;

phicutfbb[6][11][10] = 0.1;
phicutfbb[6][11][9] = 0.1;
phicutfbb[6][11][8] = 0.1;
phicutfbb[6][11][7] = 0.1;
phicutfbb[6][11][6] = 0.1;
phicutfbb[6][11][5] = 0.1;
phicutfbb[6][11][4] = 0.1;
phicutfbb[6][11][3] = 0.1;
phicutfbb[6][11][2] = 0.1;
phicutfbb[6][11][1] = 0.1;
phicutfbb[6][11][0] = 0.1;


phicutfbb[6][10][9] = 0.1;
phicutfbb[6][10][8] = 0.1;
phicutfbb[6][10][7] = 0.1;
phicutfbb[6][10][6] = 0.1;
phicutfbb[6][10][5] = 0.1;
phicutfbb[6][10][4] = 0.1;
phicutfbb[6][10][3] = 0.1;
phicutfbb[6][10][2] = 0.1;
phicutfbb[6][10][1] = 0.1;
phicutfbb[6][10][0] = 0.1;

phicutfbb[6][9][8] = 0.1;
phicutfbb[6][9][7] = 0.1;
phicutfbb[6][9][6] = 0.1;
phicutfbb[6][9][5] = 0.1;
phicutfbb[6][9][4] = 0.1;
phicutfbb[6][9][3] = 0.1;
phicutfbb[6][9][2] = 0.1;
phicutfbb[6][9][1] = 0.1;
phicutfbb[6][9][0] = 0.1;


phicutfbb[6][8][7] = 0.1;
phicutfbb[6][8][6] = 0.1;
phicutfbb[6][8][5] = 0.1;
phicutfbb[6][8][4] = 0.1;
phicutfbb[6][8][3] = 0.1;
phicutfbb[6][8][2] = 0.1;
phicutfbb[6][8][1] = 0.1;
phicutfbb[6][8][0] = 0.1;


phicutfbb[6][7][6] = 0.1;
phicutfbb[6][7][5] = 0.1;
phicutfbb[6][7][4] = 0.1;
phicutfbb[6][7][3] = 0.1;
phicutfbb[6][7][2] = 0.1;
phicutfbb[6][7][1] = 0.1;
phicutfbb[6][7][0] = 0.1;

phicutfbb[6][6][5] = 0.1;
phicutfbb[6][6][4] = 0.1;
phicutfbb[6][6][3] = 0.1;
phicutfbb[6][6][2] = 0.1;
phicutfbb[6][6][1] = 0.1;
phicutfbb[6][6][0] = 0.1;

phicutfbb[6][5][4] = 0.0008;
phicutfbb[6][5][3] = 0.0018;
phicutfbb[6][5][2] = 0.0027;
phicutfbb[6][5][1] = 0.005;
phicutfbb[6][5][0] = 0.005;

phicutfbb[6][4][3] = 0.0027;
phicutfbb[6][4][2] = 0.0027;
phicutfbb[6][4][1] = 0.005;
phicutfbb[6][4][0] = 0.005;

phicutfbb[6][3][2] = 0.005;
phicutfbb[6][3][1] = 0.005;
phicutfbb[6][3][0] = 0.005;

phicutfbb[6][2][1] = 0.005;
phicutfbb[6][2][0] = 0.005;

phicutfbb[6][1][0] = 0.003;

// +++++++++++ Last layer = 5

phicutfbb[5][12][11] = 0.1;
phicutfbb[5][12][10] = 0.1;
phicutfbb[5][12][9] = 0.1;
phicutfbb[5][12][8] = 0.1;
phicutfbb[5][12][7] = 0.1;
phicutfbb[5][12][6] = 0.1;
phicutfbb[5][12][5] = 0.1;
phicutfbb[5][12][4] = 0.1;
phicutfbb[5][12][3] = 0.1;
phicutfbb[5][12][2] = 0.1;
phicutfbb[5][12][1] = 0.1;
phicutfbb[5][12][0] = 0.1;

phicutfbb[5][11][10] = 0.1;
phicutfbb[5][11][9] = 0.1;
phicutfbb[5][11][8] = 0.1;
phicutfbb[5][11][7] = 0.1;
phicutfbb[5][11][6] = 0.1;
phicutfbb[5][11][5] = 0.1;
phicutfbb[5][11][4] = 0.1;
phicutfbb[5][11][3] = 0.1;
phicutfbb[5][11][2] = 0.1;
phicutfbb[5][11][1] = 0.1;
phicutfbb[5][11][0] = 0.1;


phicutfbb[5][10][9] = 0.1;
phicutfbb[5][10][8] = 0.1;
phicutfbb[5][10][7] = 0.1;
phicutfbb[5][10][6] = 0.1;
phicutfbb[5][10][5] = 0.1;
phicutfbb[5][10][4] = 0.1;
phicutfbb[5][10][3] = 0.1;
phicutfbb[5][10][2] = 0.1;
phicutfbb[5][10][1] = 0.1;
phicutfbb[5][10][0] = 0.1;

phicutfbb[5][9][8] = 0.1;
phicutfbb[5][9][7] = 0.1;
phicutfbb[5][9][6] = 0.1;
phicutfbb[5][9][5] = 0.1;
phicutfbb[5][9][4] = 0.1;
phicutfbb[5][9][3] = 0.1;
phicutfbb[5][9][2] = 0.1;
phicutfbb[5][9][1] = 0.1;
phicutfbb[5][9][0] = 0.1;


phicutfbb[5][8][7] = 0.1;
phicutfbb[5][8][6] = 0.1;
phicutfbb[5][8][5] = 0.1;
phicutfbb[5][8][4] = 0.1;
phicutfbb[5][8][3] = 0.1;
phicutfbb[5][8][2] = 0.1;
phicutfbb[5][8][1] = 0.1;
phicutfbb[5][8][0] = 0.1;


phicutfbb[5][7][6] = 0.1;
phicutfbb[5][7][5] = 0.1;
phicutfbb[5][7][4] = 0.1;
phicutfbb[5][7][3] = 0.1;
phicutfbb[5][7][2] = 0.1;
phicutfbb[5][7][1] = 0.1;
phicutfbb[5][7][0] = 0.1;

phicutfbb[5][6][5] = 0.1;
phicutfbb[5][6][4] = 0.1;
phicutfbb[5][6][3] = 0.1;
phicutfbb[5][6][2] = 0.1;
phicutfbb[5][6][1] = 0.1;
phicutfbb[5][6][0] = 0.1;


phicutfbb[5][5][4] = 0.1;
phicutfbb[5][5][3] = 0.1;
phicutfbb[5][5][2] = 0.1;
phicutfbb[5][5][1] = 0.1;
phicutfbb[5][5][0] = 0.1;

phicutfbb[5][4][3] = 0.0027;
phicutfbb[5][4][2] = 0.0027;
phicutfbb[5][4][1] = 0.005;
phicutfbb[5][4][0] = 0.005;

phicutfbb[5][3][2] = 0.005;
phicutfbb[5][3][1] = 0.005;
phicutfbb[5][3][0] = 0.005;

phicutfbb[5][2][1] = 0.005;
phicutfbb[5][2][0] = 0.005;

phicutfbb[5][1][0] = 0.003;

// phiwinfbb
// ++++++++++++Last layer = 13

phiwinfbb[13][12][11] = 0.0009;
phiwinfbb[13][12][10] = 0.0018;
phiwinfbb[13][12][9] = 0.0027;
phiwinfbb[13][12][8] = 0.0027;
phiwinfbb[13][12][7] = 0.0027;
phiwinfbb[13][12][6] = 0.0027;
phiwinfbb[13][12][5] = 0.0027;
phiwinfbb[13][12][4] = 0.0027;
phiwinfbb[13][12][3] = 0.0027;
phiwinfbb[13][12][2] = 0.0027;
phiwinfbb[13][12][1] = 0.005;
phiwinfbb[13][12][0] = 0.005;

phiwinfbb[13][11][10] = 0.0009;
phiwinfbb[13][11][9] = 0.0018;
phiwinfbb[13][11][8] = 0.0027;
phiwinfbb[13][11][7] = 0.0027;
phiwinfbb[13][11][6] = 0.0027;
phiwinfbb[13][11][5] = 0.0027;
phiwinfbb[13][11][4] = 0.0027;
phiwinfbb[13][11][3] = 0.0027;
phiwinfbb[13][11][2] = 0.0027;
phiwinfbb[13][11][1] = 0.005;
phiwinfbb[13][11][0] = 0.005;

phiwinfbb[13][10][9] = 0.0009;
phiwinfbb[13][10][8] = 0.0018;
phiwinfbb[13][10][7] = 0.0027;
phiwinfbb[13][10][6] = 0.0027;
phiwinfbb[13][10][5] = 0.0027;
phiwinfbb[13][10][4] = 0.0027;
phiwinfbb[13][10][3] = 0.0027;
phiwinfbb[13][10][2] = 0.0027;
phiwinfbb[13][10][1] = 0.005;
phiwinfbb[13][10][0] = 0.005;

phiwinfbb[13][9][8] = 0.0027;
phiwinfbb[13][9][7] = 0.0027;
phiwinfbb[13][9][6] = 0.0027;
phiwinfbb[13][9][5] = 0.0027;
phiwinfbb[13][9][4] = 0.0027;
phiwinfbb[13][9][3] = 0.0027;
phiwinfbb[13][9][2] = 0.0027;
phiwinfbb[13][9][1] = 0.005;
phiwinfbb[13][9][0] = 0.005;

phiwinfbb[13][8][7] = 0.0009;
phiwinfbb[13][8][6] = 0.0018;
phiwinfbb[13][8][5] = 0.0027;
phiwinfbb[13][8][4] = 0.0027;
phiwinfbb[13][8][3] = 0.0027;
phiwinfbb[13][8][2] = 0.0027;
phiwinfbb[13][8][1] = 0.005;
phiwinfbb[13][8][0] = 0.005;

phiwinfbb[13][7][6] = 0.0009;
phiwinfbb[13][7][5] = 0.0018;
phiwinfbb[13][7][4] = 0.0027;
phiwinfbb[13][7][3] = 0.0027;
phiwinfbb[13][7][2] = 0.0027;
phiwinfbb[13][7][1] = 0.005;
phiwinfbb[13][7][0] = 0.005;

phiwinfbb[13][6][5] = 0.0009;
phiwinfbb[13][6][4] = 0.0018;
phiwinfbb[13][6][3] = 0.0027;
phiwinfbb[13][6][2] = 0.0027;
phiwinfbb[13][6][1] = 0.005;
phiwinfbb[13][6][0] = 0.005;

phiwinfbb[13][5][4] = 0.0009;
phiwinfbb[13][5][3] = 0.0018;
phiwinfbb[13][5][2] = 0.0027;
phiwinfbb[13][5][1] = 0.005;
phiwinfbb[13][5][0] = 0.005;

phiwinfbb[13][4][3] = 0.0027;
phiwinfbb[13][4][2] = 0.0027;
phiwinfbb[13][4][1] = 0.005;
phiwinfbb[13][4][0] = 0.005;

phiwinfbb[13][3][2] = 0.005;
phiwinfbb[13][3][1] = 0.005;
phiwinfbb[13][3][0] = 0.005;

phiwinfbb[13][2][1] = 0.005;
phiwinfbb[13][2][0] = 0.003;

phiwinfbb[13][1][0] = 0.0035;

// +++++++++++ Last layer = 12

phiwinfbb[12][12][11] = 0.1;
phiwinfbb[12][12][10] = 0.1;
phiwinfbb[12][12][9] = 0.1;
phiwinfbb[12][12][8] = 0.1;
phiwinfbb[12][12][7] = 0.1;
phiwinfbb[12][12][6] = 0.1;
phiwinfbb[12][12][5] = 0.1;
phiwinfbb[12][12][4] = 0.1;
phiwinfbb[12][12][3] = 0.1;
phiwinfbb[12][12][2] = 0.1;
phiwinfbb[12][12][1] = 0.1;
phiwinfbb[12][12][0] = 0.1;

phiwinfbb[12][11][10] = 0.0009;
phiwinfbb[12][11][9] = 0.0018;
phiwinfbb[12][11][8] = 0.0027;
phiwinfbb[12][11][7] = 0.0027;
phiwinfbb[12][11][6] = 0.0027;
phiwinfbb[12][11][5] = 0.0027;
phiwinfbb[12][11][4] = 0.0027;
phiwinfbb[12][11][3] = 0.0027;
phiwinfbb[12][11][2] = 0.0027;
phiwinfbb[12][11][1] = 0.005;
phiwinfbb[12][11][0] = 0.005;


phiwinfbb[12][10][9] = 0.0009;
phiwinfbb[12][10][8] = 0.0018;
phiwinfbb[12][10][7] = 0.0027;
phiwinfbb[12][10][6] = 0.0027;
phiwinfbb[12][10][5] = 0.0027;
phiwinfbb[12][10][4] = 0.0027;
phiwinfbb[12][10][3] = 0.0027;
phiwinfbb[12][10][2] = 0.0027;
phiwinfbb[12][10][1] = 0.005;
phiwinfbb[12][10][0] = 0.005;

phiwinfbb[12][9][8] = 0.0027;
phiwinfbb[12][9][7] = 0.0027;
phiwinfbb[12][9][6] = 0.0027;
phiwinfbb[12][9][5] = 0.0027;
phiwinfbb[12][9][4] = 0.0027;
phiwinfbb[12][9][3] = 0.0027;
phiwinfbb[12][9][2] = 0.0027;
phiwinfbb[12][9][1] = 0.005;
phiwinfbb[12][9][0] = 0.005;

phiwinfbb[12][8][7] = 0.0009;
phiwinfbb[12][8][6] = 0.0018;
phiwinfbb[12][8][5] = 0.0027;
phiwinfbb[12][8][4] = 0.0027;
phiwinfbb[12][8][3] = 0.0027;
phiwinfbb[12][8][2] = 0.0027;
phiwinfbb[12][8][1] = 0.005;
phiwinfbb[12][8][0] = 0.005;

phiwinfbb[12][7][6] = 0.0009;
phiwinfbb[12][7][5] = 0.0018;
phiwinfbb[12][7][4] = 0.0027;
phiwinfbb[12][7][3] = 0.0027;
phiwinfbb[12][7][2] = 0.0027;
phiwinfbb[12][7][1] = 0.005;
phiwinfbb[12][7][0] = 0.005;

phiwinfbb[12][6][5] = 0.0009;
phiwinfbb[12][6][4] = 0.0018;
phiwinfbb[12][6][3] = 0.0027;
phiwinfbb[12][6][2] = 0.0027;
phiwinfbb[12][6][1] = 0.005;
phiwinfbb[12][6][0] = 0.005;

phiwinfbb[12][5][4] = 0.0009;
phiwinfbb[12][5][3] = 0.0018;
phiwinfbb[12][5][2] = 0.0027;
phiwinfbb[12][5][1] = 0.005;
phiwinfbb[12][5][0] = 0.005;

phiwinfbb[12][4][3] = 0.0027;
phiwinfbb[12][4][2] = 0.0027;
phiwinfbb[12][4][1] = 0.005;
phiwinfbb[12][4][0] = 0.005;

phiwinfbb[12][3][2] = 0.005;
phiwinfbb[12][3][1] = 0.005;
phiwinfbb[12][3][0] = 0.005;

phiwinfbb[12][2][1] = 0.005;
phiwinfbb[12][2][0] = 0.005;

phiwinfbb[12][1][0] = 0.0035;

// +++++++++++ Last layer = 11

phiwinfbb[11][12][11] = 0.1;
phiwinfbb[11][11][10] = 0.1;
phiwinfbb[11][11][9] = 0.1;
phiwinfbb[11][11][8] = 0.1;
phiwinfbb[11][11][7] = 0.1;
phiwinfbb[11][11][6] = 0.1;
phiwinfbb[11][11][5] = 0.1;
phiwinfbb[11][11][4] = 0.1;
phiwinfbb[11][11][3] = 0.1;
phiwinfbb[11][11][2] = 0.1;
phiwinfbb[11][11][1] = 0.1;
phiwinfbb[11][11][0] = 0.1;


phiwinfbb[11][11][10] = 0.1;
phiwinfbb[11][11][9] = 0.1;
phiwinfbb[11][11][8] = 0.1;
phiwinfbb[11][11][7] = 0.1;
phiwinfbb[11][11][6] = 0.1;
phiwinfbb[11][11][5] = 0.1;
phiwinfbb[11][11][4] = 0.1;
phiwinfbb[11][11][3] = 0.1;
phiwinfbb[11][11][2] = 0.1;
phiwinfbb[11][11][1] = 0.1;
phiwinfbb[11][11][0] = 0.1;


phiwinfbb[11][10][9] = 0.0009;
phiwinfbb[11][10][8] = 0.0018;
phiwinfbb[11][10][7] = 0.0027;
phiwinfbb[11][10][6] = 0.0027;
phiwinfbb[11][10][5] = 0.0027;
phiwinfbb[11][10][4] = 0.0027;
phiwinfbb[11][10][3] = 0.0027;
phiwinfbb[11][10][2] = 0.0027;
phiwinfbb[11][10][1] = 0.005;
phiwinfbb[11][10][0] = 0.005;

phiwinfbb[11][9][8] = 0.0027;
phiwinfbb[11][9][7] = 0.0027;
phiwinfbb[11][9][6] = 0.0027;
phiwinfbb[11][9][5] = 0.0027;
phiwinfbb[11][9][4] = 0.0027;
phiwinfbb[11][9][3] = 0.0027;
phiwinfbb[11][9][2] = 0.0027;
phiwinfbb[11][9][1] = 0.005;
phiwinfbb[11][9][0] = 0.005;

phiwinfbb[11][8][7] = 0.0009;
phiwinfbb[11][8][6] = 0.0018;
phiwinfbb[11][8][5] = 0.0027;
phiwinfbb[11][8][4] = 0.0027;
phiwinfbb[11][8][3] = 0.0027;
phiwinfbb[11][8][2] = 0.0027;
phiwinfbb[11][8][1] = 0.005;
phiwinfbb[11][8][0] = 0.005;

phiwinfbb[11][7][6] = 0.0009;
phiwinfbb[11][7][5] = 0.0018;
phiwinfbb[11][7][4] = 0.0027;
phiwinfbb[11][7][3] = 0.0027;
phiwinfbb[11][7][2] = 0.0027;
phiwinfbb[11][7][1] = 0.005;
phiwinfbb[11][7][0] = 0.005;

phiwinfbb[11][6][5] = 0.0009;
phiwinfbb[11][6][4] = 0.0018;
phiwinfbb[11][6][3] = 0.0027;
phiwinfbb[11][6][2] = 0.0027;
phiwinfbb[11][6][1] = 0.005;
phiwinfbb[11][6][0] = 0.005;

phiwinfbb[11][5][4] = 0.0009;
phiwinfbb[11][5][3] = 0.0018;
phiwinfbb[11][5][2] = 0.0027;
phiwinfbb[11][5][1] = 0.005;
phiwinfbb[11][5][0] = 0.005;

phiwinfbb[11][4][3] = 0.0027;
phiwinfbb[11][4][2] = 0.0027;
phiwinfbb[11][4][1] = 0.005;
phiwinfbb[11][4][0] = 0.005;

phiwinfbb[11][3][2] = 0.005;
phiwinfbb[11][3][1] = 0.005;
phiwinfbb[11][3][0] = 0.005;

phiwinfbb[11][2][1] = 0.005;
phiwinfbb[11][2][0] = 0.005;

phiwinfbb[11][1][0] = 0.0035;

// +++++++++++ Last layer = 10

phiwinfbb[10][12][11] = 0.1;
phiwinfbb[10][12][10] = 0.1;
phiwinfbb[10][12][9] = 0.1;
phiwinfbb[10][12][8] = 0.1;
phiwinfbb[10][12][7] = 0.1;
phiwinfbb[10][12][6] = 0.1;
phiwinfbb[10][12][5] = 0.1;
phiwinfbb[10][12][4] = 0.1;
phiwinfbb[10][12][3] = 0.1;
phiwinfbb[10][12][2] = 0.1;
phiwinfbb[10][12][1] = 0.1;
phiwinfbb[10][12][0] = 0.1;

phiwinfbb[10][11][10] = 0.1;
phiwinfbb[10][11][9] = 0.1;
phiwinfbb[10][11][8] = 0.1;
phiwinfbb[10][11][7] = 0.1;
phiwinfbb[10][11][6] = 0.1;
phiwinfbb[10][11][5] = 0.1;
phiwinfbb[10][11][4] = 0.1;
phiwinfbb[10][11][3] = 0.1;
phiwinfbb[10][11][2] = 0.1;
phiwinfbb[10][11][1] = 0.1;
phiwinfbb[10][11][0] = 0.1;


phiwinfbb[10][10][9] = 0.1;
phiwinfbb[10][10][8] = 0.1;
phiwinfbb[10][10][7] = 0.1;
phiwinfbb[10][10][6] = 0.1;
phiwinfbb[10][10][5] = 0.1;
phiwinfbb[10][10][4] = 0.1;
phiwinfbb[10][10][3] = 0.1;
phiwinfbb[10][10][2] = 0.1;
phiwinfbb[10][10][1] = 0.1;
phiwinfbb[10][10][0] = 0.1;

phiwinfbb[10][9][8] = 0.0009;
phiwinfbb[10][9][7] = 0.0018;
phiwinfbb[10][9][6] = 0.0027;
phiwinfbb[10][9][5] = 0.0027;
phiwinfbb[10][9][4] = 0.0027;
phiwinfbb[10][9][3] = 0.0027;
phiwinfbb[10][9][2] = 0.0027;
phiwinfbb[10][9][1] = 0.005;
phiwinfbb[10][9][0] = 0.005;

phiwinfbb[10][8][7] = 0.0009;
phiwinfbb[10][8][6] = 0.0018;
phiwinfbb[10][8][5] = 0.0027;
phiwinfbb[10][8][4] = 0.0027;
phiwinfbb[10][8][3] = 0.0027;
phiwinfbb[10][8][2] = 0.0027;
phiwinfbb[10][8][1] = 0.005;
phiwinfbb[10][8][0] = 0.005;

phiwinfbb[10][7][6] = 0.0009;
phiwinfbb[10][7][5] = 0.0018;
phiwinfbb[10][7][4] = 0.0027;
phiwinfbb[10][7][3] = 0.0027;
phiwinfbb[10][7][2] = 0.0027;
phiwinfbb[10][7][1] = 0.005;
phiwinfbb[10][7][0] = 0.005;

phiwinfbb[10][6][5] = 0.0009;
phiwinfbb[10][6][4] = 0.0018;
phiwinfbb[10][6][3] = 0.0027;
phiwinfbb[10][6][2] = 0.0027;
phiwinfbb[10][6][1] = 0.005;
phiwinfbb[10][6][0] = 0.005;

phiwinfbb[10][5][4] = 0.0009;
phiwinfbb[10][5][3] = 0.0018;
phiwinfbb[10][5][2] = 0.0027;
phiwinfbb[10][5][1] = 0.005;
phiwinfbb[10][5][0] = 0.005;

phiwinfbb[10][4][3] = 0.0027;
phiwinfbb[10][4][2] = 0.0027;
phiwinfbb[10][4][1] = 0.005;
phiwinfbb[10][4][0] = 0.005;

phiwinfbb[10][3][2] = 0.005;
phiwinfbb[10][3][1] = 0.005;
phiwinfbb[10][3][0] = 0.005;

phiwinfbb[10][2][1] = 0.005;
phiwinfbb[10][2][0] = 0.005;

phiwinfbb[10][1][0] = 0.0035;

// +++++++++++ Last layer = 9


phiwinfbb[9][12][11] = 0.1;
phiwinfbb[9][12][10] = 0.1;
phiwinfbb[9][12][9] = 0.1;
phiwinfbb[9][12][8] = 0.1;
phiwinfbb[9][12][7] = 0.1;
phiwinfbb[9][12][6] = 0.1;
phiwinfbb[9][12][5] = 0.1;
phiwinfbb[9][12][4] = 0.1;
phiwinfbb[9][12][3] = 0.1;
phiwinfbb[9][12][2] = 0.1;
phiwinfbb[9][12][1] = 0.1;
phiwinfbb[9][12][0] = 0.1;

phiwinfbb[9][11][10] = 0.1;
phiwinfbb[9][11][9] = 0.1;
phiwinfbb[9][11][8] = 0.1;
phiwinfbb[9][11][7] = 0.1;
phiwinfbb[9][11][6] = 0.1;
phiwinfbb[9][11][5] = 0.1;
phiwinfbb[9][11][4] = 0.1;
phiwinfbb[9][11][3] = 0.1;
phiwinfbb[9][11][2] = 0.1;
phiwinfbb[9][11][1] = 0.1;
phiwinfbb[9][11][0] = 0.1;


phiwinfbb[9][10][9] = 0.1;
phiwinfbb[9][10][8] = 0.1;
phiwinfbb[9][10][7] = 0.1;
phiwinfbb[9][10][6] = 0.1;
phiwinfbb[9][10][5] = 0.1;
phiwinfbb[9][10][4] = 0.1;
phiwinfbb[9][10][3] = 0.1;
phiwinfbb[9][10][2] = 0.1;
phiwinfbb[9][10][1] = 0.1;
phiwinfbb[9][10][0] = 0.1;

phiwinfbb[9][9][8] = 0.1;
phiwinfbb[9][9][7] = 0.1;
phiwinfbb[9][9][6] = 0.1;
phiwinfbb[9][9][5] = 0.1;
phiwinfbb[9][9][4] = 0.1;
phiwinfbb[9][9][3] = 0.1;
phiwinfbb[9][9][2] = 0.1;
phiwinfbb[9][9][1] = 0.1;
phiwinfbb[9][9][0] = 0.1;

phiwinfbb[9][8][7] = 0.0009;
phiwinfbb[9][8][6] = 0.0018;
phiwinfbb[9][8][5] = 0.0027;
phiwinfbb[9][8][4] = 0.0027;
phiwinfbb[9][8][3] = 0.0027;
phiwinfbb[9][8][2] = 0.0027;
phiwinfbb[9][8][1] = 0.005;
phiwinfbb[9][8][0] = 0.005;

phiwinfbb[9][7][6] = 0.0009;
phiwinfbb[9][7][5] = 0.0018;
phiwinfbb[9][7][4] = 0.0027;
phiwinfbb[9][7][3] = 0.0027;
phiwinfbb[9][7][2] = 0.0027;
phiwinfbb[9][7][1] = 0.005;
phiwinfbb[9][7][0] = 0.005;

phiwinfbb[9][6][5] = 0.0009;
phiwinfbb[9][6][4] = 0.0018;
phiwinfbb[9][6][3] = 0.0027;
phiwinfbb[9][6][2] = 0.0027;
phiwinfbb[9][6][1] = 0.005;
phiwinfbb[9][6][0] = 0.005;

phiwinfbb[9][5][4] = 0.0009;
phiwinfbb[9][5][3] = 0.0018;
phiwinfbb[9][5][2] = 0.0027;
phiwinfbb[9][5][1] = 0.005;
phiwinfbb[9][5][0] = 0.005;

phiwinfbb[9][4][3] = 0.0027;
phiwinfbb[9][4][2] = 0.0027;
phiwinfbb[9][4][1] = 0.005;
phiwinfbb[9][4][0] = 0.005;

phiwinfbb[9][3][2] = 0.005;
phiwinfbb[9][3][1] = 0.005;
phiwinfbb[9][3][0] = 0.005;

phiwinfbb[9][2][1] = 0.005;
phiwinfbb[9][2][0] = 0.005;

phiwinfbb[9][1][0] = 0.0035;

// +++++++++++ Last layer = 8

phiwinfbb[8][12][11] = 0.1;
phiwinfbb[8][12][10] = 0.1;
phiwinfbb[8][12][9] = 0.1;
phiwinfbb[8][12][8] = 0.1;
phiwinfbb[8][12][7] = 0.1;
phiwinfbb[8][12][6] = 0.1;
phiwinfbb[8][12][5] = 0.1;
phiwinfbb[8][12][4] = 0.1;
phiwinfbb[8][12][3] = 0.1;
phiwinfbb[8][12][2] = 0.1;
phiwinfbb[8][12][1] = 0.1;
phiwinfbb[8][12][0] = 0.1;

phiwinfbb[8][11][10] = 0.1;
phiwinfbb[8][11][9] = 0.1;
phiwinfbb[8][11][8] = 0.1;
phiwinfbb[8][11][7] = 0.1;
phiwinfbb[8][11][6] = 0.1;
phiwinfbb[8][11][5] = 0.1;
phiwinfbb[8][11][4] = 0.1;
phiwinfbb[8][11][3] = 0.1;
phiwinfbb[8][11][2] = 0.1;
phiwinfbb[8][11][1] = 0.1;
phiwinfbb[8][11][0] = 0.1;


phiwinfbb[8][10][9] = 0.1;
phiwinfbb[8][10][8] = 0.1;
phiwinfbb[8][10][7] = 0.1;
phiwinfbb[8][10][6] = 0.1;
phiwinfbb[8][10][5] = 0.1;
phiwinfbb[8][10][4] = 0.1;
phiwinfbb[8][10][3] = 0.1;
phiwinfbb[8][10][2] = 0.1;
phiwinfbb[8][10][1] = 0.1;
phiwinfbb[8][10][0] = 0.1;

phiwinfbb[8][9][8] = 0.1;
phiwinfbb[8][9][7] = 0.1;
phiwinfbb[8][9][6] = 0.1;
phiwinfbb[8][9][5] = 0.1;
phiwinfbb[8][9][4] = 0.1;
phiwinfbb[8][9][3] = 0.1;
phiwinfbb[8][9][2] = 0.1;
phiwinfbb[8][9][1] = 0.1;
phiwinfbb[8][9][0] = 0.1;


phiwinfbb[8][8][7] = 0.1;
phiwinfbb[8][8][6] = 0.1;
phiwinfbb[8][8][5] = 0.1;
phiwinfbb[8][8][4] = 0.1;
phiwinfbb[8][8][3] = 0.1;
phiwinfbb[8][8][2] = 0.1;
phiwinfbb[8][8][1] = 0.1;
phiwinfbb[8][8][0] = 0.1;

phiwinfbb[8][7][6] = 0.0008;
phiwinfbb[8][7][5] = 0.0018;
phiwinfbb[8][7][4] = 0.0027;
phiwinfbb[8][7][3] = 0.0027;
phiwinfbb[8][7][2] = 0.0027;
phiwinfbb[8][7][1] = 0.005;
phiwinfbb[8][7][0] = 0.005;

phiwinfbb[8][6][5] = 0.0008;
phiwinfbb[8][6][4] = 0.0018;
phiwinfbb[8][6][3] = 0.0027;
phiwinfbb[8][6][2] = 0.0027;
phiwinfbb[8][6][1] = 0.005;
phiwinfbb[8][6][0] = 0.005;

phiwinfbb[8][5][4] = 0.0008;
phiwinfbb[8][5][3] = 0.0018;
phiwinfbb[8][5][2] = 0.0027;
phiwinfbb[8][5][1] = 0.005;
phiwinfbb[8][5][0] = 0.005;

phiwinfbb[8][4][3] = 0.0027;
phiwinfbb[8][4][2] = 0.0027;
phiwinfbb[8][4][1] = 0.005;
phiwinfbb[8][4][0] = 0.005;

phiwinfbb[8][3][2] = 0.005;
phiwinfbb[8][3][1] = 0.005;
phiwinfbb[8][3][0] = 0.005;

phiwinfbb[8][2][1] = 0.005;
phiwinfbb[8][2][0] = 0.005;

phiwinfbb[8][1][0] = 0.0035;

// +++++++++++ Last layer = 7

phiwinfbb[7][12][11] = 0.1;
phiwinfbb[7][12][10] = 0.1;
phiwinfbb[7][12][9] = 0.1;
phiwinfbb[7][12][8] = 0.1;
phiwinfbb[7][12][7] = 0.1;
phiwinfbb[7][12][6] = 0.1;
phiwinfbb[7][12][5] = 0.1;
phiwinfbb[7][12][4] = 0.1;
phiwinfbb[7][12][3] = 0.1;
phiwinfbb[7][12][2] = 0.1;
phiwinfbb[7][12][1] = 0.1;
phiwinfbb[7][12][0] = 0.1;

phiwinfbb[7][11][10] = 0.1;
phiwinfbb[7][11][9] = 0.1;
phiwinfbb[7][11][8] = 0.1;
phiwinfbb[7][11][7] = 0.1;
phiwinfbb[7][11][6] = 0.1;
phiwinfbb[7][11][5] = 0.1;
phiwinfbb[7][11][4] = 0.1;
phiwinfbb[7][11][3] = 0.1;
phiwinfbb[7][11][2] = 0.1;
phiwinfbb[7][11][1] = 0.1;
phiwinfbb[7][11][0] = 0.1;


phiwinfbb[7][10][9] = 0.1;
phiwinfbb[7][10][8] = 0.1;
phiwinfbb[7][10][7] = 0.1;
phiwinfbb[7][10][6] = 0.1;
phiwinfbb[7][10][5] = 0.1;
phiwinfbb[7][10][4] = 0.1;
phiwinfbb[7][10][3] = 0.1;
phiwinfbb[7][10][2] = 0.1;
phiwinfbb[7][10][1] = 0.1;
phiwinfbb[7][10][0] = 0.1;

phiwinfbb[7][9][8] = 0.1;
phiwinfbb[7][9][7] = 0.1;
phiwinfbb[7][9][6] = 0.1;
phiwinfbb[7][9][5] = 0.1;
phiwinfbb[7][9][4] = 0.1;
phiwinfbb[7][9][3] = 0.1;
phiwinfbb[7][9][2] = 0.1;
phiwinfbb[7][9][1] = 0.1;
phiwinfbb[7][9][0] = 0.1;


phiwinfbb[7][8][7] = 0.1;
phiwinfbb[7][8][6] = 0.1;
phiwinfbb[7][8][5] = 0.1;
phiwinfbb[7][8][4] = 0.1;
phiwinfbb[7][8][3] = 0.1;
phiwinfbb[7][8][2] = 0.1;
phiwinfbb[7][8][1] = 0.1;
phiwinfbb[7][8][0] = 0.1;


phiwinfbb[7][7][6] = 0.1;
phiwinfbb[7][7][5] = 0.1;
phiwinfbb[7][7][4] = 0.1;
phiwinfbb[7][7][3] = 0.1;
phiwinfbb[7][7][2] = 0.1;
phiwinfbb[7][7][1] = 0.1;
phiwinfbb[7][7][0] = 0.1;

phiwinfbb[7][6][5] = 0.0008;
phiwinfbb[7][6][4] = 0.0018;
phiwinfbb[7][6][3] = 0.0027;
phiwinfbb[7][6][2] = 0.0027;
phiwinfbb[7][6][1] = 0.005;
phiwinfbb[7][6][0] = 0.005;

phiwinfbb[7][5][4] = 0.0008;
phiwinfbb[7][5][3] = 0.0018;
phiwinfbb[7][5][2] = 0.0027;
phiwinfbb[7][5][1] = 0.005;
phiwinfbb[7][5][0] = 0.005;

phiwinfbb[7][4][3] = 0.0027;
phiwinfbb[7][4][2] = 0.0027;
phiwinfbb[7][4][1] = 0.005;
phiwinfbb[7][4][0] = 0.005;

phiwinfbb[7][3][2] = 0.005;
phiwinfbb[7][3][1] = 0.005;
phiwinfbb[7][3][0] = 0.005;

phiwinfbb[7][2][1] = 0.005;
phiwinfbb[7][2][0] = 0.005;

phiwinfbb[7][1][0] = 0.0035;

// +++++++++++ Last layer = 6

phiwinfbb[6][12][11] = 0.1;
phiwinfbb[6][12][10] = 0.1;
phiwinfbb[6][12][9] = 0.1;
phiwinfbb[6][12][8] = 0.1;
phiwinfbb[6][12][7] = 0.1;
phiwinfbb[6][12][6] = 0.1;
phiwinfbb[6][12][5] = 0.1;
phiwinfbb[6][12][4] = 0.1;
phiwinfbb[6][12][3] = 0.1;
phiwinfbb[6][12][2] = 0.1;
phiwinfbb[6][12][1] = 0.1;
phiwinfbb[6][12][0] = 0.1;

phiwinfbb[6][11][10] = 0.1;
phiwinfbb[6][11][9] = 0.1;
phiwinfbb[6][11][8] = 0.1;
phiwinfbb[6][11][7] = 0.1;
phiwinfbb[6][11][6] = 0.1;
phiwinfbb[6][11][5] = 0.1;
phiwinfbb[6][11][4] = 0.1;
phiwinfbb[6][11][3] = 0.1;
phiwinfbb[6][11][2] = 0.1;
phiwinfbb[6][11][1] = 0.1;
phiwinfbb[6][11][0] = 0.1;


phiwinfbb[6][10][9] = 0.1;
phiwinfbb[6][10][8] = 0.1;
phiwinfbb[6][10][7] = 0.1;
phiwinfbb[6][10][6] = 0.1;
phiwinfbb[6][10][5] = 0.1;
phiwinfbb[6][10][4] = 0.1;
phiwinfbb[6][10][3] = 0.1;
phiwinfbb[6][10][2] = 0.1;
phiwinfbb[6][10][1] = 0.1;
phiwinfbb[6][10][0] = 0.1;

phiwinfbb[6][9][8] = 0.1;
phiwinfbb[6][9][7] = 0.1;
phiwinfbb[6][9][6] = 0.1;
phiwinfbb[6][9][5] = 0.1;
phiwinfbb[6][9][4] = 0.1;
phiwinfbb[6][9][3] = 0.1;
phiwinfbb[6][9][2] = 0.1;
phiwinfbb[6][9][1] = 0.1;
phiwinfbb[6][9][0] = 0.1;


phiwinfbb[6][8][7] = 0.1;
phiwinfbb[6][8][6] = 0.1;
phiwinfbb[6][8][5] = 0.1;
phiwinfbb[6][8][4] = 0.1;
phiwinfbb[6][8][3] = 0.1;
phiwinfbb[6][8][2] = 0.1;
phiwinfbb[6][8][1] = 0.1;
phiwinfbb[6][8][0] = 0.1;


phiwinfbb[6][7][6] = 0.1;
phiwinfbb[6][7][5] = 0.1;
phiwinfbb[6][7][4] = 0.1;
phiwinfbb[6][7][3] = 0.1;
phiwinfbb[6][7][2] = 0.1;
phiwinfbb[6][7][1] = 0.1;
phiwinfbb[6][7][0] = 0.1;

phiwinfbb[6][6][5] = 0.1;
phiwinfbb[6][6][4] = 0.1;
phiwinfbb[6][6][3] = 0.1;
phiwinfbb[6][6][2] = 0.1;
phiwinfbb[6][6][1] = 0.1;
phiwinfbb[6][6][0] = 0.1;

phiwinfbb[6][5][4] = 0.0008;
phiwinfbb[6][5][3] = 0.0018;
phiwinfbb[6][5][2] = 0.0027;
phiwinfbb[6][5][1] = 0.005;
phiwinfbb[6][5][0] = 0.005;

phiwinfbb[6][4][3] = 0.0027;
phiwinfbb[6][4][2] = 0.0027;
phiwinfbb[6][4][1] = 0.005;
phiwinfbb[6][4][0] = 0.005;

phiwinfbb[6][3][2] = 0.005;
phiwinfbb[6][3][1] = 0.005;
phiwinfbb[6][3][0] = 0.005;

phiwinfbb[6][2][1] = 0.005;
phiwinfbb[6][2][0] = 0.005;

phiwinfbb[6][1][0] = 0.0035;

// +++++++++++ Last layer = 5

phiwinfbb[5][12][11] = 0.1;
phiwinfbb[5][12][10] = 0.1;
phiwinfbb[5][12][9] = 0.1;
phiwinfbb[5][12][8] = 0.1;
phiwinfbb[5][12][7] = 0.1;
phiwinfbb[5][12][6] = 0.1;
phiwinfbb[5][12][5] = 0.1;
phiwinfbb[5][12][4] = 0.1;
phiwinfbb[5][12][3] = 0.1;
phiwinfbb[5][12][2] = 0.1;
phiwinfbb[5][12][1] = 0.1;
phiwinfbb[5][12][0] = 0.1;

phiwinfbb[5][11][10] = 0.1;
phiwinfbb[5][11][9] = 0.1;
phiwinfbb[5][11][8] = 0.1;
phiwinfbb[5][11][7] = 0.1;
phiwinfbb[5][11][6] = 0.1;
phiwinfbb[5][11][5] = 0.1;
phiwinfbb[5][11][4] = 0.1;
phiwinfbb[5][11][3] = 0.1;
phiwinfbb[5][11][2] = 0.1;
phiwinfbb[5][11][1] = 0.1;
phiwinfbb[5][11][0] = 0.1;


phiwinfbb[5][10][9] = 0.1;
phiwinfbb[5][10][8] = 0.1;
phiwinfbb[5][10][7] = 0.1;
phiwinfbb[5][10][6] = 0.1;
phiwinfbb[5][10][5] = 0.1;
phiwinfbb[5][10][4] = 0.1;
phiwinfbb[5][10][3] = 0.1;
phiwinfbb[5][10][2] = 0.1;
phiwinfbb[5][10][1] = 0.1;
phiwinfbb[5][10][0] = 0.1;

phiwinfbb[5][9][8] = 0.1;
phiwinfbb[5][9][7] = 0.1;
phiwinfbb[5][9][6] = 0.1;
phiwinfbb[5][9][5] = 0.1;
phiwinfbb[5][9][4] = 0.1;
phiwinfbb[5][9][3] = 0.1;
phiwinfbb[5][9][2] = 0.1;
phiwinfbb[5][9][1] = 0.1;
phiwinfbb[5][9][0] = 0.1;


phiwinfbb[5][8][7] = 0.1;
phiwinfbb[5][8][6] = 0.1;
phiwinfbb[5][8][5] = 0.1;
phiwinfbb[5][8][4] = 0.1;
phiwinfbb[5][8][3] = 0.1;
phiwinfbb[5][8][2] = 0.1;
phiwinfbb[5][8][1] = 0.1;
phiwinfbb[5][8][0] = 0.1;


phiwinfbb[5][7][6] = 0.1;
phiwinfbb[5][7][5] = 0.1;
phiwinfbb[5][7][4] = 0.1;
phiwinfbb[5][7][3] = 0.1;
phiwinfbb[5][7][2] = 0.1;
phiwinfbb[5][7][1] = 0.1;
phiwinfbb[5][7][0] = 0.1;

phiwinfbb[5][6][5] = 0.1;
phiwinfbb[5][6][4] = 0.1;
phiwinfbb[5][6][3] = 0.1;
phiwinfbb[5][6][2] = 0.1;
phiwinfbb[5][6][1] = 0.1;
phiwinfbb[5][6][0] = 0.1;


phiwinfbb[5][5][4] = 0.1;
phiwinfbb[5][5][3] = 0.1;
phiwinfbb[5][5][2] = 0.1;
phiwinfbb[5][5][1] = 0.1;
phiwinfbb[5][5][0] = 0.1;

phiwinfbb[5][4][3] = 0.0027;
phiwinfbb[5][4][2] = 0.0027;
phiwinfbb[5][4][1] = 0.005;
phiwinfbb[5][4][0] = 0.005;

phiwinfbb[5][3][2] = 0.005;
phiwinfbb[5][3][1] = 0.005;
phiwinfbb[5][3][0] = 0.005;

phiwinfbb[5][2][1] = 0.005;
phiwinfbb[5][2][0] = 0.005;

phiwinfbb[5][1][0] = 0.0035;


// Forward roads, windows, phiwinfrw

// ++++++++++++Last layer = 13

phiwinfrw[13][13][12] = 0.19;
phiwinfrw[13][13][11] = 0.15;
phiwinfrw[13][13][10] = 0.15;
phiwinfrw[13][13][9] = 0.13;
phiwinfrw[13][13][8] = 0.1;
phiwinfrw[13][13][7] = 0.1;
phiwinfrw[13][13][6] = 0.1;
phiwinfrw[13][13][5] = 0.1;
phiwinfrw[13][13][4] = 0.1;
phiwinfrw[13][13][3] = 0.1;
phiwinfrw[13][13][2] = 0.1;
phiwinfrw[13][13][1] = 0.1;
phiwinfrw[13][13][0] = 0.1;

//phiwinfrw[13][12][11] = 0.0011; in ORCA
phiwinfrw[13][12][11] = 0.01;
phiwinfrw[13][12][10] = 0.0018;
phiwinfrw[13][12][9] = 0.0027;
phiwinfrw[13][12][8] = 0.0027;
phiwinfrw[13][12][7] = 0.0027;
phiwinfrw[13][12][6] = 0.0027;
phiwinfrw[13][12][5] = 0.0027;
phiwinfrw[13][12][4] = 0.0027;
phiwinfrw[13][12][3] = 0.0027;
phiwinfrw[13][12][2] = 0.0027;
phiwinfrw[13][12][1] = 0.005;
phiwinfrw[13][12][0] = 0.005;

phiwinfrw[13][11][10] = 0.0009;
phiwinfrw[13][11][9] = 0.0018;
phiwinfrw[13][11][8] = 0.0027;
phiwinfrw[13][11][7] = 0.0027;
phiwinfrw[13][11][6] = 0.0027;
phiwinfrw[13][11][5] = 0.0027;
phiwinfrw[13][11][4] = 0.0027;
phiwinfrw[13][11][3] = 0.0027;
phiwinfrw[13][11][2] = 0.0027;
phiwinfrw[13][11][1] = 0.005;
phiwinfrw[13][11][0] = 0.005;

phiwinfrw[13][10][9] = 0.002;
phiwinfrw[13][10][8] = 0.0018;
phiwinfrw[13][10][7] = 0.0027;
phiwinfrw[13][10][6] = 0.0027;
phiwinfrw[13][10][5] = 0.0027;
phiwinfrw[13][10][4] = 0.0027;
phiwinfrw[13][10][3] = 0.0027;
phiwinfrw[13][10][2] = 0.0027;
phiwinfrw[13][10][1] = 0.005;
phiwinfrw[13][10][0] = 0.005;

phiwinfrw[13][9][8] = 0.00095;
phiwinfrw[13][9][7] = 0.0018;
phiwinfrw[13][9][6] = 0.0027;
phiwinfrw[13][9][5] = 0.0027;
phiwinfrw[13][9][4] = 0.0027;
phiwinfrw[13][9][3] = 0.0027;
phiwinfrw[13][9][2] = 0.0027;
phiwinfrw[13][9][1] = 0.005;
phiwinfrw[13][9][0] = 0.005;

phiwinfrw[13][8][7] = 0.00095;
phiwinfrw[13][8][6] = 0.0018;
phiwinfrw[13][8][5] = 0.0027;
phiwinfrw[13][8][4] = 0.0027;
phiwinfrw[13][8][3] = 0.0027;
phiwinfrw[13][8][2] = 0.0027;
phiwinfrw[13][8][1] = 0.005;
phiwinfrw[13][8][0] = 0.005;

phiwinfrw[13][7][6] = 0.0009;
phiwinfrw[13][7][5] = 0.0018;
phiwinfrw[13][7][4] = 0.0027;
phiwinfrw[13][7][3] = 0.0027;
phiwinfrw[13][7][2] = 0.0027;
phiwinfrw[13][7][1] = 0.005;
phiwinfrw[13][7][0] = 0.005;

phiwinfrw[13][6][5] = 0.0018;
phiwinfrw[13][6][4] = 0.0018;
phiwinfrw[13][6][3] = 0.0027;
phiwinfrw[13][6][2] = 0.0027;
phiwinfrw[13][6][1] = 0.005;
phiwinfrw[13][6][0] = 0.005;

phiwinfrw[13][5][4] = 0.0009;
phiwinfrw[13][5][3] = 0.0018;
phiwinfrw[13][5][2] = 0.0027;
phiwinfrw[13][5][1] = 0.005;
phiwinfrw[13][5][0] = 0.005;

phiwinfrw[13][4][3] = 0.0027;
phiwinfrw[13][4][2] = 0.0027;
phiwinfrw[13][4][1] = 0.005;
phiwinfrw[13][4][0] = 0.005;

phiwinfrw[13][3][2] = 0.005;
phiwinfrw[13][3][1] = 0.005;
phiwinfrw[13][3][0] = 0.005;

phiwinfrw[13][2][1] = 0.005;
phiwinfrw[13][2][0] = 0.003;

phiwinfrw[13][1][0] = 0.0035;

// +++++++++++ Last layer = 12

phiwinfrw[12][12][11] = 0.13;
phiwinfrw[12][12][10] = 0.13;
phiwinfrw[12][12][9] = 0.13;
phiwinfrw[12][12][8] = 0.1;
phiwinfrw[12][12][7] = 0.1;
phiwinfrw[12][12][6] = 0.1;
phiwinfrw[12][12][5] = 0.1;
phiwinfrw[12][12][4] = 0.1;
phiwinfrw[12][12][3] = 0.1;
phiwinfrw[12][12][2] = 0.1;
phiwinfrw[12][12][1] = 0.1;
phiwinfrw[12][12][0] = 0.1;

phiwinfrw[12][11][10] = 0.0014;
phiwinfrw[12][11][9] = 0.0018;
phiwinfrw[12][11][8] = 0.0027;
phiwinfrw[12][11][7] = 0.0027;
phiwinfrw[12][11][6] = 0.0027;
phiwinfrw[12][11][5] = 0.0027;
phiwinfrw[12][11][4] = 0.0027;
phiwinfrw[12][11][3] = 0.0027;
phiwinfrw[12][11][2] = 0.0027;
phiwinfrw[12][11][1] = 0.005;
phiwinfrw[12][11][0] = 0.005;


phiwinfrw[12][10][9] = 0.0009;
phiwinfrw[12][10][8] = 0.0018;
phiwinfrw[12][10][7] = 0.0027;
phiwinfrw[12][10][6] = 0.0027;
phiwinfrw[12][10][5] = 0.0027;
phiwinfrw[12][10][4] = 0.0027;
phiwinfrw[12][10][3] = 0.0027;
phiwinfrw[12][10][2] = 0.0027;
phiwinfrw[12][10][1] = 0.005;
phiwinfrw[12][10][0] = 0.005;

phiwinfrw[12][9][8] = 0.0009;
phiwinfrw[12][9][7] = 0.0018;
phiwinfrw[12][9][6] = 0.0027;
phiwinfrw[12][9][5] = 0.0027;
phiwinfrw[12][9][4] = 0.0027;
phiwinfrw[12][9][3] = 0.0027;
phiwinfrw[12][9][2] = 0.0027;
phiwinfrw[12][9][1] = 0.005;
phiwinfrw[12][9][0] = 0.005;

phiwinfrw[12][8][7] = 0.0009;
phiwinfrw[12][8][6] = 0.0018;
phiwinfrw[12][8][5] = 0.0027;
phiwinfrw[12][8][4] = 0.0027;
phiwinfrw[12][8][3] = 0.0027;
phiwinfrw[12][8][2] = 0.0027;
phiwinfrw[12][8][1] = 0.005;
phiwinfrw[12][8][0] = 0.005;

phiwinfrw[12][7][6] = 0.0009;
phiwinfrw[12][7][5] = 0.0018;
phiwinfrw[12][7][4] = 0.0027;
phiwinfrw[12][7][3] = 0.0027;
phiwinfrw[12][7][2] = 0.0027;
phiwinfrw[12][7][1] = 0.005;
phiwinfrw[12][7][0] = 0.005;

phiwinfrw[12][6][5] = 0.0015;
phiwinfrw[12][6][4] = 0.0018;
phiwinfrw[12][6][3] = 0.0027;
phiwinfrw[12][6][2] = 0.0027;
phiwinfrw[12][6][1] = 0.005;
phiwinfrw[12][6][0] = 0.005;

phiwinfrw[12][5][4] = 0.0009;
phiwinfrw[12][5][3] = 0.0018;
phiwinfrw[12][5][2] = 0.0027;
phiwinfrw[12][5][1] = 0.005;
phiwinfrw[12][5][0] = 0.005;

phiwinfrw[12][4][3] = 0.0027;
phiwinfrw[12][4][2] = 0.0027;
phiwinfrw[12][4][1] = 0.005;
phiwinfrw[12][4][0] = 0.005;

phiwinfrw[12][3][2] = 0.005;
phiwinfrw[12][3][1] = 0.005;
phiwinfrw[12][3][0] = 0.005;

phiwinfrw[12][2][1] = 0.005;
phiwinfrw[12][2][0] = 0.005;

phiwinfrw[12][1][0] = 0.0035;

// +++++++++++ Last layer = 11

phiwinfrw[11][11][10] = 0.13;
phiwinfrw[11][11][9] = 0.13;
phiwinfrw[11][11][8] = 0.13;
phiwinfrw[11][11][7] = 0.1;
phiwinfrw[11][11][6] = 0.1;
phiwinfrw[11][11][5] = 0.1;
phiwinfrw[11][11][4] = 0.1;
phiwinfrw[11][11][3] = 0.1;
phiwinfrw[11][11][2] = 0.1;
phiwinfrw[11][11][1] = 0.1;
phiwinfrw[11][11][0] = 0.1;


phiwinfrw[11][10][9] = 0.0009;
phiwinfrw[11][10][8] = 0.0018;
phiwinfrw[11][10][7] = 0.0027;
phiwinfrw[11][10][6] = 0.0027;
phiwinfrw[11][10][5] = 0.0027;
phiwinfrw[11][10][4] = 0.0027;
phiwinfrw[11][10][3] = 0.0027;
phiwinfrw[11][10][2] = 0.0027;
phiwinfrw[11][10][1] = 0.005;
phiwinfrw[11][10][0] = 0.005;

phiwinfrw[11][9][8] = 0.0009;
phiwinfrw[11][9][7] = 0.0018;
phiwinfrw[11][9][6] = 0.0027;
phiwinfrw[11][9][5] = 0.0027;
phiwinfrw[11][9][4] = 0.0027;
phiwinfrw[11][9][3] = 0.0027;
phiwinfrw[11][9][2] = 0.0027;
phiwinfrw[11][9][1] = 0.005;
phiwinfrw[11][9][0] = 0.005;

phiwinfrw[11][8][7] = 0.0014;
phiwinfrw[11][8][6] = 0.0018;
phiwinfrw[11][8][5] = 0.0027;
phiwinfrw[11][8][4] = 0.0027;
phiwinfrw[11][8][3] = 0.0027;
phiwinfrw[11][8][2] = 0.0027;
phiwinfrw[11][8][1] = 0.005;
phiwinfrw[11][8][0] = 0.005;

phiwinfrw[11][7][6] = 0.0009;
phiwinfrw[11][7][5] = 0.0018;
phiwinfrw[11][7][4] = 0.0027;
phiwinfrw[11][7][3] = 0.0027;
phiwinfrw[11][7][2] = 0.0027;
phiwinfrw[11][7][1] = 0.005;
phiwinfrw[11][7][0] = 0.005;

phiwinfrw[11][6][5] = 0.0009;
phiwinfrw[11][6][4] = 0.0018;
phiwinfrw[11][6][3] = 0.0027;
phiwinfrw[11][6][2] = 0.0027;
phiwinfrw[11][6][1] = 0.005;
phiwinfrw[11][6][0] = 0.005;

phiwinfrw[11][5][4] = 0.0009;
phiwinfrw[11][5][3] = 0.0018;
phiwinfrw[11][5][2] = 0.0027;
phiwinfrw[11][5][1] = 0.005;
phiwinfrw[11][5][0] = 0.005;

phiwinfrw[11][4][3] = 0.0027;
phiwinfrw[11][4][2] = 0.0027;
phiwinfrw[11][4][1] = 0.005;
phiwinfrw[11][4][0] = 0.005;

phiwinfrw[11][3][2] = 0.005;
phiwinfrw[11][3][1] = 0.005;
phiwinfrw[11][3][0] = 0.005;

phiwinfrw[11][2][1] = 0.005;
phiwinfrw[11][2][0] = 0.005;

phiwinfrw[11][1][0] = 0.0035;

// +++++++++++ Last layer = 10

phiwinfrw[10][10][9] = 0.13;
phiwinfrw[10][10][8] = 0.13;
phiwinfrw[10][10][7] = 0.13;
phiwinfrw[10][10][6] = 0.1;
phiwinfrw[10][10][5] = 0.1;
phiwinfrw[10][10][4] = 0.1;
phiwinfrw[10][10][3] = 0.1;
phiwinfrw[10][10][2] = 0.1;
phiwinfrw[10][10][1] = 0.1;
phiwinfrw[10][10][0] = 0.1;

phiwinfrw[10][9][8] = 0.0009;
phiwinfrw[10][9][7] = 0.0018;
phiwinfrw[10][9][6] = 0.0027;
phiwinfrw[10][9][5] = 0.0027;
phiwinfrw[10][9][4] = 0.0027;
phiwinfrw[10][9][3] = 0.0027;
phiwinfrw[10][9][2] = 0.0027;
phiwinfrw[10][9][1] = 0.005;
phiwinfrw[10][9][0] = 0.005;

phiwinfrw[10][8][7] = 0.0009;
phiwinfrw[10][8][6] = 0.0018;
phiwinfrw[10][8][5] = 0.0027;
phiwinfrw[10][8][4] = 0.0027;
phiwinfrw[10][8][3] = 0.0027;
phiwinfrw[10][8][2] = 0.0027;
phiwinfrw[10][8][1] = 0.005;
phiwinfrw[10][8][0] = 0.005;

phiwinfrw[10][7][6] = 0.0009;
phiwinfrw[10][7][5] = 0.0018;
phiwinfrw[10][7][4] = 0.0027;
phiwinfrw[10][7][3] = 0.0027;
phiwinfrw[10][7][2] = 0.0027;
phiwinfrw[10][7][1] = 0.005;
phiwinfrw[10][7][0] = 0.005;

phiwinfrw[10][6][5] = 0.0009;
phiwinfrw[10][6][4] = 0.0018;
phiwinfrw[10][6][3] = 0.0027;
phiwinfrw[10][6][2] = 0.0027;
phiwinfrw[10][6][1] = 0.005;
phiwinfrw[10][6][0] = 0.005;

phiwinfrw[10][5][4] = 0.0009;
phiwinfrw[10][5][3] = 0.0018;
phiwinfrw[10][5][2] = 0.0027;
phiwinfrw[10][5][1] = 0.005;
phiwinfrw[10][5][0] = 0.005;

phiwinfrw[10][4][3] = 0.0027;
phiwinfrw[10][4][2] = 0.0027;
phiwinfrw[10][4][1] = 0.005;
phiwinfrw[10][4][0] = 0.005;

phiwinfrw[10][3][2] = 0.005;
phiwinfrw[10][3][1] = 0.005;
phiwinfrw[10][3][0] = 0.005;

phiwinfrw[10][2][1] = 0.005;
phiwinfrw[10][2][0] = 0.005;

phiwinfrw[10][1][0] = 0.0035;

// +++++++++++ Last layer = 9


phiwinfrw[9][9][8] = 0.13;
phiwinfrw[9][9][7] = 0.13;
phiwinfrw[9][9][6] = 0.13;
phiwinfrw[9][9][5] = 0.1;
phiwinfrw[9][9][4] = 0.1;
phiwinfrw[9][9][3] = 0.1;
phiwinfrw[9][9][2] = 0.1;
phiwinfrw[9][9][1] = 0.1;
phiwinfrw[9][9][0] = 0.1;

phiwinfrw[9][8][7] = 0.0009;
phiwinfrw[9][8][6] = 0.0018;
phiwinfrw[9][8][5] = 0.0027;
phiwinfrw[9][8][4] = 0.0027;
phiwinfrw[9][8][3] = 0.0027;
phiwinfrw[9][8][2] = 0.0027;
phiwinfrw[9][8][1] = 0.005;
phiwinfrw[9][8][0] = 0.005;

phiwinfrw[9][7][6] = 0.0009;
phiwinfrw[9][7][5] = 0.0018;
phiwinfrw[9][7][4] = 0.0027;
phiwinfrw[9][7][3] = 0.0027;
phiwinfrw[9][7][2] = 0.0027;
phiwinfrw[9][7][1] = 0.005;
phiwinfrw[9][7][0] = 0.005;

phiwinfrw[9][6][5] = 0.0009;
phiwinfrw[9][6][4] = 0.0018;
phiwinfrw[9][6][3] = 0.0027;
phiwinfrw[9][6][2] = 0.0027;
phiwinfrw[9][6][1] = 0.005;
phiwinfrw[9][6][0] = 0.005;

phiwinfrw[9][5][4] = 0.0009;
phiwinfrw[9][5][3] = 0.0018;
phiwinfrw[9][5][2] = 0.0027;
phiwinfrw[9][5][1] = 0.005;
phiwinfrw[9][5][0] = 0.005;

phiwinfrw[9][4][3] = 0.0027;
phiwinfrw[9][4][2] = 0.0027;
phiwinfrw[9][4][1] = 0.005;
phiwinfrw[9][4][0] = 0.005;

phiwinfrw[9][3][2] = 0.005;
phiwinfrw[9][3][1] = 0.005;
phiwinfrw[9][3][0] = 0.005;

phiwinfrw[9][2][1] = 0.005;
phiwinfrw[9][2][0] = 0.005;

phiwinfrw[9][1][0] = 0.0035;

// +++++++++++ Last layer = 8


phiwinfrw[8][8][7] = 0.13;
phiwinfrw[8][8][6] = 0.13;
phiwinfrw[8][8][5] = 0.13;
phiwinfrw[8][8][4] = 0.1;
phiwinfrw[8][8][3] = 0.1;
phiwinfrw[8][8][2] = 0.1;
phiwinfrw[8][8][1] = 0.1;
phiwinfrw[8][8][0] = 0.1;

phiwinfrw[8][7][6] = 0.0008;
phiwinfrw[8][7][5] = 0.0018;
phiwinfrw[8][7][4] = 0.0027;
phiwinfrw[8][7][3] = 0.0027;
phiwinfrw[8][7][2] = 0.0027;
phiwinfrw[8][7][1] = 0.005;
phiwinfrw[8][7][0] = 0.005;

phiwinfrw[8][6][5] = 0.0008;
phiwinfrw[8][6][4] = 0.0018;
phiwinfrw[8][6][3] = 0.0027;
phiwinfrw[8][6][2] = 0.0027;
phiwinfrw[8][6][1] = 0.005;
phiwinfrw[8][6][0] = 0.005;

phiwinfrw[8][5][4] = 0.0008;
phiwinfrw[8][5][3] = 0.0018;
phiwinfrw[8][5][2] = 0.0027;
phiwinfrw[8][5][1] = 0.005;
phiwinfrw[8][5][0] = 0.005;

phiwinfrw[8][4][3] = 0.0027;
phiwinfrw[8][4][2] = 0.0027;
phiwinfrw[8][4][1] = 0.005;
phiwinfrw[8][4][0] = 0.005;

phiwinfrw[8][3][2] = 0.005;
phiwinfrw[8][3][1] = 0.005;
phiwinfrw[8][3][0] = 0.005;

phiwinfrw[8][2][1] = 0.005;
phiwinfrw[8][2][0] = 0.005;

phiwinfrw[8][1][0] = 0.0035;

// +++++++++++ Last layer = 7

phiwinfrw[7][7][6] = 0.13;
phiwinfrw[7][7][5] = 0.13;
phiwinfrw[7][7][4] = 0.13;
phiwinfrw[7][7][3] = 0.1;
phiwinfrw[7][7][2] = 0.1;
phiwinfrw[7][7][1] = 0.1;
phiwinfrw[7][7][0] = 0.1;

phiwinfrw[7][6][5] = 0.0008;
phiwinfrw[7][6][4] = 0.0018;
phiwinfrw[7][6][3] = 0.0027;
phiwinfrw[7][6][2] = 0.0027;
phiwinfrw[7][6][1] = 0.005;
phiwinfrw[7][6][0] = 0.005;

phiwinfrw[7][5][4] = 0.0008;
phiwinfrw[7][5][3] = 0.0018;
phiwinfrw[7][5][2] = 0.0027;
phiwinfrw[7][5][1] = 0.005;
phiwinfrw[7][5][0] = 0.005;

phiwinfrw[7][4][3] = 0.0027;
phiwinfrw[7][4][2] = 0.0027;
phiwinfrw[7][4][1] = 0.005;
phiwinfrw[7][4][0] = 0.005;

phiwinfrw[7][3][2] = 0.005;
phiwinfrw[7][3][1] = 0.005;
phiwinfrw[7][3][0] = 0.005;

phiwinfrw[7][2][1] = 0.005;
phiwinfrw[7][2][0] = 0.005;

phiwinfrw[7][1][0] = 0.0035;

// +++++++++++ Last layer = 6

phiwinfrw[6][6][5] = 0.13;
phiwinfrw[6][6][4] = 0.13;
phiwinfrw[6][6][3] = 0.13;
phiwinfrw[6][6][2] = 0.1;
phiwinfrw[6][6][1] = 0.1;
phiwinfrw[6][6][0] = 0.1;

phiwinfrw[6][5][4] = 0.0008;
phiwinfrw[6][5][3] = 0.0018;
phiwinfrw[6][5][2] = 0.0027;
phiwinfrw[6][5][1] = 0.005;
phiwinfrw[6][5][0] = 0.005;

phiwinfrw[6][4][3] = 0.0027;
phiwinfrw[6][4][2] = 0.0027;
phiwinfrw[6][4][1] = 0.005;
phiwinfrw[6][4][0] = 0.005;

phiwinfrw[6][3][2] = 0.005;
phiwinfrw[6][3][1] = 0.005;
phiwinfrw[6][3][0] = 0.005;

phiwinfrw[6][2][1] = 0.005;
phiwinfrw[6][2][0] = 0.005;

phiwinfrw[6][1][0] = 0.0035;

// +++++++++++ Last layer = 5

phiwinfrw[5][5][4] = 0.13;
phiwinfrw[5][5][3] = 0.13;
phiwinfrw[5][5][2] = 0.13;
phiwinfrw[5][5][1] = 0.1;
phiwinfrw[5][5][0] = 0.1;

phiwinfrw[5][4][3] = 0.0027;
phiwinfrw[5][4][2] = 0.0027;
phiwinfrw[5][4][1] = 0.005;
phiwinfrw[5][4][0] = 0.005;

phiwinfrw[5][3][2] = 0.005;
phiwinfrw[5][3][1] = 0.005;
phiwinfrw[5][3][0] = 0.005;

phiwinfrw[5][2][1] = 0.005;
phiwinfrw[5][2][0] = 0.005;

phiwinfrw[5][1][0] = 0.0035;

// =============== size of propagation cut in phi-z.
// =============== forward phi, phicutfrw

// ++++++++++++Last layer = 13

phicutfrw[13][13][12] = 0.06;
phicutfrw[13][13][11] = 0.014;
phicutfrw[13][13][10] = 0.014;
phicutfrw[13][13][9] = 0.014;
phicutfrw[13][13][8] = 0.014;
phicutfrw[13][13][7] = 0.014;
phicutfrw[13][13][6] = 0.014;
phicutfrw[13][13][5] = 0.014;
phicutfrw[13][13][4] = 0.014;
phicutfrw[13][13][3] = 0.014;
phicutfrw[13][13][2] = 0.014;
phicutfrw[13][13][1] = 0.014;
phicutfrw[13][13][0] = 0.014;

phicutfrw[13][12][11] = 0.004;
phicutfrw[13][12][10] = 0.0018;
phicutfrw[13][12][9] = 0.0027;
phicutfrw[13][12][8] = 0.0027;
phicutfrw[13][12][7] = 0.0027;
phicutfrw[13][12][6] = 0.0027;
phicutfrw[13][12][5] = 0.0027;
phicutfrw[13][12][4] = 0.0027;
phicutfrw[13][12][3] = 0.0027;
phicutfrw[13][12][2] = 0.0027;
phicutfrw[13][12][1] = 0.005;
phicutfrw[13][12][0] = 0.005;

phicutfrw[13][11][10] = 0.0009;
phicutfrw[13][11][9] = 0.0018;
phicutfrw[13][11][8] = 0.0027;
phicutfrw[13][11][7] = 0.0027;
phicutfrw[13][11][6] = 0.0027;
phicutfrw[13][11][5] = 0.0027;
phicutfrw[13][11][4] = 0.0027;
phicutfrw[13][11][3] = 0.0027;
phicutfrw[13][11][2] = 0.0027;
phicutfrw[13][11][1] = 0.005;
phicutfrw[13][11][0] = 0.005;

phicutfrw[13][10][9] = 0.002;
phicutfrw[13][10][8] = 0.0018;
phicutfrw[13][10][7] = 0.0027;
phicutfrw[13][10][6] = 0.0027;
phicutfrw[13][10][5] = 0.0027;
phicutfrw[13][10][4] = 0.0027;
phicutfrw[13][10][3] = 0.0027;
phicutfrw[13][10][2] = 0.0027;
phicutfrw[13][10][1] = 0.005;
phicutfrw[13][10][0] = 0.005;

phicutfrw[13][9][8] = 0.00095;
phicutfrw[13][9][7] = 0.0018;
phicutfrw[13][9][6] = 0.0027;
phicutfrw[13][9][5] = 0.0027;
phicutfrw[13][9][4] = 0.0027;
phicutfrw[13][9][3] = 0.0027;
phicutfrw[13][9][2] = 0.0027;
phicutfrw[13][9][1] = 0.005;
phicutfrw[13][9][0] = 0.005;

phicutfrw[13][8][7] = 0.00095;
phicutfrw[13][8][6] = 0.0018;
phicutfrw[13][8][5] = 0.0027;
phicutfrw[13][8][4] = 0.0027;
phicutfrw[13][8][3] = 0.0027;
phicutfrw[13][8][2] = 0.0027;
phicutfrw[13][8][1] = 0.005;
phicutfrw[13][8][0] = 0.005;

phicutfrw[13][7][6] = 0.0009;
phicutfrw[13][7][5] = 0.0018;
phicutfrw[13][7][4] = 0.0027;
phicutfrw[13][7][3] = 0.0027;
phicutfrw[13][7][2] = 0.0027;
phicutfrw[13][7][1] = 0.005;
phicutfrw[13][7][0] = 0.005;

phicutfrw[13][6][5] = 0.0018;
phicutfrw[13][6][4] = 0.0018;
phicutfrw[13][6][3] = 0.0027;
phicutfrw[13][6][2] = 0.0027;
phicutfrw[13][6][1] = 0.005;
phicutfrw[13][6][0] = 0.005;

phicutfrw[13][5][4] = 0.0009;
phicutfrw[13][5][3] = 0.0018;
phicutfrw[13][5][2] = 0.0027;
phicutfrw[13][5][1] = 0.005;
phicutfrw[13][5][0] = 0.005;

phicutfrw[13][4][3] = 0.0027;
phicutfrw[13][4][2] = 0.0027;
phicutfrw[13][4][1] = 0.005;
phicutfrw[13][4][0] = 0.005;

phicutfrw[13][3][2] = 0.005;
phicutfrw[13][3][1] = 0.005;
phicutfrw[13][3][0] = 0.005;

phicutfrw[13][2][1] = 0.005;
phicutfrw[13][2][0] = 0.003;

phicutfrw[13][1][0] = 0.0035;

// +++++++++++ Last layer = 12

phicutfrw[12][12][11] = 0.06;
phicutfrw[12][12][10] = 0.014;
phicutfrw[12][12][9] = 0.014;
phicutfrw[12][12][8] = 0.014;
phicutfrw[12][12][7] = 0.014;
phicutfrw[12][12][6] = 0.014;
phicutfrw[12][12][5] = 0.014;
phicutfrw[12][12][4] = 0.014;
phicutfrw[12][12][3] = 0.014;
phicutfrw[12][12][2] = 0.014;
phicutfrw[12][12][1] = 0.014;
phicutfrw[12][12][0] = 0.014;

phicutfrw[12][11][10] = 0.0014;
phicutfrw[12][11][9] = 0.0018;
phicutfrw[12][11][8] = 0.0027;
phicutfrw[12][11][7] = 0.0027;
phicutfrw[12][11][6] = 0.0027;
phicutfrw[12][11][5] = 0.0027;
phicutfrw[12][11][4] = 0.0027;
phicutfrw[12][11][3] = 0.0027;
phicutfrw[12][11][2] = 0.0027;
phicutfrw[12][11][1] = 0.005;
phicutfrw[12][11][0] = 0.005;


phicutfrw[12][10][9] = 0.0009;
phicutfrw[12][10][8] = 0.0018;
phicutfrw[12][10][7] = 0.0027;
phicutfrw[12][10][6] = 0.0027;
phicutfrw[12][10][5] = 0.0027;
phicutfrw[12][10][4] = 0.0027;
phicutfrw[12][10][3] = 0.0027;
phicutfrw[12][10][2] = 0.0027;
phicutfrw[12][10][1] = 0.005;
phicutfrw[12][10][0] = 0.005;

phicutfrw[12][9][8] = 0.0009;
phicutfrw[12][9][7] = 0.0018;
phicutfrw[12][9][6] = 0.0027;
phicutfrw[12][9][5] = 0.0027;
phicutfrw[12][9][4] = 0.0027;
phicutfrw[12][9][3] = 0.0027;
phicutfrw[12][9][2] = 0.0027;
phicutfrw[12][9][1] = 0.005;
phicutfrw[12][9][0] = 0.005;

phicutfrw[12][8][7] = 0.0009;
phicutfrw[12][8][6] = 0.0018;
phicutfrw[12][8][5] = 0.0027;
phicutfrw[12][8][4] = 0.0027;
phicutfrw[12][8][3] = 0.0027;
phicutfrw[12][8][2] = 0.0027;
phicutfrw[12][8][1] = 0.005;
phicutfrw[12][8][0] = 0.005;

phicutfrw[12][7][6] = 0.0009;
phicutfrw[12][7][5] = 0.0018;
phicutfrw[12][7][4] = 0.0027;
phicutfrw[12][7][3] = 0.0027;
phicutfrw[12][7][2] = 0.0027;
phicutfrw[12][7][1] = 0.005;
phicutfrw[12][7][0] = 0.005;

phicutfrw[12][6][5] = 0.0015;
phicutfrw[12][6][4] = 0.0018;
phicutfrw[12][6][3] = 0.0027;
phicutfrw[12][6][2] = 0.0027;
phicutfrw[12][6][1] = 0.005;
phicutfrw[12][6][0] = 0.005;

phicutfrw[12][5][4] = 0.0009;
phicutfrw[12][5][3] = 0.0018;
phicutfrw[12][5][2] = 0.0027;
phicutfrw[12][5][1] = 0.005;
phicutfrw[12][5][0] = 0.005;

phicutfrw[12][4][3] = 0.0027;
phicutfrw[12][4][2] = 0.0027;
phicutfrw[12][4][1] = 0.005;
phicutfrw[12][4][0] = 0.005;

phicutfrw[12][3][2] = 0.005;
phicutfrw[12][3][1] = 0.005;
phicutfrw[12][3][0] = 0.005;

phicutfrw[12][2][1] = 0.005;
phicutfrw[12][2][0] = 0.005;

phicutfrw[12][1][0] = 0.0035;

// +++++++++++ Last layer = 11

phicutfrw[11][11][10] = 0.06;
phicutfrw[11][11][9] = 0.014;
phicutfrw[11][11][8] = 0.014;
phicutfrw[11][11][7] = 0.014;
phicutfrw[11][11][6] = 0.014;
phicutfrw[11][11][5] = 0.014;
phicutfrw[11][11][4] = 0.014;
phicutfrw[11][11][3] = 0.014;
phicutfrw[11][11][2] = 0.014;
phicutfrw[11][11][1] = 0.014;
phicutfrw[11][11][0] = 0.014;


phicutfrw[11][10][9] = 0.0009;
phicutfrw[11][10][8] = 0.0018;
phicutfrw[11][10][7] = 0.0027;
phicutfrw[11][10][6] = 0.0027;
phicutfrw[11][10][5] = 0.0027;
phicutfrw[11][10][4] = 0.0027;
phicutfrw[11][10][3] = 0.0027;
phicutfrw[11][10][2] = 0.0027;
phicutfrw[11][10][1] = 0.005;
phicutfrw[11][10][0] = 0.005;

phicutfrw[11][9][8] = 0.0009;
phicutfrw[11][9][7] = 0.0018;
phicutfrw[11][9][6] = 0.0027;
phicutfrw[11][9][5] = 0.0027;
phicutfrw[11][9][4] = 0.0027;
phicutfrw[11][9][3] = 0.0027;
phicutfrw[11][9][2] = 0.0027;
phicutfrw[11][9][1] = 0.005;
phicutfrw[11][9][0] = 0.005;

phicutfrw[11][8][7] = 0.0014;
phicutfrw[11][8][6] = 0.0018;
phicutfrw[11][8][5] = 0.0027;
phicutfrw[11][8][4] = 0.0027;
phicutfrw[11][8][3] = 0.0027;
phicutfrw[11][8][2] = 0.0027;
phicutfrw[11][8][1] = 0.005;
phicutfrw[11][8][0] = 0.005;

phicutfrw[11][7][6] = 0.0009;
phicutfrw[11][7][5] = 0.0018;
phicutfrw[11][7][4] = 0.0027;
phicutfrw[11][7][3] = 0.0027;
phicutfrw[11][7][2] = 0.0027;
phicutfrw[11][7][1] = 0.005;
phicutfrw[11][7][0] = 0.005;

phicutfrw[11][6][5] = 0.0009;
phicutfrw[11][6][4] = 0.0018;
phicutfrw[11][6][3] = 0.0027;
phicutfrw[11][6][2] = 0.0027;
phicutfrw[11][6][1] = 0.005;
phicutfrw[11][6][0] = 0.005;

phicutfrw[11][5][4] = 0.0009;
phicutfrw[11][5][3] = 0.0018;
phicutfrw[11][5][2] = 0.0027;
phicutfrw[11][5][1] = 0.005;
phicutfrw[11][5][0] = 0.005;

phicutfrw[11][4][3] = 0.0027;
phicutfrw[11][4][2] = 0.0027;
phicutfrw[11][4][1] = 0.005;
phicutfrw[11][4][0] = 0.005;

phicutfrw[11][3][2] = 0.005;
phicutfrw[11][3][1] = 0.005;
phicutfrw[11][3][0] = 0.005;

phicutfrw[11][2][1] = 0.005;
phicutfrw[11][2][0] = 0.005;

phicutfrw[11][1][0] = 0.0035;

// +++++++++++ Last layer = 10

phicutfrw[10][10][9] = 0.06;
phicutfrw[10][10][8] = 0.014;
phicutfrw[10][10][7] = 0.014;
phicutfrw[10][10][6] = 0.014;
phicutfrw[10][10][5] = 0.014;
phicutfrw[10][10][4] = 0.014;
phicutfrw[10][10][3] = 0.014;
phicutfrw[10][10][2] = 0.014;
phicutfrw[10][10][1] = 0.014;
phicutfrw[10][10][0] = 0.014;

phicutfrw[10][9][8] = 0.0009;
phicutfrw[10][9][7] = 0.0018;
phicutfrw[10][9][6] = 0.0027;
phicutfrw[10][9][5] = 0.0027;
phicutfrw[10][9][4] = 0.0027;
phicutfrw[10][9][3] = 0.0027;
phicutfrw[10][9][2] = 0.0027;
phicutfrw[10][9][1] = 0.005;
phicutfrw[10][9][0] = 0.005;

phicutfrw[10][8][7] = 0.0009;
phicutfrw[10][8][6] = 0.0018;
phicutfrw[10][8][5] = 0.0027;
phicutfrw[10][8][4] = 0.0027;
phicutfrw[10][8][3] = 0.0027;
phicutfrw[10][8][2] = 0.0027;
phicutfrw[10][8][1] = 0.005;
phicutfrw[10][8][0] = 0.005;

phicutfrw[10][7][6] = 0.0009;
phicutfrw[10][7][5] = 0.0018;
phicutfrw[10][7][4] = 0.0027;
phicutfrw[10][7][3] = 0.0027;
phicutfrw[10][7][2] = 0.0027;
phicutfrw[10][7][1] = 0.005;
phicutfrw[10][7][0] = 0.005;

phicutfrw[10][6][5] = 0.0009;
phicutfrw[10][6][4] = 0.0018;
phicutfrw[10][6][3] = 0.0027;
phicutfrw[10][6][2] = 0.0027;
phicutfrw[10][6][1] = 0.005;
phicutfrw[10][6][0] = 0.005;

phicutfrw[10][5][4] = 0.0009;
phicutfrw[10][5][3] = 0.0018;
phicutfrw[10][5][2] = 0.0027;
phicutfrw[10][5][1] = 0.005;
phicutfrw[10][5][0] = 0.005;

phicutfrw[10][4][3] = 0.0027;
phicutfrw[10][4][2] = 0.0027;
phicutfrw[10][4][1] = 0.005;
phicutfrw[10][4][0] = 0.005;

phicutfrw[10][3][2] = 0.005;
phicutfrw[10][3][1] = 0.005;
phicutfrw[10][3][0] = 0.005;

phicutfrw[10][2][1] = 0.005;
phicutfrw[10][2][0] = 0.005;

phicutfrw[10][1][0] = 0.0035;

// +++++++++++ Last layer = 9


phicutfrw[9][9][8] = 0.06;
phicutfrw[9][9][7] = 0.014;
phicutfrw[9][9][6] = 0.014;
phicutfrw[9][9][5] = 0.014;
phicutfrw[9][9][4] = 0.014;
phicutfrw[9][9][3] = 0.014;
phicutfrw[9][9][2] = 0.014;
phicutfrw[9][9][1] = 0.014;
phicutfrw[9][9][0] = 0.014;

phicutfrw[9][8][7] = 0.0009;
phicutfrw[9][8][6] = 0.0018;
phicutfrw[9][8][5] = 0.0027;
phicutfrw[9][8][4] = 0.0027;
phicutfrw[9][8][3] = 0.0027;
phicutfrw[9][8][2] = 0.0027;
phicutfrw[9][8][1] = 0.005;
phicutfrw[9][8][0] = 0.005;

phicutfrw[9][7][6] = 0.0009;
phicutfrw[9][7][5] = 0.0018;
phicutfrw[9][7][4] = 0.0027;
phicutfrw[9][7][3] = 0.0027;
phicutfrw[9][7][2] = 0.0027;
phicutfrw[9][7][1] = 0.005;
phicutfrw[9][7][0] = 0.005;

phicutfrw[9][6][5] = 0.0009;
phicutfrw[9][6][4] = 0.0018;
phicutfrw[9][6][3] = 0.0027;
phicutfrw[9][6][2] = 0.0027;
phicutfrw[9][6][1] = 0.005;
phicutfrw[9][6][0] = 0.005;

phicutfrw[9][5][4] = 0.0009;
phicutfrw[9][5][3] = 0.0018;
phicutfrw[9][5][2] = 0.0027;
phicutfrw[9][5][1] = 0.005;
phicutfrw[9][5][0] = 0.005;

phicutfrw[9][4][3] = 0.0027;
phicutfrw[9][4][2] = 0.0027;
phicutfrw[9][4][1] = 0.005;
phicutfrw[9][4][0] = 0.005;

phicutfrw[9][3][2] = 0.005;
phicutfrw[9][3][1] = 0.005;
phicutfrw[9][3][0] = 0.005;

phicutfrw[9][2][1] = 0.005;
phicutfrw[9][2][0] = 0.005;

phicutfrw[9][1][0] = 0.0035;

// +++++++++++ Last layer = 8


phicutfrw[8][8][7] = 0.06;
phicutfrw[8][8][6] = 0.014;
phicutfrw[8][8][5] = 0.014;
phicutfrw[8][8][4] = 0.014;
phicutfrw[8][8][3] = 0.014;
phicutfrw[8][8][2] = 0.05;
phicutfrw[8][8][1] = 0.05;
phicutfrw[8][8][0] = 0.05;

phicutfrw[8][7][6] = 0.0008;
phicutfrw[8][7][5] = 0.0018;
phicutfrw[8][7][4] = 0.0027;
phicutfrw[8][7][3] = 0.0027;
phicutfrw[8][7][2] = 0.0027;
phicutfrw[8][7][1] = 0.005;
phicutfrw[8][7][0] = 0.005;

phicutfrw[8][6][5] = 0.0008;
phicutfrw[8][6][4] = 0.0018;
phicutfrw[8][6][3] = 0.0027;
phicutfrw[8][6][2] = 0.0027;
phicutfrw[8][6][1] = 0.005;
phicutfrw[8][6][0] = 0.005;

phicutfrw[8][5][4] = 0.0008;
phicutfrw[8][5][3] = 0.0018;
phicutfrw[8][5][2] = 0.0027;
phicutfrw[8][5][1] = 0.005;
phicutfrw[8][5][0] = 0.005;

phicutfrw[8][4][3] = 0.0027;
phicutfrw[8][4][2] = 0.0027;
phicutfrw[8][4][1] = 0.005;
phicutfrw[8][4][0] = 0.005;

phicutfrw[8][3][2] = 0.005;
phicutfrw[8][3][1] = 0.005;
phicutfrw[8][3][0] = 0.005;

phicutfrw[8][2][1] = 0.005;
phicutfrw[8][2][0] = 0.005;

phicutfrw[8][1][0] = 0.0035;

// +++++++++++ Last layer = 7

phicutfrw[7][7][6] = 0.06;
phicutfrw[7][7][5] = 0.014;
phicutfrw[7][7][4] = 0.014;
phicutfrw[7][7][3] = 0.014;
phicutfrw[7][7][2] = 0.05;
phicutfrw[7][7][1] = 0.05;
phicutfrw[7][7][0] = 0.05;

phicutfrw[7][6][5] = 0.0008;
phicutfrw[7][6][4] = 0.0018;
phicutfrw[7][6][3] = 0.0027;
phicutfrw[7][6][2] = 0.0027;
phicutfrw[7][6][1] = 0.005;
phicutfrw[7][6][0] = 0.005;

phicutfrw[7][5][4] = 0.0008;
phicutfrw[7][5][3] = 0.0018;
phicutfrw[7][5][2] = 0.0027;
phicutfrw[7][5][1] = 0.005;
phicutfrw[7][5][0] = 0.005;

phicutfrw[7][4][3] = 0.0027;
phicutfrw[7][4][2] = 0.0027;
phicutfrw[7][4][1] = 0.005;
phicutfrw[7][4][0] = 0.005;

phicutfrw[7][3][2] = 0.005;
phicutfrw[7][3][1] = 0.005;
phicutfrw[7][3][0] = 0.005;

phicutfrw[7][2][1] = 0.005;
phicutfrw[7][2][0] = 0.005;

phicutfrw[7][1][0] = 0.0035;

// +++++++++++ Last layer = 6

phicutfrw[6][6][5] = 0.06;
phicutfrw[6][6][4] = 0.018;
phicutfrw[6][6][3] = 0.027;
phicutfrw[6][6][2] = 0.05;
phicutfrw[6][6][1] = 0.05;
phicutfrw[6][6][0] = 0.05;

phicutfrw[6][5][4] = 0.0008;
phicutfrw[6][5][3] = 0.0018;
phicutfrw[6][5][2] = 0.0027;
phicutfrw[6][5][1] = 0.005;
phicutfrw[6][5][0] = 0.005;

phicutfrw[6][4][3] = 0.0027;
phicutfrw[6][4][2] = 0.0027;
phicutfrw[6][4][1] = 0.005;
phicutfrw[6][4][0] = 0.005;

phicutfrw[6][3][2] = 0.005;
phicutfrw[6][3][1] = 0.005;
phicutfrw[6][3][0] = 0.005;

phicutfrw[6][2][1] = 0.005;
phicutfrw[6][2][0] = 0.005;

phicutfrw[6][1][0] = 0.0035;

// +++++++++++ Last layer = 5

phicutfrw[5][5][4] = 0.009;
phicutfrw[5][5][3] = 0.01;
phicutfrw[5][5][2] = 0.01;
phicutfrw[5][5][1] = 0.05;
phicutfrw[5][5][0] = 0.05;

phicutfrw[5][4][3] = 0.0027;
phicutfrw[5][4][2] = 0.0027;
phicutfrw[5][4][1] = 0.005;
phicutfrw[5][4][0] = 0.005;

phicutfrw[5][3][2] = 0.005;
phicutfrw[5][3][1] = 0.005;
phicutfrw[5][3][0] = 0.005;

phicutfrw[5][2][1] = 0.005;
phicutfrw[5][2][0] = 0.005;

phicutfrw[5][1][0] = 0.0035;



// =============== forward z, zwinfbb

// ++++++++++++Last layer = 13

zwinfbb[13][12][11] = 10.;
zwinfbb[13][12][10] = 10.;
zwinfbb[13][12][9] = 10.;
zwinfbb[13][12][8] = 10.;
zwinfbb[13][12][7] = 10.;
zwinfbb[13][12][6] = 10.;
zwinfbb[13][12][5] = 10.;
zwinfbb[13][12][4] = 10.;
zwinfbb[13][12][3] = 10.;
zwinfbb[13][12][2] = 10.;
zwinfbb[13][12][1] = 10.;
zwinfbb[13][12][0] = 10.;

zwinfbb[13][11][10] = 10.;
zwinfbb[13][11][9] = 10.;
zwinfbb[13][11][8] = 10.;
zwinfbb[13][11][7] = 10.;
zwinfbb[13][11][6] = 10.;
zwinfbb[13][11][5] = 10.;
zwinfbb[13][11][4] = 10.;
zwinfbb[13][11][3] = 10.;
zwinfbb[13][11][2] = 10.;
zwinfbb[13][11][1] = 10.;
zwinfbb[13][11][0] = 10.;

zwinfbb[13][10][9] = 10.;
zwinfbb[13][10][8] = 10.;
zwinfbb[13][10][7] = 10.;
zwinfbb[13][10][6] = 10.;
zwinfbb[13][10][5] = 10.;
zwinfbb[13][10][4] = 10.;
zwinfbb[13][10][3] = 10.;
zwinfbb[13][10][2] = 10.;
zwinfbb[13][10][1] = 10.;
zwinfbb[13][10][0] = 10.;

zwinfbb[13][9][8] = 10.;
zwinfbb[13][9][7] = 10.;
zwinfbb[13][9][6] = 10.;
zwinfbb[13][9][5] = 10.;
zwinfbb[13][9][4] = 10.;
zwinfbb[13][9][3] = 10.;
zwinfbb[13][9][2] = 10.;
zwinfbb[13][9][1] = 10.;
zwinfbb[13][9][0] = 10.;


zwinfbb[13][8][7] = 10.;
zwinfbb[13][8][6] = 10.;
zwinfbb[13][8][5] = 10.;
zwinfbb[13][8][4] = 10.;
zwinfbb[13][8][3] = 10.;
zwinfbb[13][8][2] = 10.;
zwinfbb[13][8][1] = 10.;
zwinfbb[13][8][0] = 10.;


zwinfbb[13][7][6] = 10.;
zwinfbb[13][7][5] = 10.;
zwinfbb[13][7][4] = 10.;
zwinfbb[13][7][3] = 10.;
zwinfbb[13][7][2] = 10.;
zwinfbb[13][7][1] = 10.;
zwinfbb[13][7][0] = 10.;

zwinfbb[13][6][5] = 10.;
zwinfbb[13][6][4] = 10.;
zwinfbb[13][6][3] = 10.;
zwinfbb[13][6][2] = 10.;
zwinfbb[13][6][1] = 10.;
zwinfbb[13][6][0] = 10.;

zwinfbb[13][5][4] = 10.;
zwinfbb[13][5][3] = 10.;
zwinfbb[13][5][2] = 10.;
zwinfbb[13][5][1] = 10.;
zwinfbb[13][5][0] = 10.;

zwinfbb[13][4][3] = 10.;
zwinfbb[13][4][2] = 10.;
zwinfbb[13][4][1] = 10.;
zwinfbb[13][4][0] = 10.;

zwinfbb[13][3][2] = 10.;
zwinfbb[13][3][1] = 10.;
zwinfbb[13][3][0] = 10.;

zwinfbb[13][2][1] = 1.;
zwinfbb[13][2][0] = 1.;

zwinfbb[13][1][0] = 1.;


// ++++++++++++Last layer = 12

zwinfbb[12][12][11] = 10.;
zwinfbb[12][12][10] = 10.;
zwinfbb[12][12][9] = 10.;
zwinfbb[12][12][8] = 10.;
zwinfbb[12][12][7] = 10.;
zwinfbb[12][12][6] = 10.;
zwinfbb[12][12][5] = 10.;
zwinfbb[12][12][4] = 10.;
zwinfbb[12][12][3] = 10.;
zwinfbb[12][12][2] = 10.;
zwinfbb[12][12][1] = 10.;
zwinfbb[12][12][0] = 10.;

zwinfbb[12][11][10] = 10.;
zwinfbb[12][11][9] = 10.;
zwinfbb[12][11][8] = 10.;
zwinfbb[12][11][7] = 10.;
zwinfbb[12][11][6] = 10.;
zwinfbb[12][11][5] = 10.;
zwinfbb[12][11][4] = 10.;
zwinfbb[12][11][3] = 10.;
zwinfbb[12][11][2] = 10.;
zwinfbb[12][11][1] = 10.;
zwinfbb[12][11][0] = 10.;

zwinfbb[12][10][9] = 10.;
zwinfbb[12][10][8] = 10.;
zwinfbb[12][10][7] = 10.;
zwinfbb[12][10][6] = 10.;
zwinfbb[12][10][5] = 10.;
zwinfbb[12][10][4] = 10.;
zwinfbb[12][10][3] = 10.;
zwinfbb[12][10][2] = 10.;
zwinfbb[12][10][1] = 10.;
zwinfbb[12][10][0] = 10.;

zwinfbb[12][9][8] = 10.;
zwinfbb[12][9][7] = 10.;
zwinfbb[12][9][6] = 10.;
zwinfbb[12][9][5] = 10.;
zwinfbb[12][9][4] = 10.;
zwinfbb[12][9][3] = 10.;
zwinfbb[12][9][2] = 10.;
zwinfbb[12][9][1] = 10.;
zwinfbb[12][9][0] = 10.;


zwinfbb[12][8][7] = 10.;
zwinfbb[12][8][6] = 10.;
zwinfbb[12][8][5] = 10.;
zwinfbb[12][8][4] = 10.;
zwinfbb[12][8][3] = 10.;
zwinfbb[12][8][2] = 10.;
zwinfbb[12][8][1] = 10.;
zwinfbb[12][8][0] = 10.;


zwinfbb[12][7][6] = 10.;
zwinfbb[12][7][5] = 10.;
zwinfbb[12][7][4] = 10.;
zwinfbb[12][7][3] = 10.;
zwinfbb[12][7][2] = 10.;
zwinfbb[12][7][1] = 10.;
zwinfbb[12][7][0] = 10.;

zwinfbb[12][6][5] = 10.;
zwinfbb[12][6][4] = 10.;
zwinfbb[12][6][3] = 10.;
zwinfbb[12][6][2] = 10.;
zwinfbb[12][6][1] = 10.;
zwinfbb[12][6][0] = 10.;

zwinfbb[12][5][4] = 10.;
zwinfbb[12][5][3] = 10.;
zwinfbb[12][5][2] = 10.;
zwinfbb[12][5][1] = 10.;
zwinfbb[12][5][0] = 10.;

zwinfbb[12][4][3] = 10.;
zwinfbb[12][4][2] = 10.;
zwinfbb[12][4][1] = 10.;
zwinfbb[12][4][0] = 10.;

zwinfbb[12][3][2] = 10.;
zwinfbb[12][3][1] = 10.;
zwinfbb[12][3][0] = 10.;

zwinfbb[12][2][1] = 1.;
zwinfbb[12][2][0] = 1.;

zwinfbb[12][1][0] = 1.;

// ++++++++++++Last layer = 11

zwinfbb[11][12][11] = 10.;
zwinfbb[11][12][10] = 10.;
zwinfbb[11][12][9] = 10.;
zwinfbb[11][12][8] = 10.;
zwinfbb[11][12][7] = 10.;
zwinfbb[11][12][6] = 10.;
zwinfbb[11][12][5] = 10.;
zwinfbb[11][12][4] = 10.;
zwinfbb[11][12][3] = 10.;
zwinfbb[11][12][2] = 10.;
zwinfbb[11][12][1] = 10.;
zwinfbb[11][12][0] = 10.;

zwinfbb[11][11][10] = 10.;
zwinfbb[11][11][9] = 10.;
zwinfbb[11][11][8] = 10.;
zwinfbb[11][11][7] = 10.;
zwinfbb[11][11][6] = 10.;
zwinfbb[11][11][5] = 10.;
zwinfbb[11][11][4] = 10.;
zwinfbb[11][11][3] = 10.;
zwinfbb[11][11][2] = 10.;
zwinfbb[11][11][1] = 10.;
zwinfbb[11][11][0] = 10.;

zwinfbb[11][10][9] = 10.;
zwinfbb[11][10][8] = 10.;
zwinfbb[11][10][7] = 10.;
zwinfbb[11][10][6] = 10.;
zwinfbb[11][10][5] = 10.;
zwinfbb[11][10][4] = 10.;
zwinfbb[11][10][3] = 10.;
zwinfbb[11][10][2] = 10.;
zwinfbb[11][10][1] = 10.;
zwinfbb[11][10][0] = 10.;

zwinfbb[11][9][8] = 10.;
zwinfbb[11][9][7] = 10.;
zwinfbb[11][9][6] = 10.;
zwinfbb[11][9][5] = 10.;
zwinfbb[11][9][4] = 10.;
zwinfbb[11][9][3] = 10.;
zwinfbb[11][9][2] = 10.;
zwinfbb[11][9][1] = 10.;
zwinfbb[11][9][0] = 10.;


zwinfbb[11][8][7] = 10.;
zwinfbb[11][8][6] = 10.;
zwinfbb[11][8][5] = 10.;
zwinfbb[11][8][4] = 10.;
zwinfbb[11][8][3] = 10.;
zwinfbb[11][8][2] = 10.;
zwinfbb[11][8][1] = 10.;
zwinfbb[11][8][0] = 10.;


zwinfbb[11][7][6] = 10.;
zwinfbb[11][7][5] = 10.;
zwinfbb[11][7][4] = 10.;
zwinfbb[11][7][3] = 10.;
zwinfbb[11][7][2] = 10.;
zwinfbb[11][7][1] = 10.;
zwinfbb[11][7][0] = 10.;

zwinfbb[11][6][5] = 10.;
zwinfbb[11][6][4] = 10.;
zwinfbb[11][6][3] = 10.;
zwinfbb[11][6][2] = 10.;
zwinfbb[11][6][1] = 10.;
zwinfbb[11][6][0] = 10.;

zwinfbb[11][5][4] = 10.;
zwinfbb[11][5][3] = 10.;
zwinfbb[11][5][2] = 10.;
zwinfbb[11][5][1] = 10.;
zwinfbb[11][5][0] = 10.;

zwinfbb[11][4][3] = 10.;
zwinfbb[11][4][2] = 10.;
zwinfbb[11][4][1] = 10.;
zwinfbb[11][4][0] = 10.;

zwinfbb[11][3][2] = 10.;
zwinfbb[11][3][1] = 10.;
zwinfbb[11][3][0] = 10.;

zwinfbb[11][2][1] = 1.;
zwinfbb[11][2][0] = 1.;

zwinfbb[11][1][0] = 1.;


// ++++++++++++Last layer = 10

zwinfbb[10][12][11] = 10.;
zwinfbb[10][12][10] = 10.;
zwinfbb[10][12][9] = 10.;
zwinfbb[10][12][8] = 10.;
zwinfbb[10][12][7] = 10.;
zwinfbb[10][12][6] = 10.;
zwinfbb[10][12][5] = 10.;
zwinfbb[10][12][4] = 10.;
zwinfbb[10][12][3] = 10.;
zwinfbb[10][12][2] = 10.;
zwinfbb[10][12][1] = 10.;
zwinfbb[10][12][0] = 10.;

zwinfbb[10][11][10] = 10.;
zwinfbb[10][11][9] = 10.;
zwinfbb[10][11][8] = 10.;
zwinfbb[10][11][7] = 10.;
zwinfbb[10][11][6] = 10.;
zwinfbb[10][11][5] = 10.;
zwinfbb[10][11][4] = 10.;
zwinfbb[10][11][3] = 10.;
zwinfbb[10][11][2] = 10.;
zwinfbb[10][11][1] = 10.;
zwinfbb[10][11][0] = 10.;

zwinfbb[10][10][9] = 10.;
zwinfbb[10][10][8] = 10.;
zwinfbb[10][10][7] = 10.;
zwinfbb[10][10][6] = 10.;
zwinfbb[10][10][5] = 10.;
zwinfbb[10][10][4] = 10.;
zwinfbb[10][10][3] = 10.;
zwinfbb[10][10][2] = 10.;
zwinfbb[10][10][1] = 10.;
zwinfbb[10][10][0] = 10.;

zwinfbb[10][9][8] = 10.;
zwinfbb[10][9][7] = 10.;
zwinfbb[10][9][6] = 10.;
zwinfbb[10][9][5] = 10.;
zwinfbb[10][9][4] = 10.;
zwinfbb[10][9][3] = 10.;
zwinfbb[10][9][2] = 10.;
zwinfbb[10][9][1] = 10.;
zwinfbb[10][9][0] = 10.;


zwinfbb[10][8][7] = 10.;
zwinfbb[10][8][6] = 10.;
zwinfbb[10][8][5] = 10.;
zwinfbb[10][8][4] = 10.;
zwinfbb[10][8][3] = 10.;
zwinfbb[10][8][2] = 10.;
zwinfbb[10][8][1] = 10.;
zwinfbb[10][8][0] = 10.;


zwinfbb[10][7][6] = 10.;
zwinfbb[10][7][5] = 10.;
zwinfbb[10][7][4] = 10.;
zwinfbb[10][7][3] = 10.;
zwinfbb[10][7][2] = 10.;
zwinfbb[10][7][1] = 10.;
zwinfbb[10][7][0] = 10.;

zwinfbb[10][6][5] = 10.;
zwinfbb[10][6][4] = 10.;
zwinfbb[10][6][3] = 10.;
zwinfbb[10][6][2] = 10.;
zwinfbb[10][6][1] = 10.;
zwinfbb[10][6][0] = 10.;

zwinfbb[10][5][4] = 10.;
zwinfbb[10][5][3] = 10.;
zwinfbb[10][5][2] = 10.;
zwinfbb[10][5][1] = 10.;
zwinfbb[10][5][0] = 10.;

zwinfbb[10][4][3] = 10.;
zwinfbb[10][4][2] = 10.;
zwinfbb[10][4][1] = 10.;
zwinfbb[10][4][0] = 10.;

zwinfbb[10][3][2] = 10.;
zwinfbb[10][3][1] = 10.;
zwinfbb[10][3][0] = 10.;

zwinfbb[10][2][1] = 1.;
zwinfbb[10][2][0] = 1.;

zwinfbb[10][1][0] = 1.;

// ++++++++++++Last layer = 9

zwinfbb[9][12][11] = 10.;
zwinfbb[9][12][10] = 10.;
zwinfbb[9][12][9] = 10.;
zwinfbb[9][12][8] = 10.;
zwinfbb[9][12][7] = 10.;
zwinfbb[9][12][6] = 10.;
zwinfbb[9][12][5] = 10.;
zwinfbb[9][12][4] = 10.;
zwinfbb[9][12][3] = 10.;
zwinfbb[9][12][2] = 10.;
zwinfbb[9][12][1] = 10.;
zwinfbb[9][12][0] = 10.;

zwinfbb[9][11][10] = 10.;
zwinfbb[9][11][9] = 10.;
zwinfbb[9][11][8] = 10.;
zwinfbb[9][11][7] = 10.;
zwinfbb[9][11][6] = 10.;
zwinfbb[9][11][5] = 10.;
zwinfbb[9][11][4] = 10.;
zwinfbb[9][11][3] = 10.;
zwinfbb[9][11][2] = 10.;
zwinfbb[9][11][1] = 10.;
zwinfbb[9][11][0] = 10.;

zwinfbb[9][10][9] = 10.;
zwinfbb[9][10][8] = 10.;
zwinfbb[9][10][7] = 10.;
zwinfbb[9][10][6] = 10.;
zwinfbb[9][10][5] = 10.;
zwinfbb[9][10][4] = 10.;
zwinfbb[9][10][3] = 10.;
zwinfbb[9][10][2] = 10.;
zwinfbb[9][10][1] = 10.;
zwinfbb[9][10][0] = 10.;

zwinfbb[9][9][8] = 10.;
zwinfbb[9][9][7] = 10.;
zwinfbb[9][9][6] = 10.;
zwinfbb[9][9][5] = 10.;
zwinfbb[9][9][4] = 10.;
zwinfbb[9][9][3] = 10.;
zwinfbb[9][9][2] = 10.;
zwinfbb[9][9][1] = 10.;
zwinfbb[9][9][0] = 10.;


zwinfbb[9][8][7] = 10.;
zwinfbb[9][8][6] = 10.;
zwinfbb[9][8][5] = 10.;
zwinfbb[9][8][4] = 10.;
zwinfbb[9][8][3] = 10.;
zwinfbb[9][8][2] = 10.;
zwinfbb[9][8][1] = 10.;
zwinfbb[9][8][0] = 10.;


zwinfbb[9][7][6] = 10.;
zwinfbb[9][7][5] = 10.;
zwinfbb[9][7][4] = 10.;
zwinfbb[9][7][3] = 10.;
zwinfbb[9][7][2] = 10.;
zwinfbb[9][7][1] = 10.;
zwinfbb[9][7][0] = 10.;

zwinfbb[9][6][5] = 10.;
zwinfbb[9][6][4] = 10.;
zwinfbb[9][6][3] = 10.;
zwinfbb[9][6][2] = 10.;
zwinfbb[9][6][1] = 10.;
zwinfbb[9][6][0] = 10.;

zwinfbb[9][5][4] = 10.;
zwinfbb[9][5][3] = 10.;
zwinfbb[9][5][2] = 10.;
zwinfbb[9][5][1] = 10.;
zwinfbb[9][5][0] = 10.;

zwinfbb[9][4][3] = 10.;
zwinfbb[9][4][2] = 10.;
zwinfbb[9][4][1] = 10.;
zwinfbb[9][4][0] = 10.;

zwinfbb[9][3][2] = 10.;
zwinfbb[9][3][1] = 10.;
zwinfbb[9][3][0] = 10.;

zwinfbb[9][2][1] = 1.;
zwinfbb[9][2][0] = 1.;

zwinfbb[9][1][0] = 1.;

// ++++++++++++Last layer = 8

zwinfbb[8][12][11] = 10.;
zwinfbb[8][12][10] = 10.;
zwinfbb[8][12][9] = 10.;
zwinfbb[8][12][8] = 10.;
zwinfbb[8][12][7] = 10.;
zwinfbb[8][12][6] = 10.;
zwinfbb[8][12][5] = 10.;
zwinfbb[8][12][4] = 10.;
zwinfbb[8][12][3] = 10.;
zwinfbb[8][12][2] = 10.;
zwinfbb[8][12][1] = 10.;
zwinfbb[8][12][0] = 10.;

zwinfbb[8][11][10] = 10.;
zwinfbb[8][11][9] = 10.;
zwinfbb[8][11][8] = 10.;
zwinfbb[8][11][7] = 10.;
zwinfbb[8][11][6] = 10.;
zwinfbb[8][11][5] = 10.;
zwinfbb[8][11][4] = 10.;
zwinfbb[8][11][3] = 10.;
zwinfbb[8][11][2] = 10.;
zwinfbb[8][11][1] = 10.;
zwinfbb[8][11][0] = 10.;

zwinfbb[8][10][9] = 10.;
zwinfbb[8][10][8] = 10.;
zwinfbb[8][10][7] = 10.;
zwinfbb[8][10][6] = 10.;
zwinfbb[8][10][5] = 10.;
zwinfbb[8][10][4] = 10.;
zwinfbb[8][10][3] = 10.;
zwinfbb[8][10][2] = 10.;
zwinfbb[8][10][1] = 10.;
zwinfbb[8][10][0] = 10.;

zwinfbb[8][9][8] = 10.;
zwinfbb[8][9][7] = 10.;
zwinfbb[8][9][6] = 10.;
zwinfbb[8][9][5] = 10.;
zwinfbb[8][9][4] = 10.;
zwinfbb[8][9][3] = 10.;
zwinfbb[8][9][2] = 10.;
zwinfbb[8][9][1] = 10.;
zwinfbb[8][9][0] = 10.;


zwinfbb[8][8][7] = 10.;
zwinfbb[8][8][6] = 10.;
zwinfbb[8][8][5] = 10.;
zwinfbb[8][8][4] = 10.;
zwinfbb[8][8][3] = 10.;
zwinfbb[8][8][2] = 1.;
zwinfbb[8][8][1] = 1.;
zwinfbb[8][8][0] = 1.;


zwinfbb[8][7][6] = 10.;
zwinfbb[8][7][5] = 10.;
zwinfbb[8][7][4] = 10.;
zwinfbb[8][7][3] = 10.;
zwinfbb[8][7][2] = 1.;
zwinfbb[8][7][1] = 1.;
zwinfbb[8][7][0] = 1.;

zwinfbb[8][6][5] = 10.;
zwinfbb[8][6][4] = 10.;
zwinfbb[8][6][3] = 10.;
zwinfbb[8][6][2] = 10.;
zwinfbb[8][6][1] = 10.;
zwinfbb[8][6][0] = 10.;

zwinfbb[8][5][4] = 10.;
zwinfbb[8][5][3] = 10.;
zwinfbb[8][5][2] = 10.;
zwinfbb[8][5][1] = 10.;
zwinfbb[8][5][0] = 10.;

zwinfbb[8][4][3] = 10.;
zwinfbb[8][4][2] = 10.;
zwinfbb[8][4][1] = 10.;
zwinfbb[8][4][0] = 10.;

zwinfbb[8][3][2] = 10.;
zwinfbb[8][3][1] = 10.;
zwinfbb[8][3][0] = 10.;

zwinfbb[8][2][1] = 1.;
zwinfbb[8][2][0] = 1.;

zwinfbb[8][1][0] = 0.5;

// ++++++++++++Last layer = 7

zwinfbb[7][12][11] = 10.;
zwinfbb[7][12][10] = 10.;
zwinfbb[7][12][9] = 10.;
zwinfbb[7][12][8] = 10.;
zwinfbb[7][12][7] = 10.;
zwinfbb[7][12][6] = 10.;
zwinfbb[7][12][5] = 10.;
zwinfbb[7][12][4] = 10.;
zwinfbb[7][12][3] = 10.;
zwinfbb[7][12][2] = 10.;
zwinfbb[7][12][1] = 10.;
zwinfbb[7][12][0] = 10.;

zwinfbb[7][11][10] = 10.;
zwinfbb[7][11][9] = 10.;
zwinfbb[7][11][8] = 10.;
zwinfbb[7][11][7] = 10.;
zwinfbb[7][11][6] = 10.;
zwinfbb[7][11][5] = 10.;
zwinfbb[7][11][4] = 10.;
zwinfbb[7][11][3] = 10.;
zwinfbb[7][11][2] = 10.;
zwinfbb[7][11][1] = 10.;
zwinfbb[7][11][0] = 10.;

zwinfbb[7][10][9] = 10.;
zwinfbb[7][10][8] = 10.;
zwinfbb[7][10][7] = 10.;
zwinfbb[7][10][6] = 10.;
zwinfbb[7][10][5] = 10.;
zwinfbb[7][10][4] = 10.;
zwinfbb[7][10][3] = 10.;
zwinfbb[7][10][2] = 10.;
zwinfbb[7][10][1] = 10.;
zwinfbb[7][10][0] = 10.;

zwinfbb[7][9][8] = 10.;
zwinfbb[7][9][7] = 10.;
zwinfbb[7][9][6] = 10.;
zwinfbb[7][9][5] = 10.;
zwinfbb[7][9][4] = 10.;
zwinfbb[7][9][3] = 10.;
zwinfbb[7][9][2] = 10.;
zwinfbb[7][9][1] = 10.;
zwinfbb[7][9][0] = 10.;


zwinfbb[7][8][7] = 10.;
zwinfbb[7][8][6] = 10.;
zwinfbb[7][8][5] = 10.;
zwinfbb[7][8][4] = 10.;
zwinfbb[7][8][3] = 10.;
zwinfbb[7][8][2] = 1.;
zwinfbb[7][8][1] = 1.;
zwinfbb[7][8][0] = 1.;


zwinfbb[7][7][6] = 10.;
zwinfbb[7][7][5] = 10.;
zwinfbb[7][7][4] = 10.;
zwinfbb[7][7][3] = 10.;
zwinfbb[7][7][2] = 10.;
zwinfbb[7][7][1] = 10.;
zwinfbb[7][7][0] = 10.;

zwinfbb[7][6][5] = 10.;
zwinfbb[7][6][4] = 10.;
zwinfbb[7][6][3] = 10.;
zwinfbb[7][6][2] = 10.;
zwinfbb[7][6][1] = 10.;
zwinfbb[7][6][0] = 10.;

zwinfbb[7][5][4] = 10.;
zwinfbb[7][5][3] = 10.;
zwinfbb[7][5][2] = 10.;
zwinfbb[7][5][1] = 10.;
zwinfbb[7][5][0] = 10.;

zwinfbb[7][4][3] = 10.;
zwinfbb[7][4][2] = 10.;
zwinfbb[7][4][1] = 10.;
zwinfbb[7][4][0] = 10.;

zwinfbb[7][3][2] = 10.;
zwinfbb[7][3][1] = 10.;
zwinfbb[7][3][0] = 10.;

zwinfbb[7][2][1] = 1.;
zwinfbb[7][2][0] = 1.;

zwinfbb[7][1][0] = 0.5;

// ++++++++++++Last layer = 6

zwinfbb[6][12][11] = 10.;
zwinfbb[6][12][10] = 10.;
zwinfbb[6][12][9] = 10.;
zwinfbb[6][12][8] = 10.;
zwinfbb[6][12][7] = 10.;
zwinfbb[6][12][6] = 10.;
zwinfbb[6][12][5] = 10.;
zwinfbb[6][12][4] = 10.;
zwinfbb[6][12][3] = 10.;
zwinfbb[6][12][2] = 10.;
zwinfbb[6][12][1] = 10.;
zwinfbb[6][12][0] = 10.;

zwinfbb[6][11][10] = 10.;
zwinfbb[6][11][9] = 10.;
zwinfbb[6][11][8] = 10.;
zwinfbb[6][11][7] = 10.;
zwinfbb[6][11][6] = 10.;
zwinfbb[6][11][5] = 10.;
zwinfbb[6][11][4] = 10.;
zwinfbb[6][11][3] = 10.;
zwinfbb[6][11][2] = 10.;
zwinfbb[6][11][1] = 10.;
zwinfbb[6][11][0] = 10.;

zwinfbb[6][10][9] = 10.;
zwinfbb[6][10][8] = 10.;
zwinfbb[6][10][7] = 10.;
zwinfbb[6][10][6] = 10.;
zwinfbb[6][10][5] = 10.;
zwinfbb[6][10][4] = 10.;
zwinfbb[6][10][3] = 10.;
zwinfbb[6][10][2] = 10.;
zwinfbb[6][10][1] = 10.;
zwinfbb[6][10][0] = 10.;

zwinfbb[6][9][8] = 10.;
zwinfbb[6][9][7] = 10.;
zwinfbb[6][9][6] = 10.;
zwinfbb[6][9][5] = 10.;
zwinfbb[6][9][4] = 10.;
zwinfbb[6][9][3] = 10.;
zwinfbb[6][9][2] = 10.;
zwinfbb[6][9][1] = 10.;
zwinfbb[6][9][0] = 10.;


zwinfbb[6][8][7] = 10.;
zwinfbb[6][8][6] = 10.;
zwinfbb[6][8][5] = 10.;
zwinfbb[6][8][4] = 10.;
zwinfbb[6][8][3] = 10.;
zwinfbb[6][8][2] = 1.;
zwinfbb[6][8][1] = 1.;
zwinfbb[6][8][0] = 1.;


zwinfbb[6][7][6] = 10.;
zwinfbb[6][7][5] = 10.;
zwinfbb[6][7][4] = 10.;
zwinfbb[6][7][3] = 10.;
zwinfbb[6][7][2] = 10.;
zwinfbb[6][7][1] = 10.;
zwinfbb[6][7][0] = 10.;

zwinfbb[6][6][5] = 10.;
zwinfbb[6][6][4] = 10.;
zwinfbb[6][6][3] = 10.;
zwinfbb[6][6][2] = 10.;
zwinfbb[6][6][1] = 10.;
zwinfbb[6][6][0] = 10.;

zwinfbb[6][5][4] = 10.;
zwinfbb[6][5][3] = 10.;
zwinfbb[6][5][2] = 10.;
zwinfbb[6][5][1] = 10.;
zwinfbb[6][5][0] = 10.;

zwinfbb[6][4][3] = 10.;
zwinfbb[6][4][2] = 10.;
zwinfbb[6][4][1] = 10.;
zwinfbb[6][4][0] = 10.;

zwinfbb[6][3][2] = 10.;
zwinfbb[6][3][1] = 10.;
zwinfbb[6][3][0] = 10.;

zwinfbb[6][2][1] = 1.;
zwinfbb[6][2][0] = 1.;

zwinfbb[6][1][0] = 0.5;

// ++++++++++++Last layer = 5

zwinfbb[5][12][11] = 10.;
zwinfbb[5][12][10] = 10.;
zwinfbb[5][12][9] = 10.;
zwinfbb[5][12][8] = 10.;
zwinfbb[5][12][7] = 10.;
zwinfbb[5][12][6] = 10.;
zwinfbb[5][12][5] = 10.;
zwinfbb[5][12][4] = 10.;
zwinfbb[5][12][3] = 10.;
zwinfbb[5][12][2] = 10.;
zwinfbb[5][12][1] = 10.;
zwinfbb[5][12][0] = 10.;

zwinfbb[5][11][10] = 10.;
zwinfbb[5][11][9] = 10.;
zwinfbb[5][11][8] = 10.;
zwinfbb[5][11][7] = 10.;
zwinfbb[5][11][6] = 10.;
zwinfbb[5][11][5] = 10.;
zwinfbb[5][11][4] = 10.;
zwinfbb[5][11][3] = 10.;
zwinfbb[5][11][2] = 10.;
zwinfbb[5][11][1] = 10.;
zwinfbb[5][11][0] = 10.;

zwinfbb[5][10][9] = 10.;
zwinfbb[5][10][8] = 10.;
zwinfbb[5][10][7] = 10.;
zwinfbb[5][10][6] = 10.;
zwinfbb[5][10][5] = 10.;
zwinfbb[5][10][4] = 10.;
zwinfbb[5][10][3] = 10.;
zwinfbb[5][10][2] = 10.;
zwinfbb[5][10][1] = 10.;
zwinfbb[5][10][0] = 10.;

zwinfbb[5][9][8] = 10.;
zwinfbb[5][9][7] = 10.;
zwinfbb[5][9][6] = 10.;
zwinfbb[5][9][5] = 10.;
zwinfbb[5][9][4] = 10.;
zwinfbb[5][9][3] = 10.;
zwinfbb[5][9][2] = 10.;
zwinfbb[5][9][1] = 10.;
zwinfbb[5][9][0] = 10.;


zwinfbb[5][8][7] = 10.;
zwinfbb[5][8][6] = 10.;
zwinfbb[5][8][5] = 10.;
zwinfbb[5][8][4] = 10.;
zwinfbb[5][8][3] = 10.;
zwinfbb[5][8][2] = 1.;
zwinfbb[5][8][1] = 1.;
zwinfbb[5][8][0] = 1.;


zwinfbb[5][7][6] = 10.;
zwinfbb[5][7][5] = 10.;
zwinfbb[5][7][4] = 10.;
zwinfbb[5][7][3] = 10.;
zwinfbb[5][7][2] = 10.;
zwinfbb[5][7][1] = 10.;
zwinfbb[5][7][0] = 10.;

zwinfbb[5][6][5] = 10.;
zwinfbb[5][6][4] = 10.;
zwinfbb[5][6][3] = 10.;
zwinfbb[5][6][2] = 10.;
zwinfbb[5][6][1] = 10.;
zwinfbb[5][6][0] = 10.;

zwinfbb[5][5][4] = 10.;
zwinfbb[5][5][3] = 10.;
zwinfbb[5][5][2] = 10.;
zwinfbb[5][5][1] = 10.;
zwinfbb[5][5][0] = 10.;

zwinfbb[5][4][3] = 10.;
zwinfbb[5][4][2] = 10.;
zwinfbb[5][4][1] = 10.;
zwinfbb[5][4][0] = 10.;

zwinfbb[5][3][2] = 10.;
zwinfbb[5][3][1] = 10.;
zwinfbb[5][3][0] = 10.;

zwinfbb[5][2][1] = 1.;
zwinfbb[5][2][0] = 1.;

zwinfbb[5][1][0] = 0.5;

// =============== forward z, zcutfrw

// ++++++++++++Last layer = 13

zcutfbb[13][12][11] = 15.;
zcutfbb[13][12][10] = 10.;
zcutfbb[13][12][9] = 10.;
zcutfbb[13][12][8] = 10.;
zcutfbb[13][12][7] = 10.;
zcutfbb[13][12][6] = 10.;
zcutfbb[13][12][5] = 10.;
zcutfbb[13][12][4] = 10.;
zcutfbb[13][12][3] = 10.;
zcutfbb[13][12][2] = 10.;
zcutfbb[13][12][1] = 10.;
zcutfbb[13][12][0] = 10.;

zcutfbb[13][11][10] = 10.;
zcutfbb[13][11][9] = 10.;
zcutfbb[13][11][8] = 10.;
zcutfbb[13][11][7] = 10.;
zcutfbb[13][11][6] = 10.;
zcutfbb[13][11][5] = 10.;
zcutfbb[13][11][4] = 10.;
zcutfbb[13][11][3] = 10.;
zcutfbb[13][11][2] = 10.;
zcutfbb[13][11][1] = 10.;
zcutfbb[13][11][0] = 10.;

zcutfbb[13][10][9] = 10.;
zcutfbb[13][10][8] = 10.;
zcutfbb[13][10][7] = 10.;
zcutfbb[13][10][6] = 10.;
zcutfbb[13][10][5] = 10.;
zcutfbb[13][10][4] = 10.;
zcutfbb[13][10][3] = 10.;
zcutfbb[13][10][2] = 10.;
zcutfbb[13][10][1] = 10.;
zcutfbb[13][10][0] = 10.;

zcutfbb[13][9][8] = 10.;
zcutfbb[13][9][7] = 10.;
zcutfbb[13][9][6] = 10.;
zcutfbb[13][9][5] = 10.;
zcutfbb[13][9][4] = 10.;
zcutfbb[13][9][3] = 10.;
zcutfbb[13][9][2] = 10.;
zcutfbb[13][9][1] = 10.;
zcutfbb[13][9][0] = 10.;


zcutfbb[13][8][7] = 10.;
zcutfbb[13][8][6] = 10.;
zcutfbb[13][8][5] = 10.;
zcutfbb[13][8][4] = 10.;
zcutfbb[13][8][3] = 10.;
zcutfbb[13][8][2] = 10.;
zcutfbb[13][8][1] = 10.;
zcutfbb[13][8][0] = 10.;


zcutfbb[13][7][6] = 10.;
zcutfbb[13][7][5] = 10.;
zcutfbb[13][7][4] = 10.;
zcutfbb[13][7][3] = 10.;
zcutfbb[13][7][2] = 10.;
zcutfbb[13][7][1] = 10.;
zcutfbb[13][7][0] = 10.;

zcutfbb[13][6][5] = 10.;
zcutfbb[13][6][4] = 10.;
zcutfbb[13][6][3] = 10.;
zcutfbb[13][6][2] = 10.;
zcutfbb[13][6][1] = 10.;
zcutfbb[13][6][0] = 10.;

zcutfbb[13][5][4] = 10.;
zcutfbb[13][5][3] = 10.;
zcutfbb[13][5][2] = 10.;
zcutfbb[13][5][1] = 10.;
zcutfbb[13][5][0] = 10.;

zcutfbb[13][4][3] = 10.;
zcutfbb[13][4][2] = 10.;
zcutfbb[13][4][1] = 10.;
zcutfbb[13][4][0] = 10.;

zcutfbb[13][3][2] = 10.;
zcutfbb[13][3][1] = 10.;
zcutfbb[13][3][0] = 10.;

zcutfbb[13][2][1] = 1.;
zcutfbb[13][2][0] = 1.;

zcutfbb[13][1][0] = 1.;


// ++++++++++++Last layer = 12

zcutfbb[12][12][11] = 10.;
zcutfbb[12][12][10] = 10.;
zcutfbb[12][12][9] = 10.;
zcutfbb[12][12][8] = 10.;
zcutfbb[12][12][7] = 10.;
zcutfbb[12][12][6] = 10.;
zcutfbb[12][12][5] = 10.;
zcutfbb[12][12][4] = 10.;
zcutfbb[12][12][3] = 10.;
zcutfbb[12][12][2] = 10.;
zcutfbb[12][12][1] = 10.;
zcutfbb[12][12][0] = 10.;

zcutfbb[12][11][10] = 10.;
zcutfbb[12][11][9] = 10.;
zcutfbb[12][11][8] = 10.;
zcutfbb[12][11][7] = 10.;
zcutfbb[12][11][6] = 10.;
zcutfbb[12][11][5] = 10.;
zcutfbb[12][11][4] = 10.;
zcutfbb[12][11][3] = 10.;
zcutfbb[12][11][2] = 10.;
zcutfbb[12][11][1] = 10.;
zcutfbb[12][11][0] = 10.;

zcutfbb[12][10][9] = 10.;
zcutfbb[12][10][8] = 10.;
zcutfbb[12][10][7] = 10.;
zcutfbb[12][10][6] = 10.;
zcutfbb[12][10][5] = 10.;
zcutfbb[12][10][4] = 10.;
zcutfbb[12][10][3] = 10.;
zcutfbb[12][10][2] = 10.;
zcutfbb[12][10][1] = 10.;
zcutfbb[12][10][0] = 10.;

zcutfbb[12][9][8] = 10.;
zcutfbb[12][9][7] = 10.;
zcutfbb[12][9][6] = 10.;
zcutfbb[12][9][5] = 10.;
zcutfbb[12][9][4] = 10.;
zcutfbb[12][9][3] = 10.;
zcutfbb[12][9][2] = 10.;
zcutfbb[12][9][1] = 10.;
zcutfbb[12][9][0] = 10.;


zcutfbb[12][8][7] = 10.;
zcutfbb[12][8][6] = 10.;
zcutfbb[12][8][5] = 10.;
zcutfbb[12][8][4] = 10.;
zcutfbb[12][8][3] = 10.;
zcutfbb[12][8][2] = 10.;
zcutfbb[12][8][1] = 10.;
zcutfbb[12][8][0] = 10.;


zcutfbb[12][7][6] = 10.;
zcutfbb[12][7][5] = 10.;
zcutfbb[12][7][4] = 10.;
zcutfbb[12][7][3] = 10.;
zcutfbb[12][7][2] = 10.;
zcutfbb[12][7][1] = 10.;
zcutfbb[12][7][0] = 10.;

zcutfbb[12][6][5] = 10.;
zcutfbb[12][6][4] = 10.;
zcutfbb[12][6][3] = 10.;
zcutfbb[12][6][2] = 10.;
zcutfbb[12][6][1] = 10.;
zcutfbb[12][6][0] = 10.;

zcutfbb[12][5][4] = 10.;
zcutfbb[12][5][3] = 10.;
zcutfbb[12][5][2] = 10.;
zcutfbb[12][5][1] = 10.;
zcutfbb[12][5][0] = 10.;

zcutfbb[12][4][3] = 10.;
zcutfbb[12][4][2] = 10.;
zcutfbb[12][4][1] = 10.;
zcutfbb[12][4][0] = 10.;

zcutfbb[12][3][2] = 10.;
zcutfbb[12][3][1] = 10.;
zcutfbb[12][3][0] = 10.;

zcutfbb[12][2][1] = 1.;
zcutfbb[12][2][0] = 1.;

zcutfbb[12][1][0] = 1.;

// ++++++++++++Last layer = 11

zcutfbb[11][12][11] = 10.;
zcutfbb[11][12][10] = 10.;
zcutfbb[11][12][9] = 10.;
zcutfbb[11][12][8] = 10.;
zcutfbb[11][12][7] = 10.;
zcutfbb[11][12][6] = 10.;
zcutfbb[11][12][5] = 10.;
zcutfbb[11][12][4] = 10.;
zcutfbb[11][12][3] = 10.;
zcutfbb[11][12][2] = 10.;
zcutfbb[11][12][1] = 10.;
zcutfbb[11][12][0] = 10.;

zcutfbb[11][11][10] = 10.;
zcutfbb[11][11][9] = 10.;
zcutfbb[11][11][8] = 10.;
zcutfbb[11][11][7] = 10.;
zcutfbb[11][11][6] = 10.;
zcutfbb[11][11][5] = 10.;
zcutfbb[11][11][4] = 10.;
zcutfbb[11][11][3] = 10.;
zcutfbb[11][11][2] = 10.;
zcutfbb[11][11][1] = 10.;
zcutfbb[11][11][0] = 10.;

zcutfbb[11][10][9] = 10.;
zcutfbb[11][10][8] = 10.;
zcutfbb[11][10][7] = 10.;
zcutfbb[11][10][6] = 10.;
zcutfbb[11][10][5] = 10.;
zcutfbb[11][10][4] = 10.;
zcutfbb[11][10][3] = 10.;
zcutfbb[11][10][2] = 10.;
zcutfbb[11][10][1] = 10.;
zcutfbb[11][10][0] = 10.;

zcutfbb[11][9][8] = 10.;
zcutfbb[11][9][7] = 10.;
zcutfbb[11][9][6] = 10.;
zcutfbb[11][9][5] = 10.;
zcutfbb[11][9][4] = 10.;
zcutfbb[11][9][3] = 10.;
zcutfbb[11][9][2] = 10.;
zcutfbb[11][9][1] = 10.;
zcutfbb[11][9][0] = 10.;


zcutfbb[11][8][7] = 10.;
zcutfbb[11][8][6] = 10.;
zcutfbb[11][8][5] = 10.;
zcutfbb[11][8][4] = 10.;
zcutfbb[11][8][3] = 10.;
zcutfbb[11][8][2] = 10.;
zcutfbb[11][8][1] = 10.;
zcutfbb[11][8][0] = 10.;


zcutfbb[11][7][6] = 10.;
zcutfbb[11][7][5] = 10.;
zcutfbb[11][7][4] = 10.;
zcutfbb[11][7][3] = 10.;
zcutfbb[11][7][2] = 10.;
zcutfbb[11][7][1] = 10.;
zcutfbb[11][7][0] = 10.;

zcutfbb[11][6][5] = 10.;
zcutfbb[11][6][4] = 10.;
zcutfbb[11][6][3] = 10.;
zcutfbb[11][6][2] = 10.;
zcutfbb[11][6][1] = 10.;
zcutfbb[11][6][0] = 10.;

zcutfbb[11][5][4] = 10.;
zcutfbb[11][5][3] = 10.;
zcutfbb[11][5][2] = 10.;
zcutfbb[11][5][1] = 10.;
zcutfbb[11][5][0] = 10.;

zcutfbb[11][4][3] = 10.;
zcutfbb[11][4][2] = 10.;
zcutfbb[11][4][1] = 10.;
zcutfbb[11][4][0] = 10.;

zcutfbb[11][3][2] = 10.;
zcutfbb[11][3][1] = 10.;
zcutfbb[11][3][0] = 10.;

zcutfbb[11][2][1] = 1.;
zcutfbb[11][2][0] = 1.;

zcutfbb[11][1][0] = 1.;


// ++++++++++++Last layer = 10

zcutfbb[10][12][11] = 10.;
zcutfbb[10][12][10] = 10.;
zcutfbb[10][12][9] = 10.;
zcutfbb[10][12][8] = 10.;
zcutfbb[10][12][7] = 10.;
zcutfbb[10][12][6] = 10.;
zcutfbb[10][12][5] = 10.;
zcutfbb[10][12][4] = 10.;
zcutfbb[10][12][3] = 10.;
zcutfbb[10][12][2] = 10.;
zcutfbb[10][12][1] = 10.;
zcutfbb[10][12][0] = 10.;

zcutfbb[10][11][10] = 10.;
zcutfbb[10][11][9] = 10.;
zcutfbb[10][11][8] = 10.;
zcutfbb[10][11][7] = 10.;
zcutfbb[10][11][6] = 10.;
zcutfbb[10][11][5] = 10.;
zcutfbb[10][11][4] = 10.;
zcutfbb[10][11][3] = 10.;
zcutfbb[10][11][2] = 10.;
zcutfbb[10][11][1] = 10.;
zcutfbb[10][11][0] = 10.;

zcutfbb[10][10][9] = 10.;
zcutfbb[10][10][8] = 10.;
zcutfbb[10][10][7] = 10.;
zcutfbb[10][10][6] = 10.;
zcutfbb[10][10][5] = 10.;
zcutfbb[10][10][4] = 10.;
zcutfbb[10][10][3] = 10.;
zcutfbb[10][10][2] = 10.;
zcutfbb[10][10][1] = 10.;
zcutfbb[10][10][0] = 10.;

zcutfbb[10][9][8] = 10.;
zcutfbb[10][9][7] = 10.;
zcutfbb[10][9][6] = 10.;
zcutfbb[10][9][5] = 10.;
zcutfbb[10][9][4] = 10.;
zcutfbb[10][9][3] = 10.;
zcutfbb[10][9][2] = 10.;
zcutfbb[10][9][1] = 10.;
zcutfbb[10][9][0] = 10.;


zcutfbb[10][8][7] = 10.;
zcutfbb[10][8][6] = 10.;
zcutfbb[10][8][5] = 10.;
zcutfbb[10][8][4] = 10.;
zcutfbb[10][8][3] = 10.;
zcutfbb[10][8][2] = 10.;
zcutfbb[10][8][1] = 10.;
zcutfbb[10][8][0] = 10.;


zcutfbb[10][7][6] = 10.;
zcutfbb[10][7][5] = 10.;
zcutfbb[10][7][4] = 10.;
zcutfbb[10][7][3] = 10.;
zcutfbb[10][7][2] = 10.;
zcutfbb[10][7][1] = 10.;
zcutfbb[10][7][0] = 10.;

zcutfbb[10][6][5] = 10.;
zcutfbb[10][6][4] = 10.;
zcutfbb[10][6][3] = 10.;
zcutfbb[10][6][2] = 10.;
zcutfbb[10][6][1] = 10.;
zcutfbb[10][6][0] = 10.;

zcutfbb[10][5][4] = 10.;
zcutfbb[10][5][3] = 10.;
zcutfbb[10][5][2] = 10.;
zcutfbb[10][5][1] = 10.;
zcutfbb[10][5][0] = 10.;

zcutfbb[10][4][3] = 10.;
zcutfbb[10][4][2] = 10.;
zcutfbb[10][4][1] = 10.;
zcutfbb[10][4][0] = 10.;

zcutfbb[10][3][2] = 10.;
zcutfbb[10][3][1] = 10.;
zcutfbb[10][3][0] = 10.;

zcutfbb[10][2][1] = 1.;
zcutfbb[10][2][0] = 1.;

zcutfbb[10][1][0] = 1.;

// ++++++++++++Last layer = 9

zcutfbb[9][12][11] = 10.;
zcutfbb[9][12][10] = 10.;
zcutfbb[9][12][9] = 10.;
zcutfbb[9][12][8] = 10.;
zcutfbb[9][12][7] = 10.;
zcutfbb[9][12][6] = 10.;
zcutfbb[9][12][5] = 10.;
zcutfbb[9][12][4] = 10.;
zcutfbb[9][12][3] = 10.;
zcutfbb[9][12][2] = 10.;
zcutfbb[9][12][1] = 10.;
zcutfbb[9][12][0] = 10.;

zcutfbb[9][11][10] = 10.;
zcutfbb[9][11][9] = 10.;
zcutfbb[9][11][8] = 10.;
zcutfbb[9][11][7] = 10.;
zcutfbb[9][11][6] = 10.;
zcutfbb[9][11][5] = 10.;
zcutfbb[9][11][4] = 10.;
zcutfbb[9][11][3] = 10.;
zcutfbb[9][11][2] = 10.;
zcutfbb[9][11][1] = 10.;
zcutfbb[9][11][0] = 10.;

zcutfbb[9][10][9] = 10.;
zcutfbb[9][10][8] = 10.;
zcutfbb[9][10][7] = 10.;
zcutfbb[9][10][6] = 10.;
zcutfbb[9][10][5] = 10.;
zcutfbb[9][10][4] = 10.;
zcutfbb[9][10][3] = 10.;
zcutfbb[9][10][2] = 10.;
zcutfbb[9][10][1] = 10.;
zcutfbb[9][10][0] = 10.;

zcutfbb[9][9][8] = 10.;
zcutfbb[9][9][7] = 10.;
zcutfbb[9][9][6] = 10.;
zcutfbb[9][9][5] = 10.;
zcutfbb[9][9][4] = 10.;
zcutfbb[9][9][3] = 10.;
zcutfbb[9][9][2] = 10.;
zcutfbb[9][9][1] = 10.;
zcutfbb[9][9][0] = 10.;


zcutfbb[9][8][7] = 10.;
zcutfbb[9][8][6] = 10.;
zcutfbb[9][8][5] = 10.;
zcutfbb[9][8][4] = 10.;
zcutfbb[9][8][3] = 10.;
zcutfbb[9][8][2] = 10.;
zcutfbb[9][8][1] = 10.;
zcutfbb[9][8][0] = 10.;


zcutfbb[9][7][6] = 10.;
zcutfbb[9][7][5] = 10.;
zcutfbb[9][7][4] = 10.;
zcutfbb[9][7][3] = 10.;
zcutfbb[9][7][2] = 10.;
zcutfbb[9][7][1] = 10.;
zcutfbb[9][7][0] = 10.;

zcutfbb[9][6][5] = 10.;
zcutfbb[9][6][4] = 10.;
zcutfbb[9][6][3] = 10.;
zcutfbb[9][6][2] = 10.;
zcutfbb[9][6][1] = 10.;
zcutfbb[9][6][0] = 10.;

zcutfbb[9][5][4] = 10.;
zcutfbb[9][5][3] = 10.;
zcutfbb[9][5][2] = 10.;
zcutfbb[9][5][1] = 10.;
zcutfbb[9][5][0] = 10.;

zcutfbb[9][4][3] = 10.;
zcutfbb[9][4][2] = 10.;
zcutfbb[9][4][1] = 10.;
zcutfbb[9][4][0] = 10.;

zcutfbb[9][3][2] = 10.;
zcutfbb[9][3][1] = 10.;
zcutfbb[9][3][0] = 10.;

zcutfbb[9][2][1] = 1.;
zcutfbb[9][2][0] = 1.;

zcutfbb[9][1][0] = 1.;

// ++++++++++++Last layer = 8

zcutfbb[8][12][11] = 10.;
zcutfbb[8][12][10] = 10.;
zcutfbb[8][12][9] = 10.;
zcutfbb[8][12][8] = 10.;
zcutfbb[8][12][7] = 10.;
zcutfbb[8][12][6] = 10.;
zcutfbb[8][12][5] = 10.;
zcutfbb[8][12][4] = 10.;
zcutfbb[8][12][3] = 10.;
zcutfbb[8][12][2] = 10.;
zcutfbb[8][12][1] = 10.;
zcutfbb[8][12][0] = 10.;

zcutfbb[8][11][10] = 10.;
zcutfbb[8][11][9] = 10.;
zcutfbb[8][11][8] = 10.;
zcutfbb[8][11][7] = 10.;
zcutfbb[8][11][6] = 10.;
zcutfbb[8][11][5] = 10.;
zcutfbb[8][11][4] = 10.;
zcutfbb[8][11][3] = 10.;
zcutfbb[8][11][2] = 10.;
zcutfbb[8][11][1] = 10.;
zcutfbb[8][11][0] = 10.;

zcutfbb[8][10][9] = 10.;
zcutfbb[8][10][8] = 10.;
zcutfbb[8][10][7] = 10.;
zcutfbb[8][10][6] = 10.;
zcutfbb[8][10][5] = 10.;
zcutfbb[8][10][4] = 10.;
zcutfbb[8][10][3] = 10.;
zcutfbb[8][10][2] = 10.;
zcutfbb[8][10][1] = 10.;
zcutfbb[8][10][0] = 10.;

zcutfbb[8][9][8] = 10.;
zcutfbb[8][9][7] = 10.;
zcutfbb[8][9][6] = 10.;
zcutfbb[8][9][5] = 10.;
zcutfbb[8][9][4] = 10.;
zcutfbb[8][9][3] = 10.;
zcutfbb[8][9][2] = 10.;
zcutfbb[8][9][1] = 10.;
zcutfbb[8][9][0] = 10.;


zcutfbb[8][8][7] = 10.;
zcutfbb[8][8][6] = 10.;
zcutfbb[8][8][5] = 10.;
zcutfbb[8][8][4] = 10.;
zcutfbb[8][8][3] = 10.;
zcutfbb[8][8][2] = 10.;
zcutfbb[8][8][1] = 10.;
zcutfbb[8][8][0] = 10.;


zcutfbb[8][7][6] = 10.;
zcutfbb[8][7][5] = 10.;
zcutfbb[8][7][4] = 10.;
zcutfbb[8][7][3] = 10.;
zcutfbb[8][7][2] = 10.;
zcutfbb[8][7][1] = 10.;
zcutfbb[8][7][0] = 10.;

zcutfbb[8][6][5] = 10.;
zcutfbb[8][6][4] = 10.;
zcutfbb[8][6][3] = 10.;
zcutfbb[8][6][2] = 10.;
zcutfbb[8][6][1] = 10.;
zcutfbb[8][6][0] = 10.;

zcutfbb[8][5][4] = 10.;
zcutfbb[8][5][3] = 10.;
zcutfbb[8][5][2] = 10.;
zcutfbb[8][5][1] = 10.;
zcutfbb[8][5][0] = 10.;

zcutfbb[8][4][3] = 10.;
zcutfbb[8][4][2] = 10.;
zcutfbb[8][4][1] = 10.;
zcutfbb[8][4][0] = 10.;

zcutfbb[8][3][2] = 10.;
zcutfbb[8][3][1] = 10.;
zcutfbb[8][3][0] = 10.;

zcutfbb[8][2][1] = 1.;
zcutfbb[8][2][0] = 1.;

zcutfbb[8][1][0] = 0.5;

// ++++++++++++Last layer = 7

zcutfbb[7][12][11] = 10.;
zcutfbb[7][12][10] = 10.;
zcutfbb[7][12][9] = 10.;
zcutfbb[7][12][8] = 10.;
zcutfbb[7][12][7] = 10.;
zcutfbb[7][12][6] = 10.;
zcutfbb[7][12][5] = 10.;
zcutfbb[7][12][4] = 10.;
zcutfbb[7][12][3] = 10.;
zcutfbb[7][12][2] = 10.;
zcutfbb[7][12][1] = 10.;
zcutfbb[7][12][0] = 10.;

zcutfbb[7][11][10] = 10.;
zcutfbb[7][11][9] = 10.;
zcutfbb[7][11][8] = 10.;
zcutfbb[7][11][7] = 10.;
zcutfbb[7][11][6] = 10.;
zcutfbb[7][11][5] = 10.;
zcutfbb[7][11][4] = 10.;
zcutfbb[7][11][3] = 10.;
zcutfbb[7][11][2] = 10.;
zcutfbb[7][11][1] = 10.;
zcutfbb[7][11][0] = 10.;

zcutfbb[7][10][9] = 10.;
zcutfbb[7][10][8] = 10.;
zcutfbb[7][10][7] = 10.;
zcutfbb[7][10][6] = 10.;
zcutfbb[7][10][5] = 10.;
zcutfbb[7][10][4] = 10.;
zcutfbb[7][10][3] = 10.;
zcutfbb[7][10][2] = 10.;
zcutfbb[7][10][1] = 10.;
zcutfbb[7][10][0] = 10.;

zcutfbb[7][9][8] = 10.;
zcutfbb[7][9][7] = 10.;
zcutfbb[7][9][6] = 10.;
zcutfbb[7][9][5] = 10.;
zcutfbb[7][9][4] = 10.;
zcutfbb[7][9][3] = 10.;
zcutfbb[7][9][2] = 10.;
zcutfbb[7][9][1] = 10.;
zcutfbb[7][9][0] = 10.;


zcutfbb[7][8][7] = 10.;
zcutfbb[7][8][6] = 10.;
zcutfbb[7][8][5] = 10.;
zcutfbb[7][8][4] = 10.;
zcutfbb[7][8][3] = 10.;
zcutfbb[7][8][2] = 10.;
zcutfbb[7][8][1] = 10.;
zcutfbb[7][8][0] = 10.;


zcutfbb[7][7][6] = 10.;
zcutfbb[7][7][5] = 10.;
zcutfbb[7][7][4] = 10.;
zcutfbb[7][7][3] = 10.;
zcutfbb[7][7][2] = 10.;
zcutfbb[7][7][1] = 10.;
zcutfbb[7][7][0] = 10.;

zcutfbb[7][6][5] = 10.;
zcutfbb[7][6][4] = 10.;
zcutfbb[7][6][3] = 10.;
zcutfbb[7][6][2] = 10.;
zcutfbb[7][6][1] = 10.;
zcutfbb[7][6][0] = 10.;

zcutfbb[7][5][4] = 10.;
zcutfbb[7][5][3] = 10.;
zcutfbb[7][5][2] = 10.;
zcutfbb[7][5][1] = 10.;
zcutfbb[7][5][0] = 10.;

zcutfbb[7][4][3] = 10.;
zcutfbb[7][4][2] = 10.;
zcutfbb[7][4][1] = 10.;
zcutfbb[7][4][0] = 10.;

zcutfbb[7][3][2] = 10.;
zcutfbb[7][3][1] = 10.;
zcutfbb[7][3][0] = 10.;

zcutfbb[7][2][1] = 1.;
zcutfbb[7][2][0] = 1.;

zcutfbb[7][1][0] = 0.5;

// ++++++++++++Last layer = 6

zcutfbb[6][12][11] = 10.;
zcutfbb[6][12][10] = 10.;
zcutfbb[6][12][9] = 10.;
zcutfbb[6][12][8] = 10.;
zcutfbb[6][12][7] = 10.;
zcutfbb[6][12][6] = 10.;
zcutfbb[6][12][5] = 10.;
zcutfbb[6][12][4] = 10.;
zcutfbb[6][12][3] = 10.;
zcutfbb[6][12][2] = 10.;
zcutfbb[6][12][1] = 10.;
zcutfbb[6][12][0] = 10.;

zcutfbb[6][11][10] = 10.;
zcutfbb[6][11][9] = 10.;
zcutfbb[6][11][8] = 10.;
zcutfbb[6][11][7] = 10.;
zcutfbb[6][11][6] = 10.;
zcutfbb[6][11][5] = 10.;
zcutfbb[6][11][4] = 10.;
zcutfbb[6][11][3] = 10.;
zcutfbb[6][11][2] = 10.;
zcutfbb[6][11][1] = 10.;
zcutfbb[6][11][0] = 10.;

zcutfbb[6][10][9] = 10.;
zcutfbb[6][10][8] = 10.;
zcutfbb[6][10][7] = 10.;
zcutfbb[6][10][6] = 10.;
zcutfbb[6][10][5] = 10.;
zcutfbb[6][10][4] = 10.;
zcutfbb[6][10][3] = 10.;
zcutfbb[6][10][2] = 10.;
zcutfbb[6][10][1] = 10.;
zcutfbb[6][10][0] = 10.;

zcutfbb[6][9][8] = 10.;
zcutfbb[6][9][7] = 10.;
zcutfbb[6][9][6] = 10.;
zcutfbb[6][9][5] = 10.;
zcutfbb[6][9][4] = 10.;
zcutfbb[6][9][3] = 10.;
zcutfbb[6][9][2] = 10.;
zcutfbb[6][9][1] = 10.;
zcutfbb[6][9][0] = 10.;


zcutfbb[6][8][7] = 10.;
zcutfbb[6][8][6] = 10.;
zcutfbb[6][8][5] = 10.;
zcutfbb[6][8][4] = 10.;
zcutfbb[6][8][3] = 10.;
zcutfbb[6][8][2] = 10.;
zcutfbb[6][8][1] = 10.;
zcutfbb[6][8][0] = 10.;


zcutfbb[6][7][6] = 10.;
zcutfbb[6][7][5] = 10.;
zcutfbb[6][7][4] = 10.;
zcutfbb[6][7][3] = 10.;
zcutfbb[6][7][2] = 10.;
zcutfbb[6][7][1] = 10.;
zcutfbb[6][7][0] = 10.;

zcutfbb[6][6][5] = 10.;
zcutfbb[6][6][4] = 10.;
zcutfbb[6][6][3] = 10.;
zcutfbb[6][6][2] = 10.;
zcutfbb[6][6][1] = 10.;
zcutfbb[6][6][0] = 10.;

zcutfbb[6][5][4] = 10.;
zcutfbb[6][5][3] = 10.;
zcutfbb[6][5][2] = 10.;
zcutfbb[6][5][1] = 10.;
zcutfbb[6][5][0] = 10.;

zcutfbb[6][4][3] = 10.;
zcutfbb[6][4][2] = 10.;
zcutfbb[6][4][1] = 10.;
zcutfbb[6][4][0] = 10.;

zcutfbb[6][3][2] = 10.;
zcutfbb[6][3][1] = 10.;
zcutfbb[6][3][0] = 10.;

zcutfbb[6][2][1] = 1.;
zcutfbb[6][2][0] = 1.;

zcutfbb[6][1][0] = 0.5;

// ++++++++++++Last layer = 5

zcutfbb[5][12][11] = 10.;
zcutfbb[5][12][10] = 10.;
zcutfbb[5][12][9] = 10.;
zcutfbb[5][12][8] = 10.;
zcutfbb[5][12][7] = 10.;
zcutfbb[5][12][6] = 10.;
zcutfbb[5][12][5] = 10.;
zcutfbb[5][12][4] = 10.;
zcutfbb[5][12][3] = 10.;
zcutfbb[5][12][2] = 10.;
zcutfbb[5][12][1] = 10.;
zcutfbb[5][12][0] = 10.;

zcutfbb[5][11][10] = 10.;
zcutfbb[5][11][9] = 10.;
zcutfbb[5][11][8] = 10.;
zcutfbb[5][11][7] = 10.;
zcutfbb[5][11][6] = 10.;
zcutfbb[5][11][5] = 10.;
zcutfbb[5][11][4] = 10.;
zcutfbb[5][11][3] = 10.;
zcutfbb[5][11][2] = 10.;
zcutfbb[5][11][1] = 10.;
zcutfbb[5][11][0] = 10.;

zcutfbb[5][10][9] = 10.;
zcutfbb[5][10][8] = 10.;
zcutfbb[5][10][7] = 10.;
zcutfbb[5][10][6] = 10.;
zcutfbb[5][10][5] = 10.;
zcutfbb[5][10][4] = 10.;
zcutfbb[5][10][3] = 10.;
zcutfbb[5][10][2] = 10.;
zcutfbb[5][10][1] = 10.;
zcutfbb[5][10][0] = 10.;

zcutfbb[5][9][8] = 10.;
zcutfbb[5][9][7] = 10.;
zcutfbb[5][9][6] = 10.;
zcutfbb[5][9][5] = 10.;
zcutfbb[5][9][4] = 10.;
zcutfbb[5][9][3] = 10.;
zcutfbb[5][9][2] = 10.;
zcutfbb[5][9][1] = 10.;
zcutfbb[5][9][0] = 10.;


zcutfbb[5][8][7] = 10.;
zcutfbb[5][8][6] = 10.;
zcutfbb[5][8][5] = 10.;
zcutfbb[5][8][4] = 10.;
zcutfbb[5][8][3] = 10.;
zcutfbb[5][8][2] = 10.;
zcutfbb[5][8][1] = 10.;
zcutfbb[5][8][0] = 10.;


zcutfbb[5][7][6] = 10.;
zcutfbb[5][7][5] = 10.;
zcutfbb[5][7][4] = 10.;
zcutfbb[5][7][3] = 10.;
zcutfbb[5][7][2] = 10.;
zcutfbb[5][7][1] = 10.;
zcutfbb[5][7][0] = 10.;

zcutfbb[5][6][5] = 10.;
zcutfbb[5][6][4] = 10.;
zcutfbb[5][6][3] = 10.;
zcutfbb[5][6][2] = 10.;
zcutfbb[5][6][1] = 10.;
zcutfbb[5][6][0] = 10.;

zcutfbb[5][5][4] = 10.;
zcutfbb[5][5][3] = 10.;
zcutfbb[5][5][2] = 10.;
zcutfbb[5][5][1] = 10.;
zcutfbb[5][5][0] = 10.;

zcutfbb[5][4][3] = 10.;
zcutfbb[5][4][2] = 10.;
zcutfbb[5][4][1] = 10.;
zcutfbb[5][4][0] = 10.;

zcutfbb[5][3][2] = 10.;
zcutfbb[5][3][1] = 10.;
zcutfbb[5][3][0] = 10.;

zcutfbb[5][2][1] = 1.;
zcutfbb[5][2][0] = 1.;

zcutfbb[5][1][0] = 0.5;


// Zbarrel cut if last layer in forward
// ++++++++++++Last layer = 13

zcutfrw[13][13][12] = 10.;
zcutfrw[13][13][11] = 10.;
zcutfrw[13][13][10] = 10.;
zcutfrw[13][13][9] = 10.;
zcutfrw[13][13][8] = 10.;
zcutfrw[13][13][7] = 10.;
zcutfrw[13][13][6] = 10.;
zcutfrw[13][13][5] = 10.;
zcutfrw[13][13][4] = 10.;
zcutfrw[13][13][3] = 10.;
zcutfrw[13][13][2] = 10.;
zcutfrw[13][13][1] = 10.;
zcutfrw[13][13][0] = 10.;

zcutfrw[13][12][11] = 11.;
zcutfrw[13][12][10] = 10.;
zcutfrw[13][12][9] = 10.;
zcutfrw[13][12][8] = 10.;
zcutfrw[13][12][7] = 10.;
zcutfrw[13][12][6] = 10.;
zcutfrw[13][12][5] = 10.;
zcutfrw[13][12][4] = 10.;
zcutfrw[13][12][3] = 10.;
zcutfrw[13][12][2] = 10.;
zcutfrw[13][12][1] = 10.;
zcutfrw[13][12][0] = 10.;

zcutfrw[13][11][10] = 10.;
zcutfrw[13][11][9] = 10.;
zcutfrw[13][11][8] = 10.;
zcutfrw[13][11][7] = 10.;
zcutfrw[13][11][6] = 10.;
zcutfrw[13][11][5] = 10.;
zcutfrw[13][11][4] = 10.;
zcutfrw[13][11][3] = 10.;
zcutfrw[13][11][2] = 10.;
zcutfrw[13][11][1] = 10.;
zcutfrw[13][11][0] = 10.;


zcutfrw[13][10][9] = 10.;
zcutfrw[13][10][8] = 10.;
zcutfrw[13][10][7] = 10.;
zcutfrw[13][10][6] = 10.;
zcutfrw[13][10][5] = 10.;
zcutfrw[13][10][4] = 10.;
zcutfrw[13][10][3] = 10.;
zcutfrw[13][10][2] = 10.;
zcutfrw[13][10][1] = 10.;
zcutfrw[13][10][0] = 10.;

zcutfrw[13][9][8] = 10.;
zcutfrw[13][9][7] = 10.;
zcutfrw[13][9][6] = 10.;
zcutfrw[13][9][5] = 10.;
zcutfrw[13][9][4] = 10.;
zcutfrw[13][9][3] = 10.;
zcutfrw[13][9][2] = 10.;
zcutfrw[13][9][1] = 10.;
zcutfrw[13][9][0] = 10.;

zcutfrw[13][8][7] = 10.;
zcutfrw[13][8][6] = 10.;
zcutfrw[13][8][5] = 10.;
zcutfrw[13][8][4] = 10.;
zcutfrw[13][8][3] = 10.;
zcutfrw[13][8][2] = 10.;
zcutfrw[13][8][1] = 10.;
zcutfrw[13][8][0] = 10.;

zcutfrw[13][7][6] = 10.;
zcutfrw[13][7][5] = 10.;
zcutfrw[13][7][4] = 10;
zcutfrw[13][7][3] = 10;
zcutfrw[13][7][2] = 10;
zcutfrw[13][7][1] = 10;
zcutfrw[13][7][0] = 10;

zcutfrw[13][6][5] = 10;
zcutfrw[13][6][4] = 10;
zcutfrw[13][6][3] = 10;
zcutfrw[13][6][2] = 10;
zcutfrw[13][6][1] = 10;
zcutfrw[13][6][0] = 10;

zcutfrw[13][5][4] = 10.;
zcutfrw[13][5][3] = 10.;
zcutfrw[13][5][2] = 10.;
zcutfrw[13][5][1] = 10.;
zcutfrw[13][5][0] = 10.;

zcutfrw[13][4][3] = 10.;
zcutfrw[13][4][2] = 10.;
zcutfrw[13][4][1] = 10.;
zcutfrw[13][4][0] = 10.;

zcutfrw[13][3][2] = 10.;
zcutfrw[13][3][1] = 10.;
zcutfrw[13][3][0] = 10.;

zcutfrw[13][2][1] = 1.;
zcutfrw[13][2][0] = 1.;

zcutfrw[13][1][0] = 0.7;

// +++++++++++ Last layer = 12

zcutfrw[12][12][11] = 10.;
zcutfrw[12][12][10] = 10.;
zcutfrw[12][12][9] = 10.;
zcutfrw[12][12][8] = 10.;
zcutfrw[12][12][7] = 10.;
zcutfrw[12][12][6] = 10.;
zcutfrw[12][12][5] = 10.;
zcutfrw[12][12][4] = 10.;
zcutfrw[12][12][3] = 10.;
zcutfrw[12][12][2] = 10.;
zcutfrw[12][12][1] = 10.;
zcutfrw[12][12][0] = 10.;

zcutfrw[12][11][10] = 10.;
zcutfrw[12][11][9] = 10.;
zcutfrw[12][11][8] = 10.;
zcutfrw[12][11][7] = 10.;
zcutfrw[12][11][6] = 10.;
zcutfrw[12][11][5] = 10.;
zcutfrw[12][11][4] = 10.;
zcutfrw[12][11][3] = 10.;
zcutfrw[12][11][2] = 10.;
zcutfrw[12][11][1] = 10.;
zcutfrw[12][11][0] = 10.;


zcutfrw[12][10][9] = 10.;
zcutfrw[12][10][8] = 10.;
zcutfrw[12][10][7] = 10.;
zcutfrw[12][10][6] = 10.;
zcutfrw[12][10][5] = 10.;
zcutfrw[12][10][4] = 10.;
zcutfrw[12][10][3] = 10.;
zcutfrw[12][10][2] = 10.;
zcutfrw[12][10][1] = 10.;
zcutfrw[12][10][0] = 10.;

zcutfrw[12][9][12] = 10.;
zcutfrw[12][9][11] = 10.;
zcutfrw[12][9][10] = 10.;
zcutfrw[12][9][9] = 10.;
zcutfrw[12][9][8] = 10.;
zcutfrw[12][9][7] = 10.;
zcutfrw[12][9][6] = 10.;
zcutfrw[12][9][5] = 10.;
zcutfrw[12][9][4] = 10.;
zcutfrw[12][9][3] = 10.;
zcutfrw[12][9][2] = 10.;
zcutfrw[12][9][1] = 10.;
zcutfrw[12][9][0] = 10.;

zcutfrw[12][8][7] = 10.;
zcutfrw[12][8][6] = 10.;
zcutfrw[12][8][5] = 10.;
zcutfrw[12][8][4] = 10.;
zcutfrw[12][8][3] = 10.;
zcutfrw[12][8][2] = 10.;
zcutfrw[12][8][1] = 10.;
zcutfrw[12][8][0] = 10.;

zcutfrw[12][7][6] = 10.;
zcutfrw[12][7][5] = 10.;
zcutfrw[12][7][4] = 10.;
zcutfrw[12][7][3] = 10.;
zcutfrw[12][7][2] = 10.;
zcutfrw[12][7][1] = 10.;
zcutfrw[12][7][0] = 10.;

zcutfrw[12][6][5] = 10.;
zcutfrw[12][6][4] = 10.;
zcutfrw[12][6][3] = 10.;
zcutfrw[12][6][2] = 10.;
zcutfrw[12][6][1] = 10.;
zcutfrw[12][6][0] = 10.;

zcutfrw[12][5][4] = 10.;
zcutfrw[12][5][3] = 10.;
zcutfrw[12][5][2] = 10.;
zcutfrw[12][5][1] = 10.;
zcutfrw[12][5][0] = 10.;

zcutfrw[12][4][3] = 10.;
zcutfrw[12][4][2] = 10.;
zcutfrw[12][4][1] = 10.;
zcutfrw[12][4][0] = 10.;

zcutfrw[12][3][2] = 10.;
zcutfrw[12][3][1] = 10.;
zcutfrw[12][3][0] = 10.;

zcutfrw[12][2][1] = 1.;
zcutfrw[12][2][0] = 1.;

zcutfrw[12][1][0] = 0.7;


// +++++++++++ Last layer = 11

zcutfrw[11][11][10] = 10.;
zcutfrw[11][11][9] = 10.;
zcutfrw[11][11][8] = 10.;
zcutfrw[11][11][7] = 10.;
zcutfrw[11][11][6] = 10.;
zcutfrw[11][11][5] = 10.;
zcutfrw[11][11][4] = 10.;
zcutfrw[11][11][3] = 10.;
zcutfrw[11][11][2] = 10.;
zcutfrw[11][11][1] = 10.;
zcutfrw[11][11][0] = 10.;


zcutfrw[11][10][9] = 10.;
zcutfrw[11][10][8] = 10.;
zcutfrw[11][10][7] = 10.;
zcutfrw[11][10][6] = 10.;
zcutfrw[11][10][5] = 10.;
zcutfrw[11][10][4] = 10.;
zcutfrw[11][10][3] = 10.;
zcutfrw[11][10][2] = 10.;
zcutfrw[11][10][1] = 10.;
zcutfrw[11][10][0] = 10.;

zcutfrw[11][9][11] = 10.;
zcutfrw[11][9][11] = 10.;
zcutfrw[11][9][10] = 10.;
zcutfrw[11][9][9] = 10.;
zcutfrw[11][9][8] = 10.;
zcutfrw[11][9][7] = 10.;
zcutfrw[11][9][6] = 10.;
zcutfrw[11][9][5] = 10.;
zcutfrw[11][9][4] = 10.;
zcutfrw[11][9][3] = 10.;
zcutfrw[11][9][2] = 10.;
zcutfrw[11][9][1] = 10.;
zcutfrw[11][9][0] = 10.;

zcutfrw[11][8][7] = 10.;
zcutfrw[11][8][6] = 10.;
zcutfrw[11][8][5] = 10.;
zcutfrw[11][8][4] = 10.;
zcutfrw[11][8][3] = 10.;
zcutfrw[11][8][2] = 10.;
zcutfrw[11][8][1] = 10.;
zcutfrw[11][8][0] = 10.;

zcutfrw[11][7][6] = 10.;
zcutfrw[11][7][5] = 10.;
zcutfrw[11][7][4] = 10.;
zcutfrw[11][7][3] = 10.;
zcutfrw[11][7][2] = 10.;
zcutfrw[11][7][1] = 10.;
zcutfrw[11][7][0] = 10.;

zcutfrw[11][6][5] = 10.;
zcutfrw[11][6][4] = 10.;
zcutfrw[11][6][3] = 10.;
zcutfrw[11][6][2] = 10.;
zcutfrw[11][6][1] = 10.;
zcutfrw[11][6][0] = 10.;

zcutfrw[11][5][4] = 10.;
zcutfrw[11][5][3] = 10.;
zcutfrw[11][5][2] = 10.;
zcutfrw[11][5][1] = 10.;
zcutfrw[11][5][0] = 10.;

zcutfrw[11][4][3] = 10.;
zcutfrw[11][4][2] = 10.;
zcutfrw[11][4][1] = 10.;
zcutfrw[11][4][0] = 10.;

zcutfrw[11][3][2] = 10.;
zcutfrw[11][3][1] = 10.;
zcutfrw[11][3][0] = 10.;

zcutfrw[11][2][1] = 1.;
zcutfrw[11][2][0] = 1.;
zcutfrw[11][1][0] = 0.7;

// +++++++++++ Last layer = 10

zcutfrw[10][10][9] = 10.;
zcutfrw[10][10][8] = 10.;
zcutfrw[10][10][7] = 10.;
zcutfrw[10][10][6] = 10.;
zcutfrw[10][10][5] = 10.;
zcutfrw[10][10][4] = 10.;
zcutfrw[10][10][3] = 10.;
zcutfrw[10][10][2] = 10.;
zcutfrw[10][10][1] = 10.;
zcutfrw[10][10][0] = 10.;

zcutfrw[10][9][8] = 15.;
zcutfrw[10][9][7] = 10.;
zcutfrw[10][9][6] = 10.;
zcutfrw[10][9][5] = 10.;
zcutfrw[10][9][4] = 10.;
zcutfrw[10][9][3] = 10.;
zcutfrw[10][9][2] = 10.;
zcutfrw[10][9][1] = 10.;
zcutfrw[10][9][0] = 10.;

zcutfrw[10][8][7] = 10.;
zcutfrw[10][8][6] = 10.;
zcutfrw[10][8][5] = 10.;
zcutfrw[10][8][4] = 10.;
zcutfrw[10][8][3] = 10.;
zcutfrw[10][8][2] = 10.;
zcutfrw[10][8][1] = 10.;
zcutfrw[10][8][0] = 10.;

zcutfrw[10][7][6] = 10.;
zcutfrw[10][7][5] = 10.;
zcutfrw[10][7][4] = 10.;
zcutfrw[10][7][3] = 10.;
zcutfrw[10][7][2] = 10.;
zcutfrw[10][7][1] = 10.;
zcutfrw[10][7][0] = 10.;

zcutfrw[10][6][5] = 10.;
zcutfrw[10][6][4] = 10.;
zcutfrw[10][6][3] = 10.;
zcutfrw[10][6][2] = 10.;
zcutfrw[10][6][1] = 10.;
zcutfrw[10][6][0] = 10.;

zcutfrw[10][5][4] = 10.;
zcutfrw[10][5][3] = 10.;
zcutfrw[10][5][2] = 10.;
zcutfrw[10][5][1] = 10.;
zcutfrw[10][5][0] = 10.;

zcutfrw[10][4][3] = 10.;
zcutfrw[10][4][2] = 10.;
zcutfrw[10][4][1] = 10.;
zcutfrw[10][4][0] = 10.;

zcutfrw[10][3][2] = 10.;
zcutfrw[10][3][1] = 10.;
zcutfrw[10][3][0] = 10.;

zcutfrw[10][2][1] = 1.;
zcutfrw[10][2][0] = 1.;

zcutfrw[10][1][0] = 0.7;

// +++++++++++ Last layer = 9


zcutfrw[9][9][8] = 10.;
zcutfrw[9][9][7] = 10.;
zcutfrw[9][9][6] = 10.;
zcutfrw[9][9][5] = 10.;
zcutfrw[9][9][4] = 10.;
zcutfrw[9][9][3] = 10.;
zcutfrw[9][9][2] = 10.;
zcutfrw[9][9][1] = 10.;
zcutfrw[9][9][0] = 10.;

zcutfrw[9][8][7] = 10.;
zcutfrw[9][8][6] = 10.;
zcutfrw[9][8][5] = 10.;
zcutfrw[9][8][4] = 10.;
zcutfrw[9][8][3] = 10.;
zcutfrw[9][8][2] = 10.;
zcutfrw[9][8][1] = 10.;
zcutfrw[9][8][0] = 10.;

zcutfrw[9][7][6] = 10.;
zcutfrw[9][7][5] = 10.;
zcutfrw[9][7][4] = 10.;
zcutfrw[9][7][3] = 10.;
zcutfrw[9][7][2] = 10.;
zcutfrw[9][7][1] = 10.;
zcutfrw[9][7][0] = 10.;

zcutfrw[9][6][5] = 10.;
zcutfrw[9][6][4] = 10.;
zcutfrw[9][6][3] = 10.;
zcutfrw[9][6][2] = 10.;
zcutfrw[9][6][1] = 10.;
zcutfrw[9][6][0] = 10.;

zcutfrw[9][5][4] = 10.;
zcutfrw[9][5][3] = 10.;
zcutfrw[9][5][2] = 10.;
zcutfrw[9][5][1] = 10.;
zcutfrw[9][5][0] = 10.;

zcutfrw[9][4][3] = 10.;
zcutfrw[9][4][2] = 10.;
zcutfrw[9][4][1] = 10.;
zcutfrw[9][4][0] = 10.;

zcutfrw[9][3][2] = 10.;
zcutfrw[9][3][1] = 10.;
zcutfrw[9][3][0] = 10.;

zcutfrw[9][2][1] = 1.;
zcutfrw[9][2][0] = 1.;

zcutfrw[9][1][0] = 0.7;

// +++++++++++ Last layer = 8


zcutfrw[8][8][7] = 10.;
zcutfrw[8][8][6] = 10.;
zcutfrw[8][8][5] = 10.;
zcutfrw[8][8][4] = 10.;
zcutfrw[8][8][3] = 10.;
zcutfrw[8][8][2] = 10.;
zcutfrw[8][8][1] = 10.;
zcutfrw[8][8][0] = 10.;

zcutfrw[8][7][6] = 10.;
zcutfrw[8][7][5] = 10.;
zcutfrw[8][7][4] = 10.;
zcutfrw[8][7][3] = 10.;
zcutfrw[8][7][2] = 10.;
zcutfrw[8][7][1] = 10.;
zcutfrw[8][7][0] = 10.;

zcutfrw[8][6][5] = 10.;
zcutfrw[8][6][4] = 10.;
zcutfrw[8][6][3] = 10.;
zcutfrw[8][6][2] = 10.;
zcutfrw[8][6][1] = 10.;
zcutfrw[8][6][0] = 10.;

zcutfrw[8][5][4] = 10.;
zcutfrw[8][5][3] = 10.;
zcutfrw[8][5][2] = 10.;
zcutfrw[8][5][1] = 10.;
zcutfrw[8][5][0] = 10.;

zcutfrw[8][4][3] = 10.;
zcutfrw[8][4][2] = 10.;
zcutfrw[8][4][1] = 10.;
zcutfrw[8][4][0] = 10.;

zcutfrw[8][3][2] = 10.;
zcutfrw[8][3][1] = 10.;
zcutfrw[8][3][0] = 10.;

zcutfrw[8][2][1] = 1.;
zcutfrw[8][2][0] = 1.;

zcutfrw[8][1][0] = 0.7;

// +++++++++++ Last layer = 7

zcutfrw[7][7][6] = 10.;
zcutfrw[7][7][5] = 10.;
zcutfrw[7][7][4] = 10.;
zcutfrw[7][7][3] = 10.;
zcutfrw[7][7][2] = 10.;
zcutfrw[7][7][1] = 10.;
zcutfrw[7][7][0] = 10.;

zcutfrw[7][6][5] = 10.;
zcutfrw[7][6][4] = 10.;
zcutfrw[7][6][3] = 10.;
zcutfrw[7][6][2] = 10.;
zcutfrw[7][6][1] = 10.;
zcutfrw[7][6][0] = 10.;

zcutfrw[7][5][4] = 10.;
zcutfrw[7][5][3] = 10.;
zcutfrw[7][5][2] = 10.;
zcutfrw[7][5][1] = 10.;
zcutfrw[7][5][0] = 10.;

zcutfrw[7][4][3] = 10.;
zcutfrw[7][4][2] = 10.;
zcutfrw[7][4][1] = 10.;
zcutfrw[7][4][0] = 10.;

zcutfrw[7][3][2] = 10.;
zcutfrw[7][3][1] = 10.;
zcutfrw[7][3][0] = 10.;

zcutfrw[7][2][1] = 1.;
zcutfrw[7][1][0] = 0.7;


// +++++++++++ Last layer = 6

zcutfrw[6][6][5] = 10.;
zcutfrw[6][6][4] = 10.;
zcutfrw[6][6][3] = 10.;
zcutfrw[6][6][2] = 10.;
zcutfrw[6][6][1] = 10.;
zcutfrw[6][6][0] = 10.;

zcutfrw[6][5][4] = 10.;
zcutfrw[6][5][3] = 10.;
zcutfrw[6][5][2] = 10.;
zcutfrw[6][5][1] = 10.;
zcutfrw[6][5][0] = 10.;

zcutfrw[6][4][3] = 10.;
zcutfrw[6][4][2] = 10.;
zcutfrw[6][4][1] = 10.;
zcutfrw[6][4][0] = 10.;

zcutfrw[6][3][2] = 10.;
zcutfrw[6][3][1] = 10.;
zcutfrw[6][3][0] = 10.;

zcutfrw[6][2][1] = 1.;
zcutfrw[6][2][0] = 1.;

zcutfrw[6][1][0] = 0.7;

// +++++++++++ Last layer = 5

zcutfrw[5][5][4] = 10.;
zcutfrw[5][5][3] = 10.;
zcutfrw[5][5][2] = 10.;
zcutfrw[5][5][1] = 10.;
zcutfrw[5][5][0] = 10.;

zcutfrw[5][4][3] = 10.;
zcutfrw[5][4][2] = 10.;
zcutfrw[5][4][1] = 10.;
zcutfrw[5][4][0] = 10.;

zcutfrw[5][3][2] = 10.;
zcutfrw[5][3][1] = 10.;
zcutfrw[5][3][0] = 10.;

zcutfrw[5][2][1] = 1.;
zcutfrw[5][2][0] = 1.;

zcutfrw[5][1][0] = 0.7;

// Forward-barrel roads, phiwinbfrw

phiwinbfrw[13][5][12] = 0.025;
phiwinbfrw[13][5][11] = 0.025;
phiwinbfrw[13][5][10] = 0.025;
phiwinbfrw[13][5][9] = 0.025;
phiwinbfrw[13][5][8] = 0.025;
phiwinbfrw[13][5][2] = 0.031;
phiwinbfrw[13][5][1] = 0.031;
phiwinbfrw[13][5][0] = 0.031;
phiwinbfrw[13][1][2] = 0.005;
phiwinbfrw[13][1][1] = 0.005;
phiwinbfrw[13][1][0] = 0.005;
phiwinbfrw[13][0][2] = 0.005;
phiwinbfrw[13][0][1] = 0.005;
phiwinbfrw[13][0][0] = 0.005;


phiwinbfrw[12][5][12] = 0.025;
phiwinbfrw[12][5][11] = 0.025;
phiwinbfrw[12][5][10] = 0.025;
phiwinbfrw[12][5][9] = 0.025;
phiwinbfrw[12][5][8] = 0.025;
phiwinbfrw[12][5][2] = 0.031;
phiwinbfrw[12][5][1] = 0.031;
phiwinbfrw[12][5][0] = 0.031;
phiwinbfrw[12][1][2] = 0.005;
phiwinbfrw[12][1][1] = 0.005;
phiwinbfrw[12][1][0] = 0.005;
phiwinbfrw[12][0][2] = 0.005;
phiwinbfrw[12][0][1] = 0.005;
phiwinbfrw[12][0][0] = 0.005;

phiwinbfrw[11][5][12] = 0.025;
phiwinbfrw[11][5][11] = 0.025;
phiwinbfrw[11][5][10] = 0.025;
phiwinbfrw[11][5][9] = 0.025;
phiwinbfrw[11][5][8] = 0.025;
phiwinbfrw[11][5][2] = 0.031;
phiwinbfrw[11][5][1] = 0.031;
phiwinbfrw[11][5][0] = 0.031;
phiwinbfrw[11][1][2] = 0.005;
phiwinbfrw[11][1][1] = 0.005;
phiwinbfrw[11][1][0] = 0.005;
phiwinbfrw[11][0][2] = 0.005;
phiwinbfrw[11][0][1] = 0.005;
phiwinbfrw[11][0][0] = 0.005;


phiwinbfrw[10][5][12] = 0.025;
phiwinbfrw[10][5][11] = 0.025;
phiwinbfrw[10][5][10] = 0.025;
phiwinbfrw[10][5][9] = 0.025;
phiwinbfrw[10][5][8] = 0.025;
phiwinbfrw[10][5][2] = 0.031;
phiwinbfrw[10][5][1] = 0.031;
phiwinbfrw[10][5][0] = 0.031;
phiwinbfrw[10][1][2] = 0.025;
phiwinbfrw[10][1][1] = 0.025;
phiwinbfrw[10][1][0] = 0.025;
phiwinbfrw[10][0][2] = 0.025;
phiwinbfrw[10][0][1] = 0.025;
phiwinbfrw[10][0][0] = 0.025;

phiwinbfrw[9][5][12] = 0.025;
phiwinbfrw[9][5][11] = 0.025;
phiwinbfrw[9][5][10] = 0.025;
phiwinbfrw[9][5][9] = 0.025;
phiwinbfrw[9][5][8] = 0.025;
phiwinbfrw[9][5][2] = 0.025;
phiwinbfrw[9][5][1] = 0.025;
phiwinbfrw[9][5][0] = 0.025;
phiwinbfrw[9][1][2] = 0.025;
phiwinbfrw[9][1][1] = 0.025;
phiwinbfrw[9][1][0] = 0.025;
phiwinbfrw[9][0][2] = 0.025;
phiwinbfrw[9][0][1] = 0.025;
phiwinbfrw[9][0][0] = 0.025;

phiwinbfrw[8][5][12] = 0.025;
phiwinbfrw[8][5][11] = 0.025;
phiwinbfrw[8][5][10] = 0.025;
phiwinbfrw[8][5][9] = 0.025;
phiwinbfrw[8][5][8] = 0.025;
phiwinbfrw[8][5][2] = 0.025;
phiwinbfrw[8][5][1] = 0.025;
phiwinbfrw[8][5][0] = 0.025;
phiwinbfrw[8][1][2] = 0.025;
phiwinbfrw[8][1][1] = 0.025;
phiwinbfrw[8][1][0] = 0.025;
phiwinbfrw[8][0][2] = 0.025;
phiwinbfrw[8][0][1] = 0.025;
phiwinbfrw[8][0][0] = 0.025;

phiwinbfrw[7][5][12] = 0.025;
phiwinbfrw[7][5][11] = 0.025;
phiwinbfrw[7][5][10] = 0.025;
phiwinbfrw[7][5][9] = 0.025;
phiwinbfrw[7][5][8] = 0.025;
phiwinbfrw[7][5][2] = 0.025;
phiwinbfrw[7][5][1] = 0.025;
phiwinbfrw[7][5][0] = 0.025;
phiwinbfrw[7][1][2] = 0.025;
phiwinbfrw[7][1][1] = 0.025;
phiwinbfrw[7][1][0] = 0.025;
phiwinbfrw[7][0][2] = 0.025;
phiwinbfrw[7][0][1] = 0.025;
phiwinbfrw[7][0][0] = 0.025;

phiwinbfrw[6][5][12] = 0.025;
phiwinbfrw[6][5][11] = 0.025;
phiwinbfrw[6][5][10] = 0.025;
phiwinbfrw[6][5][9] = 0.025;
phiwinbfrw[6][5][8] = 0.025;
phiwinbfrw[6][5][2] = 0.025;
phiwinbfrw[6][5][1] = 0.025;
phiwinbfrw[6][5][0] = 0.025;
phiwinbfrw[6][1][2] = 0.025;
phiwinbfrw[6][1][1] = 0.025;
phiwinbfrw[6][1][0] = 0.025;
phiwinbfrw[6][0][2] = 0.025;
phiwinbfrw[6][0][1] = 0.025;
phiwinbfrw[6][0][0] = 0.025;

phiwinbfrw[5][5][12] = 0.1;
phiwinbfrw[5][5][11] = 0.1;
phiwinbfrw[5][5][10] = 0.1;
phiwinbfrw[5][5][9] = 0.1;
phiwinbfrw[5][5][8] = 0.1;
phiwinbfrw[5][5][2] = 0.1;
phiwinbfrw[5][5][1] = 0.1;
phiwinbfrw[5][5][0] = 0.1;
phiwinbfrw[5][1][2] = 0.1;
phiwinbfrw[5][1][1] = 0.1;
phiwinbfrw[5][1][0] = 0.1;
phiwinbfrw[5][0][2] = 0.1;
phiwinbfrw[5][0][1] = 0.1;
phiwinbfrw[5][0][0] = 0.1;

// propagation cut, phicutbfrw

phicutbfrw[13][5][12] = 0.025;
phicutbfrw[13][5][11] = 0.025;
phicutbfrw[13][5][10] = 0.025;
phicutbfrw[13][5][9] = 0.025;
phicutbfrw[13][5][8] = 0.025;
phicutbfrw[13][5][2] = 0.025;
phicutbfrw[13][5][1] = 0.025;
phicutbfrw[13][5][0] = 0.025;
phicutbfrw[13][1][2] = 0.005;
phicutbfrw[13][1][1] = 0.005;
phicutbfrw[13][1][0] = 0.005;
phicutbfrw[13][0][2] = 0.005;
phicutbfrw[13][0][1] = 0.005;
phicutbfrw[13][0][0] = 0.005;


phicutbfrw[12][5][12] = 0.025;
phicutbfrw[12][5][11] = 0.025;
phicutbfrw[12][5][10] = 0.025;
phicutbfrw[12][5][9] = 0.025;
phicutbfrw[12][5][8] = 0.025;
phicutbfrw[12][5][2] = 0.025;
phicutbfrw[12][5][1] = 0.025;
phicutbfrw[12][5][0] = 0.025;
phicutbfrw[12][1][2] = 0.005;
phicutbfrw[12][1][1] = 0.005;
phicutbfrw[12][1][0] = 0.005;
phicutbfrw[12][0][2] = 0.005;
phicutbfrw[12][0][1] = 0.005;
phicutbfrw[12][0][0] = 0.005;

phicutbfrw[11][5][12] = 0.025;
phicutbfrw[11][5][11] = 0.025;
phicutbfrw[11][5][10] = 0.025;
phicutbfrw[11][5][9] = 0.025;
phicutbfrw[11][5][8] = 0.025;
phicutbfrw[11][5][2] = 0.025;
phicutbfrw[11][5][1] = 0.025;
phicutbfrw[11][5][0] = 0.025;
phicutbfrw[11][1][2] = 0.005;
phicutbfrw[11][1][1] = 0.005;
phicutbfrw[11][1][0] = 0.005;
phicutbfrw[11][0][2] = 0.005;
phicutbfrw[11][0][1] = 0.005;
phicutbfrw[11][0][0] = 0.005;


phicutbfrw[10][5][12] = 0.025;
phicutbfrw[10][5][11] = 0.025;
phicutbfrw[10][5][10] = 0.025;
phicutbfrw[10][5][9] = 0.025;
phicutbfrw[10][5][8] = 0.025;
phicutbfrw[10][5][2] = 0.025;
phicutbfrw[10][5][1] = 0.025;
phicutbfrw[10][5][0] = 0.025;
phicutbfrw[10][1][2] = 0.025;
phicutbfrw[10][1][1] = 0.025;
phicutbfrw[10][1][0] = 0.025;
phicutbfrw[10][0][2] = 0.025;
phicutbfrw[10][0][1] = 0.025;
phicutbfrw[10][0][0] = 0.025;

phicutbfrw[9][5][12] = 0.025;
phicutbfrw[9][5][11] = 0.025;
phicutbfrw[9][5][10] = 0.025;
phicutbfrw[9][5][9] = 0.025;
phicutbfrw[9][5][8] = 0.025;
phicutbfrw[9][5][2] = 0.025;
phicutbfrw[9][5][1] = 0.025;
phicutbfrw[9][5][0] = 0.025;
phicutbfrw[9][1][2] = 0.025;
phicutbfrw[9][1][1] = 0.025;
phicutbfrw[9][1][0] = 0.025;
phicutbfrw[9][0][2] = 0.025;
phicutbfrw[9][0][1] = 0.025;
phicutbfrw[9][0][0] = 0.025;

phicutbfrw[8][5][12] = 0.025;
phicutbfrw[8][5][11] = 0.025;
phicutbfrw[8][5][10] = 0.025;
phicutbfrw[8][5][9] = 0.025;
phicutbfrw[8][5][8] = 0.025;
phicutbfrw[8][5][2] = 0.025;
phicutbfrw[8][5][1] = 0.025;
phicutbfrw[8][5][0] = 0.025;
phicutbfrw[8][1][2] = 0.025;
phicutbfrw[8][1][1] = 0.025;
phicutbfrw[8][1][0] = 0.025;
phicutbfrw[8][0][2] = 0.025;
phicutbfrw[8][0][1] = 0.025;
phicutbfrw[8][0][0] = 0.025;

phicutbfrw[7][5][12] = 0.025;
phicutbfrw[7][5][11] = 0.025;
phicutbfrw[7][5][10] = 0.025;
phicutbfrw[7][5][9] = 0.025;
phicutbfrw[7][5][8] = 0.025;
phicutbfrw[7][5][2] = 0.025;
phicutbfrw[7][5][1] = 0.025;
phicutbfrw[7][5][0] = 0.025;
phicutbfrw[7][1][2] = 0.025;
phicutbfrw[7][1][1] = 0.025;
phicutbfrw[7][1][0] = 0.025;
phicutbfrw[7][0][2] = 0.025;
phicutbfrw[7][0][1] = 0.025;
phicutbfrw[7][0][0] = 0.025;

phicutbfrw[6][5][12] = 0.025;
phicutbfrw[6][5][11] = 0.025;
phicutbfrw[6][5][10] = 0.025;
phicutbfrw[6][5][9] = 0.025;
phicutbfrw[6][5][8] = 0.025;
phicutbfrw[6][5][2] = 0.025;
phicutbfrw[6][5][1] = 0.025;
phicutbfrw[6][5][0] = 0.025;
phicutbfrw[6][1][2] = 0.025;
phicutbfrw[6][1][1] = 0.025;
phicutbfrw[6][1][0] = 0.025;
phicutbfrw[6][0][2] = 0.025;
phicutbfrw[6][0][1] = 0.025;
phicutbfrw[6][0][0] = 0.025;

phicutbfrw[5][5][12] = 0.025;
phicutbfrw[5][5][11] = 0.025;
phicutbfrw[5][5][10] = 0.025;
phicutbfrw[5][5][9] = 0.025;
phicutbfrw[5][5][8] = 0.025;
phicutbfrw[5][5][2] = 0.025;
phicutbfrw[5][5][1] = 0.025;
phicutbfrw[5][5][0] = 0.025;
phicutbfrw[5][1][2] = 0.025;
phicutbfrw[5][1][1] = 0.025;
phicutbfrw[5][1][0] = 0.025;
phicutbfrw[5][0][2] = 0.025;
phicutbfrw[5][0][1] = 0.025;
phicutbfrw[5][0][0] = 0.025;


// size of window in phi-z.
// Barrel
 
// +++++++++++ Last layer = 12, zwinbar

zwinbar[12][12][11] = 15.;
zwinbar[12][12][10] = 15.;
zwinbar[12][12][9] = 15.;
zwinbar[12][12][8] = 15.;
zwinbar[12][12][7] = 15.;
zwinbar[12][12][6] = 15.;
zwinbar[12][12][5] = 15.;
zwinbar[12][12][4] = 15.;
zwinbar[12][12][3] = 15.;
zwinbar[12][12][2] = 15.;
zwinbar[12][12][1] = 15.;
zwinbar[12][12][0] = 15.;

zwinbar[12][11][10] = 15.;
zwinbar[12][11][9] = 15.;
zwinbar[12][11][8] = 15.;
zwinbar[12][11][7] = 15.;
zwinbar[12][11][6] = 15.;
zwinbar[12][11][5] = 15.;
zwinbar[12][11][4] = 15.;
zwinbar[12][11][3] = 15.;
zwinbar[12][11][2] = 15.;
zwinbar[12][11][1] = 15.;
zwinbar[12][11][0] = 15.;

zwinbar[12][10][9] = 15.;
zwinbar[12][10][8] = 15.;
zwinbar[12][10][7] = 15.;
zwinbar[12][10][6] = 15.;
zwinbar[12][10][5] = 15.;
zwinbar[12][10][4] = 15.;
zwinbar[12][10][3] = 15.;
zwinbar[12][10][2] = 15.;
zwinbar[12][10][1] = 15.;
zwinbar[12][10][0] = 15.;

zwinbar[12][9][8] = 15.;
zwinbar[12][9][7] = 15.;
zwinbar[12][9][6] = 15.;
zwinbar[12][9][5] = 15.;
zwinbar[12][9][4] = 15.;
zwinbar[12][9][3] = 15.;
zwinbar[12][9][2] = 15.;
zwinbar[12][9][1] = 15.;
zwinbar[12][9][0] = 15.;

zwinbar[12][8][7] = 15.;
zwinbar[12][8][6] = 15.;
zwinbar[12][8][5] = 15.;
zwinbar[12][8][4] = 15.;
zwinbar[12][8][3] = 15.;
zwinbar[12][8][2] = 15.;
zwinbar[12][8][1] = 15.;
zwinbar[12][8][0] = 15.;

zwinbar[12][7][6] = 15.;
zwinbar[12][7][5] = 15.;
zwinbar[12][7][4] = 15.;
zwinbar[12][7][3] = 15.;
zwinbar[12][7][2] = 15.;
zwinbar[12][7][1] = 15.;
zwinbar[12][7][0] = 15.;

zwinbar[12][6][5] = 15.;
zwinbar[12][6][4] = 15.;
zwinbar[12][6][3] = 15.;
zwinbar[12][6][2] = 15.;
zwinbar[12][6][1] = 15.;
zwinbar[12][6][0] = 15.;

zwinbar[12][5][4] = 15.;
zwinbar[12][5][3] = 15.;
zwinbar[12][5][2] = 15.;
zwinbar[12][5][1] = 15.;
zwinbar[12][5][0] = 15.;

zwinbar[12][4][3] = 15.;
zwinbar[12][4][2] = 15.;
zwinbar[12][4][1] = 15.;
zwinbar[12][4][0] = 15.;

zwinbar[12][3][2] = 10.;
zwinbar[12][3][1] = 10.;
zwinbar[12][3][0] = 10.;

zwinbar[12][2][1] = 1.;
zwinbar[12][2][0] = 1.;

zwinbar[12][1][0] = 0.7;


// +++++++++++ Last layer = 11

zwinbar[11][11][10] = 18.;
zwinbar[11][11][9] = 15.;
zwinbar[11][11][8] = 15.;
zwinbar[11][11][7] = 15.;
zwinbar[11][11][6] = 15.;
zwinbar[11][11][5] = 15.;
zwinbar[11][11][4] = 15.;
zwinbar[11][11][3] = 15.;
zwinbar[11][11][2] = 15.;
zwinbar[11][11][1] = 15.;
zwinbar[11][11][0] = 15.;


zwinbar[11][10][9] = 15.;
zwinbar[11][10][8] = 15.;
zwinbar[11][10][7] = 15.;
zwinbar[11][10][6] = 15.;
zwinbar[11][10][5] = 15.;
zwinbar[11][10][4] = 15.;
zwinbar[11][10][3] = 15.;
zwinbar[11][10][2] = 15.;
zwinbar[11][10][1] = 15.;
zwinbar[11][10][0] = 15.;

zwinbar[11][9][8] = 15.;
zwinbar[11][9][7] = 15.;
zwinbar[11][9][6] = 15.;
zwinbar[11][9][5] = 15.;
zwinbar[11][9][4] = 15.;
zwinbar[11][9][3] = 15.;
zwinbar[11][9][2] = 15.;
zwinbar[11][9][1] = 15.;
zwinbar[11][9][0] = 15.;

zwinbar[11][8][7] = 15.;
zwinbar[11][8][6] = 15.;
zwinbar[11][8][5] = 15.;
zwinbar[11][8][4] = 15.;
zwinbar[11][8][3] = 15.;
zwinbar[11][8][2] = 15.;
zwinbar[11][8][1] = 15.;
zwinbar[11][8][0] = 15.;

zwinbar[11][7][6] = 15.;
zwinbar[11][7][5] = 15.;
zwinbar[11][7][4] = 15.;
zwinbar[11][7][3] = 15.;
zwinbar[11][7][2] = 15.;
zwinbar[11][7][1] = 15.;
zwinbar[11][7][0] = 15.;

zwinbar[11][6][5] = 15.;
zwinbar[11][6][4] = 15.;
zwinbar[11][6][3] = 15.;
zwinbar[11][6][2] = 15.;
zwinbar[11][6][1] = 15.;
zwinbar[11][6][0] = 15.;

zwinbar[11][5][4] = 10.;
zwinbar[11][5][3] = 10.;
zwinbar[11][5][2] = 10.;
zwinbar[11][5][1] = 10.;
zwinbar[11][5][0] = 10.;

zwinbar[11][4][3] = 10.;
zwinbar[11][4][2] = 10.;
zwinbar[11][4][1] = 10.;
zwinbar[11][4][0] = 10.;

zwinbar[11][3][2] = 10.;
zwinbar[11][3][1] = 10.;
zwinbar[11][3][0] = 10.;

zwinbar[11][2][1] = 1.;
zwinbar[11][2][0] = 1.;
zwinbar[11][1][0] = 0.7;

// +++++++++++ Last layer = 10

zwinbar[10][10][9] = 15.;
zwinbar[10][10][8] = 15.;
zwinbar[10][10][7] = 15.;
zwinbar[10][10][6] = 10.;
zwinbar[10][10][5] = 10.;
zwinbar[10][10][4] = 10.;
zwinbar[10][10][3] = 10.;
zwinbar[10][10][2] = 10.;
zwinbar[10][10][1] = 10.;
zwinbar[10][10][0] = 10.;

zwinbar[10][9][8] = 15.;
zwinbar[10][9][7] = 15.;
zwinbar[10][9][6] = 10.;
zwinbar[10][9][5] = 10.;
zwinbar[10][9][4] = 10.;
zwinbar[10][9][3] = 10.;
zwinbar[10][9][2] = 10.;
zwinbar[10][9][1] = 10.;
zwinbar[10][9][0] = 10.;

zwinbar[10][8][7] = 15.;
zwinbar[10][8][6] = 15.;
zwinbar[10][8][5] = 10.;
zwinbar[10][8][4] = 10.;
zwinbar[10][8][3] = 10.;
zwinbar[10][8][2] = 10.;
zwinbar[10][8][1] = 10.;
zwinbar[10][8][0] = 10.;

zwinbar[10][7][6] = 15.;
zwinbar[10][7][5] = 15.;
zwinbar[10][7][4] = 10.;
zwinbar[10][7][3] = 10.;
zwinbar[10][7][2] = 10.;
zwinbar[10][7][1] = 10.;
zwinbar[10][7][0] = 10.;

zwinbar[10][6][5] = 15.;
zwinbar[10][6][4] = 15.;
zwinbar[10][6][3] = 10.;
zwinbar[10][6][2] = 10.;
zwinbar[10][6][1] = 10.;
zwinbar[10][6][0] = 10.;

zwinbar[10][5][4] = 10.;
zwinbar[10][5][3] = 10.;
zwinbar[10][5][2] = 10.;
zwinbar[10][5][1] = 10.;
zwinbar[10][5][0] = 10.;

zwinbar[10][4][3] = 10.;
zwinbar[10][4][2] = 10.;
zwinbar[10][4][1] = 10.;
zwinbar[10][4][0] = 10.;

zwinbar[10][3][2] = 10.;
zwinbar[10][3][1] = 10.;
zwinbar[10][3][0] = 10.;

zwinbar[10][2][1] = 1.;
zwinbar[10][2][0] = 1.;

zwinbar[10][1][0] = 0.7;

//propagation cut

// +++++++++++ Last layer = 12, zcutbar

zcutbar[12][12][11] = 15.;
zcutbar[12][12][10] = 15.;
zcutbar[12][12][9] = 15.;
zcutbar[12][12][8] = 15.;
zcutbar[12][12][7] = 15.;
zcutbar[12][12][6] = 15.;
zcutbar[12][12][5] = 15.;
zcutbar[12][12][4] = 15.;
zcutbar[12][12][3] = 15.;
zcutbar[12][12][2] = 15.;
zcutbar[12][12][1] = 15.;
zcutbar[12][12][0] = 15.;

zcutbar[12][11][10] = 17.;
zcutbar[12][11][9] = 17.;
zcutbar[12][11][8] = 17.;
zcutbar[12][11][7] = 17.;
zcutbar[12][11][6] = 17.;
zcutbar[12][11][5] = 17.;
zcutbar[12][11][4] = 17.;
zcutbar[12][11][3] = 17.;
zcutbar[12][11][2] = 17.;
zcutbar[12][11][1] = 17.;
zcutbar[12][11][0] = 17.;

zcutbar[12][10][9] = 17.;
zcutbar[12][10][8] = 17.;
zcutbar[12][10][7] = 17.;
zcutbar[12][10][6] = 17.;
zcutbar[12][10][5] = 17.;
zcutbar[12][10][4] = 17.;
zcutbar[12][10][3] = 17.;
zcutbar[12][10][2] = 17.;
zcutbar[12][10][1] = 17.;
zcutbar[12][10][0] = 17.;

zcutbar[12][9][8] = 10.;
zcutbar[12][9][7] = 10.;
zcutbar[12][9][6] = 10.;
zcutbar[12][9][5] = 10.;
zcutbar[12][9][4] = 10.;
zcutbar[12][9][3] = 10.;
zcutbar[12][9][2] = 10.;
zcutbar[12][9][1] = 10.;
zcutbar[12][9][0] = 10.;

zcutbar[12][8][7] = 10.;
zcutbar[12][8][6] = 10.;
zcutbar[12][8][5] = 10.;
zcutbar[12][8][4] = 10.;
zcutbar[12][8][3] = 10.;
zcutbar[12][8][2] = 10.;
zcutbar[12][8][1] = 10.;
zcutbar[12][8][0] = 10.;

zcutbar[12][7][6] = 10.;
zcutbar[12][7][5] = 10.;
zcutbar[12][7][4] = 10.;
zcutbar[12][7][3] = 10.;
zcutbar[12][7][2] = 10.;
zcutbar[12][7][1] = 10.;
zcutbar[12][7][0] = 10.;

zcutbar[12][6][5] = 10.;
zcutbar[12][6][4] = 10.;
zcutbar[12][6][3] = 10.;
zcutbar[12][6][2] = 10.;
zcutbar[12][6][1] = 10.;
zcutbar[12][6][0] = 10.;

zcutbar[12][5][4] = 10.;
zcutbar[12][5][3] = 10.;
zcutbar[12][5][2] = 10.;
zcutbar[12][5][1] = 10.;
zcutbar[12][5][0] = 10.;

zcutbar[12][4][3] = 10.;
zcutbar[12][4][2] = 10.;
zcutbar[12][4][1] = 10.;
zcutbar[12][4][0] = 10.;

zcutbar[12][3][2] = 10.;
zcutbar[12][3][1] = 10.;
zcutbar[12][3][0] = 10.;

zcutbar[12][2][1] = 1.;
zcutbar[12][2][0] = 1.;

zcutbar[12][1][0] = 0.7;


// +++++++++++ Last layer = 11

zcutbar[11][11][10] = 12.;
zcutbar[11][11][9] = 10.;
zcutbar[11][11][8] = 10.;
zcutbar[11][11][7] = 10.;
zcutbar[11][11][6] = 10.;
zcutbar[11][11][5] = 10.;
zcutbar[11][11][4] = 10.;
zcutbar[11][11][3] = 10.;
zcutbar[11][11][2] = 10.;
zcutbar[11][11][1] = 10.;
zcutbar[11][11][0] = 10.;


zcutbar[11][10][9] = 10.;
zcutbar[11][10][8] = 10.;
zcutbar[11][10][7] = 10.;
zcutbar[11][10][6] = 10.;
zcutbar[11][10][5] = 10.;
zcutbar[11][10][4] = 10.;
zcutbar[11][10][3] = 10.;
zcutbar[11][10][2] = 10.;
zcutbar[11][10][1] = 10.;
zcutbar[11][10][0] = 10.;

zcutbar[11][9][8] = 10.;
zcutbar[11][9][7] = 10.;
zcutbar[11][9][6] = 10.;
zcutbar[11][9][5] = 10.;
zcutbar[11][9][4] = 10.;
zcutbar[11][9][3] = 10.;
zcutbar[11][9][2] = 10.;
zcutbar[11][9][1] = 10.;
zcutbar[11][9][0] = 10.;

zcutbar[11][8][7] = 10.;
zcutbar[11][8][6] = 10.;
zcutbar[11][8][5] = 10.;
zcutbar[11][8][4] = 10.;
zcutbar[11][8][3] = 10.;
zcutbar[11][8][2] = 10.;
zcutbar[11][8][1] = 10.;
zcutbar[11][8][0] = 10.;

zcutbar[11][7][6] = 10.;
zcutbar[11][7][5] = 10.;
zcutbar[11][7][4] = 10.;
zcutbar[11][7][3] = 10.;
zcutbar[11][7][2] = 10.;
zcutbar[11][7][1] = 10.;
zcutbar[11][7][0] = 10.;

zcutbar[11][6][5] = 10.;
zcutbar[11][6][4] = 10.;
zcutbar[11][6][3] = 10.;
zcutbar[11][6][2] = 10.;
zcutbar[11][6][1] = 10.;
zcutbar[11][6][0] = 10.;

zcutbar[11][5][4] = 10.;
zcutbar[11][5][3] = 10.;
zcutbar[11][5][2] = 10.;
zcutbar[11][5][1] = 10.;
zcutbar[11][5][0] = 10.;

zcutbar[11][4][3] = 10.;
zcutbar[11][4][2] = 10.;
zcutbar[11][4][1] = 10.;
zcutbar[11][4][0] = 10.;

zcutbar[11][3][2] = 10.;
zcutbar[11][3][1] = 10.;
zcutbar[11][3][0] = 10.;

zcutbar[11][2][1] = 1.;
zcutbar[11][2][0] = 1.;
zcutbar[11][1][0] = 0.7;

// +++++++++++ Last layer = 10

zcutbar[10][10][9] = 12.;
zcutbar[10][10][8] = 10.;
zcutbar[10][10][7] = 10.;
zcutbar[10][10][6] = 10.;
zcutbar[10][10][5] = 10.;
zcutbar[10][10][4] = 10.;
zcutbar[10][10][3] = 10.;
zcutbar[10][10][2] = 10.;
zcutbar[10][10][1] = 10.;
zcutbar[10][10][0] = 10.;

zcutbar[10][9][8] = 10.;
zcutbar[10][9][7] = 10.;
zcutbar[10][9][6] = 10.;
zcutbar[10][9][5] = 10.;
zcutbar[10][9][4] = 10.;
zcutbar[10][9][3] = 10.;
zcutbar[10][9][2] = 10.;
zcutbar[10][9][1] = 10.;
zcutbar[10][9][0] = 10.;

zcutbar[10][8][7] = 10.;
zcutbar[10][8][6] = 10.;
zcutbar[10][8][5] = 10.;
zcutbar[10][8][4] = 10.;
zcutbar[10][8][3] = 10.;
zcutbar[10][8][2] = 10.;
zcutbar[10][8][1] = 10.;
zcutbar[10][8][0] = 10.;

zcutbar[10][7][6] = 10.;
zcutbar[10][7][5] = 10.;
zcutbar[10][7][4] = 10.;
zcutbar[10][7][3] = 10.;
zcutbar[10][7][2] = 10.;
zcutbar[10][7][1] = 10.;
zcutbar[10][7][0] = 10.;

zcutbar[10][6][5] = 10.;
zcutbar[10][6][4] = 10.;
zcutbar[10][6][3] = 10.;
zcutbar[10][6][2] = 10.;
zcutbar[10][6][1] = 10.;
zcutbar[10][6][0] = 10.;

zcutbar[10][5][4] = 10.;
zcutbar[10][5][3] = 10.;
zcutbar[10][5][2] = 10.;
zcutbar[10][5][1] = 10.;
zcutbar[10][5][0] = 10.;

zcutbar[10][4][3] = 10.;
zcutbar[10][4][2] = 10.;
zcutbar[10][4][1] = 10.;
zcutbar[10][4][0] = 10.;

zcutbar[10][3][2] = 10.;
zcutbar[10][3][1] = 10.;
zcutbar[10][3][0] = 10.;

zcutbar[10][2][1] = 1.;
zcutbar[10][2][0] = 1.;

zcutbar[10][1][0] = 0.7;


// Forward roads

// ++++++++++++Last layer = 13, zwinfrw

zwinfrw[13][13][12] = 15.;
zwinfrw[13][13][11] = 10.;
zwinfrw[13][13][10] = 10.;
zwinfrw[13][13][9] = 10.;
zwinfrw[13][13][8] = 10.;
zwinfrw[13][13][7] = 10.;
zwinfrw[13][13][6] = 10.;
zwinfrw[13][13][5] = 10.;
zwinfrw[13][13][4] = 10.;
zwinfrw[13][13][3] = 10.;
zwinfrw[13][13][2] = 10.;
zwinfrw[13][13][1] = 10.;
zwinfrw[13][13][0] = 10.;

zwinfrw[13][12][11] = 15.;
zwinfrw[13][12][10] = 10.;
zwinfrw[13][12][9] = 10.;
zwinfrw[13][12][8] = 10.;
zwinfrw[13][12][7] = 10.;
zwinfrw[13][12][6] = 10.;
zwinfrw[13][12][5] = 10.;
zwinfrw[13][12][4] = 10.;
zwinfrw[13][12][3] = 10.;
zwinfrw[13][12][2] = 10.;
zwinfrw[13][12][1] = 10.;
zwinfrw[13][12][0] = 10.;

zwinfrw[13][11][10] = 10.;
zwinfrw[13][11][9] = 10.;
zwinfrw[13][11][8] = 10.;
zwinfrw[13][11][7] = 10.;
zwinfrw[13][11][6] = 10.;
zwinfrw[13][11][5] = 10.;
zwinfrw[13][11][4] = 10.;
zwinfrw[13][11][3] = 10.;
zwinfrw[13][11][2] = 10.;
zwinfrw[13][11][1] = 10.;
zwinfrw[13][11][0] = 10.;


zwinfrw[13][10][9] = 10.;
zwinfrw[13][10][8] = 10.;
zwinfrw[13][10][7] = 10.;
zwinfrw[13][10][6] = 10.;
zwinfrw[13][10][5] = 10.;
zwinfrw[13][10][4] = 10.;
zwinfrw[13][10][3] = 10.;
zwinfrw[13][10][2] = 10.;
zwinfrw[13][10][1] = 10.;
zwinfrw[13][10][0] = 10.;

zwinfrw[13][9][8] = 10.;
zwinfrw[13][9][7] = 10.;
zwinfrw[13][9][6] = 10.;
zwinfrw[13][9][5] = 10.;
zwinfrw[13][9][4] = 10.;
zwinfrw[13][9][3] = 10.;
zwinfrw[13][9][2] = 10.;
zwinfrw[13][9][1] = 10.;
zwinfrw[13][9][0] = 10.;

zwinfrw[13][8][7] = 10.;
zwinfrw[13][8][6] = 10.;
zwinfrw[13][8][5] = 10.;
zwinfrw[13][8][4] = 10.;
zwinfrw[13][8][3] = 10.;
zwinfrw[13][8][2] = 10.;
zwinfrw[13][8][1] = 10.;
zwinfrw[13][8][0] = 10.;

zwinfrw[13][7][6] = 10.;
zwinfrw[13][7][5] = 10.;
zwinfrw[13][7][4] = 10;
zwinfrw[13][7][3] = 10;
zwinfrw[13][7][2] = 10;
zwinfrw[13][7][1] = 10;
zwinfrw[13][7][0] = 10;

zwinfrw[13][6][5] = 10;
zwinfrw[13][6][4] = 10;
zwinfrw[13][6][3] = 10;
zwinfrw[13][6][2] = 10;
zwinfrw[13][6][1] = 10;
zwinfrw[13][6][0] = 10;

zwinfrw[13][5][4] = 10.;
zwinfrw[13][5][3] = 10.;
zwinfrw[13][5][2] = 10.;
zwinfrw[13][5][1] = 10.;
zwinfrw[13][5][0] = 10.;

zwinfrw[13][4][3] = 10.;
zwinfrw[13][4][2] = 10.;
zwinfrw[13][4][1] = 10.;
zwinfrw[13][4][0] = 10.;

zwinfrw[13][3][2] = 10.;
zwinfrw[13][3][1] = 10.;
zwinfrw[13][3][0] = 10.;

zwinfrw[13][2][1] = 1.;
zwinfrw[13][2][0] = 1.;

zwinfrw[13][1][0] = 0.7;

// +++++++++++ Last layer = 12

zwinfrw[12][12][11] = 10.;
zwinfrw[12][12][10] = 10.;
zwinfrw[12][12][9] = 10.;
zwinfrw[12][12][8] = 10.;
zwinfrw[12][12][7] = 10.;
zwinfrw[12][12][6] = 10.;
zwinfrw[12][12][5] = 10.;
zwinfrw[12][12][4] = 10.;
zwinfrw[12][12][3] = 10.;
zwinfrw[12][12][2] = 10.;
zwinfrw[12][12][1] = 10.;
zwinfrw[12][12][0] = 10.;

zwinfrw[12][11][10] = 10.;
zwinfrw[12][11][9] = 10.;
zwinfrw[12][11][8] = 10.;
zwinfrw[12][11][7] = 10.;
zwinfrw[12][11][6] = 10.;
zwinfrw[12][11][5] = 10.;
zwinfrw[12][11][4] = 10.;
zwinfrw[12][11][3] = 10.;
zwinfrw[12][11][2] = 10.;
zwinfrw[12][11][1] = 10.;
zwinfrw[12][11][0] = 10.;


zwinfrw[12][10][9] = 10.;
zwinfrw[12][10][8] = 10.;
zwinfrw[12][10][7] = 10.;
zwinfrw[12][10][6] = 10.;
zwinfrw[12][10][5] = 10.;
zwinfrw[12][10][4] = 10.;
zwinfrw[12][10][3] = 10.;
zwinfrw[12][10][2] = 10.;
zwinfrw[12][10][1] = 10.;
zwinfrw[12][10][0] = 10.;

zwinfrw[12][9][12] = 10.;
zwinfrw[12][9][11] = 10.;
zwinfrw[12][9][10] = 10.;
zwinfrw[12][9][9] = 10.;
zwinfrw[12][9][8] = 10.;
zwinfrw[12][9][7] = 10.;
zwinfrw[12][9][6] = 10.;
zwinfrw[12][9][5] = 10.;
zwinfrw[12][9][4] = 10.;
zwinfrw[12][9][3] = 10.;
zwinfrw[12][9][2] = 10.;
zwinfrw[12][9][1] = 10.;
zwinfrw[12][9][0] = 10.;

zwinfrw[12][8][7] = 10.;
zwinfrw[12][8][6] = 10.;
zwinfrw[12][8][5] = 10.;
zwinfrw[12][8][4] = 10.;
zwinfrw[12][8][3] = 10.;
zwinfrw[12][8][2] = 10.;
zwinfrw[12][8][1] = 10.;
zwinfrw[12][8][0] = 10.;

zwinfrw[12][7][6] = 10.;
zwinfrw[12][7][5] = 10.;
zwinfrw[12][7][4] = 10.;
zwinfrw[12][7][3] = 10.;
zwinfrw[12][7][2] = 10.;
zwinfrw[12][7][1] = 10.;
zwinfrw[12][7][0] = 10.;

zwinfrw[12][6][5] = 10.;
zwinfrw[12][6][4] = 10.;
zwinfrw[12][6][3] = 10.;
zwinfrw[12][6][2] = 10.;
zwinfrw[12][6][1] = 10.;
zwinfrw[12][6][0] = 10.;

zwinfrw[12][5][4] = 10.;
zwinfrw[12][5][3] = 10.;
zwinfrw[12][5][2] = 10.;
zwinfrw[12][5][1] = 10.;
zwinfrw[12][5][0] = 10.;

zwinfrw[12][4][3] = 10.;
zwinfrw[12][4][2] = 10.;
zwinfrw[12][4][1] = 10.;
zwinfrw[12][4][0] = 10.;

zwinfrw[12][3][2] = 10.;
zwinfrw[12][3][1] = 10.;
zwinfrw[12][3][0] = 10.;

zwinfrw[12][2][1] = 1.;
zwinfrw[12][2][0] = 1.;

zwinfrw[12][1][0] = 0.7;


// +++++++++++ Last layer = 11

zwinfrw[11][11][10] = 10.;
zwinfrw[11][11][9] = 10.;
zwinfrw[11][11][8] = 10.;
zwinfrw[11][11][7] = 10.;
zwinfrw[11][11][6] = 10.;
zwinfrw[11][11][5] = 10.;
zwinfrw[11][11][4] = 10.;
zwinfrw[11][11][3] = 10.;
zwinfrw[11][11][2] = 10.;
zwinfrw[11][11][1] = 10.;
zwinfrw[11][11][0] = 10.;


zwinfrw[11][10][9] = 10.;
zwinfrw[11][10][8] = 10.;
zwinfrw[11][10][7] = 10.;
zwinfrw[11][10][6] = 10.;
zwinfrw[11][10][5] = 10.;
zwinfrw[11][10][4] = 10.;
zwinfrw[11][10][3] = 10.;
zwinfrw[11][10][2] = 10.;
zwinfrw[11][10][1] = 10.;
zwinfrw[11][10][0] = 10.;

zwinfrw[11][9][11] = 10.;
zwinfrw[11][9][11] = 10.;
zwinfrw[11][9][10] = 10.;
zwinfrw[11][9][9] = 10.;
zwinfrw[11][9][8] = 10.;
zwinfrw[11][9][7] = 10.;
zwinfrw[11][9][6] = 10.;
zwinfrw[11][9][5] = 10.;
zwinfrw[11][9][4] = 10.;
zwinfrw[11][9][3] = 10.;
zwinfrw[11][9][2] = 10.;
zwinfrw[11][9][1] = 10.;
zwinfrw[11][9][0] = 10.;

zwinfrw[11][8][7] = 10.;
zwinfrw[11][8][6] = 10.;
zwinfrw[11][8][5] = 10.;
zwinfrw[11][8][4] = 10.;
zwinfrw[11][8][3] = 10.;
zwinfrw[11][8][2] = 10.;
zwinfrw[11][8][1] = 10.;
zwinfrw[11][8][0] = 10.;

zwinfrw[11][7][6] = 10.;
zwinfrw[11][7][5] = 10.;
zwinfrw[11][7][4] = 10.;
zwinfrw[11][7][3] = 10.;
zwinfrw[11][7][2] = 10.;
zwinfrw[11][7][1] = 10.;
zwinfrw[11][7][0] = 10.;

zwinfrw[11][6][5] = 10.;
zwinfrw[11][6][4] = 10.;
zwinfrw[11][6][3] = 10.;
zwinfrw[11][6][2] = 10.;
zwinfrw[11][6][1] = 10.;
zwinfrw[11][6][0] = 10.;

zwinfrw[11][5][4] = 10.;
zwinfrw[11][5][3] = 10.;
zwinfrw[11][5][2] = 10.;
zwinfrw[11][5][1] = 10.;
zwinfrw[11][5][0] = 10.;

zwinfrw[11][4][3] = 10.;
zwinfrw[11][4][2] = 10.;
zwinfrw[11][4][1] = 10.;
zwinfrw[11][4][0] = 10.;

zwinfrw[11][3][2] = 10.;
zwinfrw[11][3][1] = 10.;
zwinfrw[11][3][0] = 10.;

zwinfrw[11][2][1] = 1.;
zwinfrw[11][2][0] = 1.;
zwinfrw[11][1][0] = 0.7;

// +++++++++++ Last layer = 10

zwinfrw[10][10][9] = 10.;
zwinfrw[10][10][8] = 10.;
zwinfrw[10][10][7] = 10.;
zwinfrw[10][10][6] = 10.;
zwinfrw[10][10][5] = 10.;
zwinfrw[10][10][4] = 10.;
zwinfrw[10][10][3] = 10.;
zwinfrw[10][10][2] = 10.;
zwinfrw[10][10][1] = 10.;
zwinfrw[10][10][0] = 10.;

zwinfrw[10][9][8] = 15.;
zwinfrw[10][9][7] = 10.;
zwinfrw[10][9][6] = 10.;
zwinfrw[10][9][5] = 10.;
zwinfrw[10][9][4] = 10.;
zwinfrw[10][9][3] = 10.;
zwinfrw[10][9][2] = 10.;
zwinfrw[10][9][1] = 10.;
zwinfrw[10][9][0] = 10.;

zwinfrw[10][8][7] = 10.;
zwinfrw[10][8][6] = 10.;
zwinfrw[10][8][5] = 10.;
zwinfrw[10][8][4] = 10.;
zwinfrw[10][8][3] = 10.;
zwinfrw[10][8][2] = 10.;
zwinfrw[10][8][1] = 10.;
zwinfrw[10][8][0] = 10.;

zwinfrw[10][7][6] = 10.;
zwinfrw[10][7][5] = 10.;
zwinfrw[10][7][4] = 10.;
zwinfrw[10][7][3] = 10.;
zwinfrw[10][7][2] = 10.;
zwinfrw[10][7][1] = 10.;
zwinfrw[10][7][0] = 10.;

zwinfrw[10][6][5] = 10.;
zwinfrw[10][6][4] = 10.;
zwinfrw[10][6][3] = 10.;
zwinfrw[10][6][2] = 10.;
zwinfrw[10][6][1] = 10.;
zwinfrw[10][6][0] = 10.;

zwinfrw[10][5][4] = 10.;
zwinfrw[10][5][3] = 10.;
zwinfrw[10][5][2] = 10.;
zwinfrw[10][5][1] = 10.;
zwinfrw[10][5][0] = 10.;

zwinfrw[10][4][3] = 10.;
zwinfrw[10][4][2] = 10.;
zwinfrw[10][4][1] = 10.;
zwinfrw[10][4][0] = 10.;

zwinfrw[10][3][2] = 10.;
zwinfrw[10][3][1] = 10.;
zwinfrw[10][3][0] = 10.;

zwinfrw[10][2][1] = 1.;
zwinfrw[10][2][0] = 1.;

zwinfrw[10][1][0] = 0.7;

// +++++++++++ Last layer = 9


zwinfrw[9][9][8] = 10.;
zwinfrw[9][9][7] = 10.;
zwinfrw[9][9][6] = 10.;
zwinfrw[9][9][5] = 10.;
zwinfrw[9][9][4] = 10.;
zwinfrw[9][9][3] = 10.;
zwinfrw[9][9][2] = 10.;
zwinfrw[9][9][1] = 10.;
zwinfrw[9][9][0] = 10.;

zwinfrw[9][8][7] = 10.;
zwinfrw[9][8][6] = 10.;
zwinfrw[9][8][5] = 10.;
zwinfrw[9][8][4] = 10.;
zwinfrw[9][8][3] = 10.;
zwinfrw[9][8][2] = 10.;
zwinfrw[9][8][1] = 10.;
zwinfrw[9][8][0] = 10.;

zwinfrw[9][7][6] = 10.;
zwinfrw[9][7][5] = 10.;
zwinfrw[9][7][4] = 10.;
zwinfrw[9][7][3] = 10.;
zwinfrw[9][7][2] = 10.;
zwinfrw[9][7][1] = 10.;
zwinfrw[9][7][0] = 10.;

zwinfrw[9][6][5] = 10.;
zwinfrw[9][6][4] = 10.;
zwinfrw[9][6][3] = 10.;
zwinfrw[9][6][2] = 10.;
zwinfrw[9][6][1] = 10.;
zwinfrw[9][6][0] = 10.;

zwinfrw[9][5][4] = 10.;
zwinfrw[9][5][3] = 10.;
zwinfrw[9][5][2] = 10.;
zwinfrw[9][5][1] = 10.;
zwinfrw[9][5][0] = 10.;

zwinfrw[9][4][3] = 10.;
zwinfrw[9][4][2] = 10.;
zwinfrw[9][4][1] = 10.;
zwinfrw[9][4][0] = 10.;

zwinfrw[9][3][2] = 10.;
zwinfrw[9][3][1] = 10.;
zwinfrw[9][3][0] = 10.;

zwinfrw[9][2][1] = 1.;
zwinfrw[9][2][0] = 1.;

zwinfrw[9][1][0] = 0.7;

// +++++++++++ Last layer = 8


zwinfrw[8][8][7] = 10.;
zwinfrw[8][8][6] = 10.;
zwinfrw[8][8][5] = 10.;
zwinfrw[8][8][4] = 10.;
zwinfrw[8][8][3] = 10.;
zwinfrw[8][8][2] = 10.;
zwinfrw[8][8][1] = 10.;
zwinfrw[8][8][0] = 10.;

zwinfrw[8][7][6] = 10.;
zwinfrw[8][7][5] = 10.;
zwinfrw[8][7][4] = 10.;
zwinfrw[8][7][3] = 10.;
zwinfrw[8][7][2] = 10.;
zwinfrw[8][7][1] = 10.;
zwinfrw[8][7][0] = 10.;

zwinfrw[8][6][5] = 10.;
zwinfrw[8][6][4] = 10.;
zwinfrw[8][6][3] = 10.;
zwinfrw[8][6][2] = 10.;
zwinfrw[8][6][1] = 10.;
zwinfrw[8][6][0] = 10.;

zwinfrw[8][5][4] = 10.;
zwinfrw[8][5][3] = 10.;
zwinfrw[8][5][2] = 10.;
zwinfrw[8][5][1] = 10.;
zwinfrw[8][5][0] = 10.;

zwinfrw[8][4][3] = 10.;
zwinfrw[8][4][2] = 10.;
zwinfrw[8][4][1] = 10.;
zwinfrw[8][4][0] = 10.;

zwinfrw[8][3][2] = 10.;
zwinfrw[8][3][1] = 10.;
zwinfrw[8][3][0] = 10.;

zwinfrw[8][2][1] = 1.;
zwinfrw[8][2][0] = 1.;

zwinfrw[8][1][0] = 0.7;

// +++++++++++ Last layer = 7

zwinfrw[7][7][6] = 10.;
zwinfrw[7][7][5] = 10.;
zwinfrw[7][7][4] = 10.;
zwinfrw[7][7][3] = 10.;
zwinfrw[7][7][2] = 10.;
zwinfrw[7][7][1] = 10.;
zwinfrw[7][7][0] = 10.;

zwinfrw[7][6][5] = 10.;
zwinfrw[7][6][4] = 10.;
zwinfrw[7][6][3] = 10.;
zwinfrw[7][6][2] = 10.;
zwinfrw[7][6][1] = 10.;
zwinfrw[7][6][0] = 10.;

zwinfrw[7][5][4] = 10.;
zwinfrw[7][5][3] = 10.;
zwinfrw[7][5][2] = 10.;
zwinfrw[7][5][1] = 10.;
zwinfrw[7][5][0] = 10.;

zwinfrw[7][4][3] = 10.;
zwinfrw[7][4][2] = 10.;
zwinfrw[7][4][1] = 10.;
zwinfrw[7][4][0] = 10.;

zwinfrw[7][3][2] = 10.;
zwinfrw[7][3][1] = 10.;
zwinfrw[7][3][0] = 10.;

zwinfrw[7][2][1] = 1.;
zwinfrw[7][1][0] = 0.7;


// +++++++++++ Last layer = 6

zwinfrw[6][6][5] = 10.;
zwinfrw[6][6][4] = 10.;
zwinfrw[6][6][3] = 10.;
zwinfrw[6][6][2] = 10.;
zwinfrw[6][6][1] = 10.;
zwinfrw[6][6][0] = 10.;

zwinfrw[6][5][4] = 10.;
zwinfrw[6][5][3] = 10.;
zwinfrw[6][5][2] = 10.;
zwinfrw[6][5][1] = 10.;
zwinfrw[6][5][0] = 10.;

zwinfrw[6][4][3] = 10.;
zwinfrw[6][4][2] = 10.;
zwinfrw[6][4][1] = 10.;
zwinfrw[6][4][0] = 10.;

zwinfrw[6][3][2] = 10.;
zwinfrw[6][3][1] = 10.;
zwinfrw[6][3][0] = 10.;

zwinfrw[6][2][1] = 1.;
zwinfrw[6][2][0] = 1.;

zwinfrw[6][1][0] = 0.7;

// +++++++++++ Last layer = 5

zwinfrw[5][5][4] = 10.;
zwinfrw[5][5][3] = 10.;
zwinfrw[5][5][2] = 10.;
zwinfrw[5][5][1] = 10.;
zwinfrw[5][5][0] = 10.;

zwinfrw[5][4][3] = 10.;
zwinfrw[5][4][2] = 10.;
zwinfrw[5][4][1] = 10.;
zwinfrw[5][4][0] = 10.;

zwinfrw[5][3][2] = 10.;
zwinfrw[5][3][1] = 10.;
zwinfrw[5][3][0] = 10.;

zwinfrw[5][2][1] = 1.;
zwinfrw[5][2][0] = 1.;

zwinfrw[5][1][0] = 0.7;


// Forward-barrel roads, zwinbfrw

zwinbfrw[13][5][12] = 10.;
zwinbfrw[13][5][11] = 10.;
zwinbfrw[13][5][10] = 10.;
zwinbfrw[13][5][9] = 10.;
zwinbfrw[13][5][8] = 10.;
zwinbfrw[13][5][2] = 10.;
zwinbfrw[13][5][1] = 10.;
zwinbfrw[13][5][0] = 10.;
zwinbfrw[13][1][2] = 1.;
zwinbfrw[13][1][1] = 1.;
zwinbfrw[13][1][0] = 1.;
zwinbfrw[13][0][2] = 1.;
zwinbfrw[13][0][1] = 1.;
zwinbfrw[13][0][0] = 1.;


zwinbfrw[12][5][12] = 10.;
zwinbfrw[12][5][11] = 10.;
zwinbfrw[12][5][10] = 10.;
zwinbfrw[12][5][9] = 10.;
zwinbfrw[12][5][8] = 10.;
zwinbfrw[12][5][2] = 10.;
zwinbfrw[12][5][1] = 10.;
zwinbfrw[12][5][0] = 10.;
zwinbfrw[12][1][2] = 1.;
zwinbfrw[12][1][1] = 1.;
zwinbfrw[12][1][0] = 1.;
zwinbfrw[12][0][2] = 1.;
zwinbfrw[12][0][1] = 1.;
zwinbfrw[12][0][0] = 1.;

zwinbfrw[11][5][12] = 10.;
zwinbfrw[11][5][11] = 10.;
zwinbfrw[11][5][10] = 10.;
zwinbfrw[11][5][9] = 10.;
zwinbfrw[11][5][8] = 10.;
zwinbfrw[11][5][2] = 10.;
zwinbfrw[11][5][1] = 10.;
zwinbfrw[11][5][0] = 10.;
zwinbfrw[11][1][2] = 1.;
zwinbfrw[11][1][1] = 1.;
zwinbfrw[11][1][0] = 1.;
zwinbfrw[11][0][2] = 1.;
zwinbfrw[11][0][1] = 1.;
zwinbfrw[11][0][0] = 1.;

zwinbfrw[10][5][12] = 10.;
zwinbfrw[10][5][11] = 10.;
zwinbfrw[10][5][10] = 10.;
zwinbfrw[10][5][9] = 10.;
zwinbfrw[10][5][8] = 10.;
zwinbfrw[10][5][2] = 10.;
zwinbfrw[10][5][1] = 10.;
zwinbfrw[10][5][0] = 10.;
zwinbfrw[10][1][2] = 10.;
zwinbfrw[10][1][1] = 10.;
zwinbfrw[10][1][0] = 10.;
zwinbfrw[10][0][2] = 10.;
zwinbfrw[10][0][1] = 10.;
zwinbfrw[10][0][0] = 10.;

zwinbfrw[9][5][12] = 10.;
zwinbfrw[9][5][11] = 10.;
zwinbfrw[9][5][10] = 10.;
zwinbfrw[9][5][9] = 10.;
zwinbfrw[9][5][8] = 10.;
zwinbfrw[9][5][2] = 10.;
zwinbfrw[9][5][1] = 10.;
zwinbfrw[9][5][0] = 10.;
zwinbfrw[9][1][2] = 10.;
zwinbfrw[9][1][1] = 10.;
zwinbfrw[9][1][0] = 10.;
zwinbfrw[9][0][2] = 10.;
zwinbfrw[9][0][1] = 10.;
zwinbfrw[9][0][0] = 10.;

zwinbfrw[8][5][12] = 10.;
zwinbfrw[8][5][11] = 10.;
zwinbfrw[8][5][10] = 10.;
zwinbfrw[8][5][9] = 10.;
zwinbfrw[8][5][8] = 10.;
zwinbfrw[8][5][2] = 10.;
zwinbfrw[8][5][1] = 10.;
zwinbfrw[8][5][0] = 10.;
zwinbfrw[8][1][2] = 10.;
zwinbfrw[8][1][1] = 10.;
zwinbfrw[8][1][0] = 10.;
zwinbfrw[8][0][2] = 10.;
zwinbfrw[8][0][1] = 10.;
zwinbfrw[8][0][0] = 10.;

zwinbfrw[7][5][12] = 10.;
zwinbfrw[7][5][11] = 10.;
zwinbfrw[7][5][10] = 10.;
zwinbfrw[7][5][9] = 10.;
zwinbfrw[7][5][8] = 10.;
zwinbfrw[7][5][2] = 10.;
zwinbfrw[7][5][1] = 10.;
zwinbfrw[7][5][0] = 10.;
zwinbfrw[7][1][2] = 10.;
zwinbfrw[7][1][1] = 10.;
zwinbfrw[7][1][0] = 10.;
zwinbfrw[7][0][2] = 10.;
zwinbfrw[7][0][1] = 10.;
zwinbfrw[7][0][0] = 10.;

zwinbfrw[6][5][12] = 10.;
zwinbfrw[6][5][11] = 10.;
zwinbfrw[6][5][10] = 10.;
zwinbfrw[6][5][9] = 10.;
zwinbfrw[6][5][8] = 10.;
zwinbfrw[6][5][2] = 10.;
zwinbfrw[6][5][1] = 10.;
zwinbfrw[6][5][0] = 10.;
zwinbfrw[6][1][2] = 10.;
zwinbfrw[6][1][1] = 10.;
zwinbfrw[6][1][0] = 10.;
zwinbfrw[6][0][2] = 10.;
zwinbfrw[6][0][1] = 10.;
zwinbfrw[6][0][0] = 10.;

zwinbfrw[5][5][12] = 10.;
zwinbfrw[5][5][11] = 10.;
zwinbfrw[5][5][10] = 10.;
zwinbfrw[5][5][9] = 10.;
zwinbfrw[5][5][8] = 10.;
zwinbfrw[5][5][2] = 10.;
zwinbfrw[5][5][1] = 10.;
zwinbfrw[5][5][0] = 10.;
zwinbfrw[5][1][2] = 10.;
zwinbfrw[5][1][1] = 10.;
zwinbfrw[5][1][0] = 10.;
zwinbfrw[5][0][2] = 10.;
zwinbfrw[5][0][1] = 10.;
zwinbfrw[5][0][0] = 10.;



// propagation cut, zcutbfrw

zcutbfrw[13][5][12] = 10.;
zcutbfrw[13][5][11] = 10.;
zcutbfrw[13][5][10] = 10.;
zcutbfrw[13][5][9] = 10.;
zcutbfrw[13][5][8] = 10.;
zcutbfrw[13][5][2] = 10.;
zcutbfrw[13][5][1] = 10.;
zcutbfrw[13][5][0] = 10.;
zcutbfrw[13][1][2] = 1.;
zcutbfrw[13][1][1] = 1.;
zcutbfrw[13][1][0] = 1.;
zcutbfrw[13][0][2] = 1.;
zcutbfrw[13][0][1] = 1.;
zcutbfrw[13][0][0] = 1.;


zcutbfrw[12][5][12] = 10.;
zcutbfrw[12][5][11] = 10.;
zcutbfrw[12][5][10] = 10.;
zcutbfrw[12][5][9] = 10.;
zcutbfrw[12][5][8] = 10.;
zcutbfrw[12][5][2] = 10.;
zcutbfrw[12][5][1] = 10.;
zcutbfrw[12][5][0] = 10.;
zcutbfrw[12][1][2] = 1.;
zcutbfrw[12][1][1] = 1.;
zcutbfrw[12][1][0] = 1.;
zcutbfrw[12][0][2] = 1.;
zcutbfrw[12][0][1] = 1.;
zcutbfrw[12][0][0] = 1.;

zcutbfrw[11][5][12] = 10.;
zcutbfrw[11][5][11] = 10.;
zcutbfrw[11][5][10] = 10.;
zcutbfrw[11][5][9] = 10.;
zcutbfrw[11][5][8] = 10.;
zcutbfrw[11][5][2] = 10.;
zcutbfrw[11][5][1] = 10.;
zcutbfrw[11][5][0] = 10.;
zcutbfrw[11][1][2] = 1.;
zcutbfrw[11][1][1] = 1.;
zcutbfrw[11][1][0] = 1.;
zcutbfrw[11][0][2] = 1.;
zcutbfrw[11][0][1] = 1.;
zcutbfrw[11][0][0] = 1.;

zcutbfrw[10][5][12] = 10.;
zcutbfrw[10][5][11] = 10.;
zcutbfrw[10][5][10] = 10.;
zcutbfrw[10][5][9] = 10.;
zcutbfrw[10][5][8] = 10.;
zcutbfrw[10][5][2] = 10.;
zcutbfrw[10][5][1] = 10.;
zcutbfrw[10][5][0] = 10.;
zcutbfrw[10][1][2] = 10.;
zcutbfrw[10][1][1] = 10.;
zcutbfrw[10][1][0] = 10.;
zcutbfrw[10][0][2] = 10.;
zcutbfrw[10][0][1] = 10.;
zcutbfrw[10][0][0] = 10.;

zcutbfrw[9][5][12] = 10.;
zcutbfrw[9][5][11] = 10.;
zcutbfrw[9][5][10] = 10.;
zcutbfrw[9][5][9] = 10.;
zcutbfrw[9][5][8] = 10.;
zcutbfrw[9][5][2] = 10.;
zcutbfrw[9][5][1] = 10.;
zcutbfrw[9][5][0] = 10.;
zcutbfrw[9][1][2] = 10.;
zcutbfrw[9][1][1] = 10.;
zcutbfrw[9][1][0] = 10.;
zcutbfrw[9][0][2] = 10.;
zcutbfrw[9][0][1] = 10.;
zcutbfrw[9][0][0] = 10.;

zcutbfrw[8][5][12] = 10.;
zcutbfrw[8][5][11] = 10.;
zcutbfrw[8][5][10] = 10.;
zcutbfrw[8][5][9] = 10.;
zcutbfrw[8][5][8] = 10.;
zcutbfrw[8][5][2] = 10.;
zcutbfrw[8][5][1] = 10.;
zcutbfrw[8][5][0] = 10.;
zcutbfrw[8][1][2] = 10.;
zcutbfrw[8][1][1] = 10.;
zcutbfrw[8][1][0] = 10.;
zcutbfrw[8][0][2] = 10.;
zcutbfrw[8][0][1] = 10.;
zcutbfrw[8][0][0] = 10.;

zcutbfrw[7][5][12] = 10.;
zcutbfrw[7][5][11] = 10.;
zcutbfrw[7][5][10] = 10.;
zcutbfrw[7][5][9] = 10.;
zcutbfrw[7][5][8] = 10.;
zcutbfrw[7][5][2] = 10.;
zcutbfrw[7][5][1] = 10.;
zcutbfrw[7][5][0] = 10.;
zcutbfrw[7][1][2] = 10.;
zcutbfrw[7][1][1] = 10.;
zcutbfrw[7][1][0] = 10.;
zcutbfrw[7][0][2] = 10.;
zcutbfrw[7][0][1] = 10.;
zcutbfrw[7][0][0] = 10.;

zcutbfrw[6][5][12] = 10.;
zcutbfrw[6][5][11] = 10.;
zcutbfrw[6][5][10] = 10.;
zcutbfrw[6][5][9] = 10.;
zcutbfrw[6][5][8] = 10.;
zcutbfrw[6][5][2] = 10.;
zcutbfrw[6][5][1] = 10.;
zcutbfrw[6][5][0] = 10.;
zcutbfrw[6][1][2] = 10.;
zcutbfrw[6][1][1] = 10.;
zcutbfrw[6][1][0] = 10.;
zcutbfrw[6][0][2] = 10.;
zcutbfrw[6][0][1] = 10.;
zcutbfrw[6][0][0] = 10.;

zcutbfrw[5][5][12] = 10.;
zcutbfrw[5][5][11] = 10.;
zcutbfrw[5][5][10] = 10.;
zcutbfrw[5][5][9] = 10.;
zcutbfrw[5][5][8] = 10.;
zcutbfrw[5][5][2] = 10.;
zcutbfrw[5][5][1] = 10.;
zcutbfrw[5][5][0] = 10.;
zcutbfrw[5][1][2] = 10.;
zcutbfrw[5][1][1] = 10.;
zcutbfrw[5][1][0] = 10.;
zcutbfrw[5][0][2] = 10.;
zcutbfrw[5][0][1] = 10.;
zcutbfrw[5][0][0] = 10.;


//
//
//



phiwin[0]=0.005;
phiwin[1]=0.007;
phiwin[2]=0.006;
phiwin[3]=0.005;
phiwin[4]=0.005;
phiwin[5]=0.005;
phiwin[6]=0.005;
phiwin[7]=0.005;
phiwin[8]=0.005;
phiwin[9]=0.0072;
phiwin[10]=0.005;
phiwin[11]=0.007;
phiwin[12]=0.5;


zwin[0]=0.5;
zwin[1]=1.;
zwin[2]=1.;
zwin[3]=30.;
zwin[4]=30.;
zwin[5]=30.;
zwin[6]=30.;
zwin[7]=30.;
zwin[8]=30.;
zwin[9]=15.;
zwin[10]=17.;
zwin[11]=16.;
zwin[12]=20.;

// ===========================
//phiro[0]=0.0013;

phiro[0]=0.0055;
phiro[1]=0.0017;
phiro[2]=0.0017;
//phiro[1]=0.0008;
//phiro[2]=0.0008;
phiro[3]=0.0015;
phiro[4]=0.0015;
phiro[5]=0.0015;
phiro[6]=0.0015;
phiro[7]=0.0015;
//phiro[8]=0.0008;
phiro[8]=0.0015;
//phiro[9]=0.0007;  // last change.
phiro[9]=0.0015;
phiro[10]=0.0011;
phiro[11]=0.00006;
phiro[12]=0.1;


tetro[0]=0.005;
tetro[1]=0.006;
tetro[2]=0.0007;
tetro[3]=10.;
tetro[4]=10.;
tetro[5]=10.;
tetro[6]=10.;
tetro[7]=10.;
tetro[8]=12.;
tetro[9]=24.;
tetro[10]=24.;
//tetro[9]=14.;
//tetro[10]=14.;
tetro[11]=14.;
tetro[12]=15.;

// for the first run=============			     
//static const float phiro[13]={0.0013, 0.0035, 0.002, 0.00015, 0.00015, 0.00015, 0.00015, 0.00015, 
//                             0.00015, 0.0014, 0.0014, 0.000001, 0.1};
//static const float tetro[13]={0.005, 0.003, 20., 30., 30., 30., 30.,  30., 30., 15., 10., 10., 25.};

phicut[0]=0.015;
phicut[1]=0.003;
phicut[2]=0.005;
phicut[3]=0.0009;
phicut[4]=0.0009;
phicut[5]=0.0009;
phicut[6]=0.0009;
phicut[7]=0.0009;
phicut[8]=0.0009;
phicut[9]=0.0009;
phicut[10]=0.0009;
phicut[11]=0.04;
phicut[12]=0.5;

zcut[0]=0.07;
zcut[1]=0.3;
zcut[2]=1.2;
zcut[3]=15.;
zcut[4]=15.;
zcut[5]=15.;
zcut[6]=15.;
zcut[7]=15.;
zcut[8]=15.;
zcut[9]=25.;
zcut[10]=17.;
zcut[11]=16.;
zcut[12]=35.;

phism[0]=0.0004;
phism[1]=0.0007;
phism[2]=0.00026;
phism[3]=0.00011;
phism[4]=0.00011;
phism[5]=0.00011;
phism[6]=0.00011;
phism[7]=0.00011;
phism[8]=0.00026;
phism[9]=0.00026;
phism[10]=0.00026;
phism[11]=0.00017;
phism[12]=0.00017;

zsm[0]=0.025;
zsm[1]=0.025;
zsm[2]=0.07;
zsm[3]=10.;
zsm[4]=10.;
zsm[5]=10.;
zsm[6]=10.;
zsm[7]=10.;
zsm[8]=14.;
zsm[9]=14.;
zsm[10]=14.;
zsm[11]=14.;
zsm[12]=14.;

// ==============================

phiwinf[0]=0.007;
phiwinf[1]=0.007;
phiwinf[2]=0.007;
phiwinf[3]=0.007;
phiwinf[4]=0.007;
phiwinf[5]=0.007;
phiwinf[6]=0.007;
phiwinf[7]=0.007;
phiwinf[8]=0.007;
phiwinf[9]=0.007;
phiwinf[10]=0.007;
phiwinf[11]=0.007;
phiwinf[12]=0.007;
phiwinf[13]=0.5;

zwinf[0]=1.;  
zwinf[1]=1.;   
zwinf[2]=15.; 
zwinf[3]=15.;  
zwinf[4]=15.;  
zwinf[5]=15.;
zwinf[6]=15.;
zwinf[7]=15.;
zwinf[8]=15.;
zwinf[9]=15.;
zwinf[10]=15.;
zwinf[11]=15.;
zwinf[12]=15.;
zwinf[13]=40.;
//
phirof[0]=0.0024; 
phirof[1]=0.0024; 
phirof[2]=0.0006; 
phirof[3]=0.0006;
phirof[4]=0.0006;
phirof[5]=0.0007;
phirof[6]=0.0006; 
phirof[7]=0.0006; 
phirof[8]=0.0008; 
phirof[9]=0.0006; 
phirof[10]=0.0006; 
phirof[11]=0.0006; 
phirof[12]=0.00006; 
phirof[13]=0.1;

tetrof[0]=0.0003;
tetrof[1]=0.0003; 
tetrof[2]=10.; 
tetrof[3]=10.; 
tetrof[4]=10.; 
tetrof[5]=10.; 
tetrof[6]=10.;  
tetrof[7]=10.; 
tetrof[8]=10.; 
tetrof[9]=10.; 
tetrof[10]=10.; 
tetrof[11]=10.; 
tetrof[12]=10.;  
tetrof[13]=10.;
//
phismf[0]=0.0004;
phismf[1]=0.0006;
phismf[2]=0.00026;
phismf[3]=0.00011;
phismf[4]=0.00011;
phismf[5]=0.00011;
phismf[6]=0.00011;
phismf[7]=0.00011;
phismf[8]=0.00026;
phismf[9]=0.00026;
phismf[10]=0.00026;
phismf[11]=0.00017;
phismf[12]=0.00017;
phismf[13]=0.00017;


//phismf[0]=0.03; 
//phismf[1]=0.05; 
//phismf[2]=0.012; 
//phismf[3]=0.012; 
//phismf[4]=0.012; 
//phismf[5]=0.012; 
//phismf[6]=0.012; 
//phismf[7]=0.012; 
//phismf[8]=0.012; 
//phismf[9]=0.012; 
//phismf[10]=0.012; 
//phismf[11]=0.012; 
//phismf[12]=0.02; 
//phismf[13]=0.5;



tetsmf[0]=1.;
tetsmf[1]=1.; 
tetsmf[2]=15.; 
tetsmf[3]=15.; 
tetsmf[4]=15.; 
tetsmf[5]=15.;
tetsmf[6]=15.;  
tetsmf[7]=15.; 
tetsmf[8]=15.; 
tetsmf[9]=15.; 
tetsmf[10]=15.; 
tetsmf[11]=15.; 
tetsmf[12]=15.;
tetsmf[13]=15.;
//
phicutf[0]=0.002; 
phicutf[1]=0.004; 
phicutf[2]=0.004;
phicutf[3]=0.0006;
phicutf[4]=0.0006; 
phicutf[5]=0.0009; 
phicutf[6]=0.0009;
phicutf[7]=0.0009; 
//phicutf[5]=0.0007; 
//phicutf[6]=0.0007;
//phicutf[7]=0.0007; 
//phicutf[8]=0.001;
//phicutf[9]=0.0006; 
phicutf[8]=0.0011;
phicutf[9]=0.0008; 
phicutf[10]=0.0008; 
//phicutf[11]=0.0006; 
//phicutf[12]=0.007; 
phicutf[11]=0.0009; 
phicutf[12]=0.008; 
phicutf[13]=0.5;

tetcutf[0]=0.3;
tetcutf[1]=0.7; 
tetcutf[2]=1.;
tetcutf[3]=1.; 
tetcutf[4]=1.; 
tetcutf[5]=1.;
tetcutf[6]=1.;  
tetcutf[7]=1.; 
tetcutf[8]=1.; 
tetcutf[9]=1.; 
tetcutf[10]=1.; 
tetcutf[11]=1.; 
tetcutf[12]=1.;
tetcutf[13]=40.;

// last selections
//static const double chicut=2.;
//static const double Zcut1=0.025;


filtrphi[0]=0.001;filtrphi[1]=0.0008;filtrphi[2]=0.0008;filtrphi[3]=0.0004;filtrphi[4]=0.0008;filtrphi[5]=0.0005;

filtrz[0]=0.004;filtrz[1]=0.004;filtrz[2]=12.;filtrz[3]=12.;filtrz[4]=12.;filtrz[5]=12.;
		 
chicut=4.;
Zcut1=0.045;
Rcut1=0.2;
Rcut=0.2;
ChiLimit=3.; 
zvert=2.;
atra=0.006;
// Step and boundary on Pt

ptboun=1.;
step=0.05;
ptbmax=2.;
mubarrelrad=513.;
muforwardrad=800.;
// Matching muon-tracks
zmatchend[0] = 25.;
zmatchend[1] = 50.;
zmatchbar[0] = 50.;
zmatchbar[1] = 50.;
phimatchend[0] = 0.07;
phimatchend[1] = 0.25;
phimatchbar[0] = 0.12;
phimatchbar[1] = 0.25;

} // end of constructor
}



