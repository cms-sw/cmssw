#ifndef HIC_CONST_FOR_PROPAGATION
#define HIC_CONST_FOR_PROPAGATION
namespace cms {
class HICConst {
public:
HICConst();
virtual ~HICConst(){}
void setVertex(double a);
public:
double chicut;
double Zcut1;
double Rcut1;
double Rcut;
double ChiLimit; 

float ptboun;
float ptbmax;
float step;
int numbar;
int numfrw;
int nummxt;
int nminus_gen;
int nplus_gen;
int numbarlost;
float zvert;
float atra;
float mubarrelrad;
float muforwardrad;

float phias[28];
float phibs[28];
float phiai[28];
float phibi[28];
			 
float newparam[3];
float newparamgt40[3];
float forwparam[2];

int numbargroup[3];
int numfrwgroup[10];
int barlay[5][9][8];
int frwlay[11][11][11];
int mxtlay[5];
int nofirstpixel;
//
// roads
//
// barrel phi
float phiwinbar[13][13][13];
float phicutbar[13][13][13];
float phiwinbfrw[14][14][14];
float phicutbfrw[14][14][14];
float phiwinfbb[14][14][14];
float phicutfbb[14][14][14];

// barrel z
float zwinbar[13][13][13];
float zcutbar[13][13][13];
float zwinbfrw[14][14][14];
float zcutbfrw[14][14][14];
float zwinfbb[14][14][14];
float zcutfbb[14][14][14];

// forward phi
float phiwinfrw[14][14][14];
float phicutfrw[14][14][14];

// forward z
float zwinfrw[14][14][14];
float zcutfrw[14][14][14];


float phiwin[13];
float zwin[13];

float phicut[13];
float zcut[13];

float phism[13];
float zsm[13];

float phiro[14];
float tetro[14];

float phiwinf[14];
float zwinf[14];

float phirof[14];
float tetrof[14];

float phicutf[14];
float tetcutf[14];

float phismf[14];
float tetsmf[14];

double filtrz[6];
double filtrphi[6];

double zmatchbar[2];
double zmatchend[2];
double phimatchbar[2];
double phimatchend[2];



};
}
#endif



