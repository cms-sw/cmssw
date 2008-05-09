#ifndef FMP_CONST_FOR_PROPAGATION
#define FMP_CONST_FOR_PROPAGATION

/** The COMMON BLOCK used by FastMuonPropagator.
 */
namespace cms {
class FmpConst{
public:
  FmpConst(){
//newparam  
 newparam[0]=-0.67927; newparam[1]=0.44696; newparam[2]=-0.00457;
//newparamgt40 
 newparamgt40[0]=0.38813; newparamgt40[1]=0.41003; newparamgt40[2]=-0.0019956;
//forwparam 
 forwparam[0]=0.0307; forwparam[1]=3.475;
// Coordinate of the initial point r (in barrel) and z (in forward)
 mubarrelrad=513.;  muforwardrad=800.; muoffset=10.;
// Step and boundary on Pt
 ptboun=1.;
 step=0.05;
 ptbmax=2.;
 ptstep=5.;
// size of window in phi-z.
//phiwinb
 phiwinb[0]=0.5;
 phiwinb[1]=0.5;
 phiwinb[2]=0.15;
 phiwinb[3]=0.15;
 phiwinb[4]=0.1;
 phiwinb[5]=0.1;
 phiwinb[6]=0.1;
 phiwinb[7]=0.08;
//phiwinf  
 phiwinf[0]=0.2;
 phiwinf[1]=0.2;
 phiwinf[2]=0.2;
 phiwinf[3]=0.2;
 phiwinf[4]=0.2;
 phiwinf[5]=0.2;
 phiwinf[6]=0.2;
 phiwinf[7]=0.2;
//ptwmax  
 ptwmax[0]=5.;
 ptwmax[1]=10.;
 ptwmax[2]=15.;
 ptwmax[3]=20.;
 ptwmax[4]=25.;
 ptwmax[5]=30.;
 ptwmax[6]=35.;
 ptwmax[7]=40.;
//ptwmin 
 ptwmin[0]=0.;
 ptwmin[1]=5.;
 ptwmin[2]=10.;
 ptwmin[3]=15.;
 ptwmin[4]=20.;
 ptwmin[5]=25.;
 ptwmin[6]=30.;
 ptwmin[7]=35.;
 zwin=25.;
 sigz=1.5;
 sigf=1.5;
 cylinderoffset=5.;
 diskoffset=5.;			     
 partrack=0.006;
  }

 float newparam[3];
 float newparamgt40[3];
 float forwparam[2];
// Coordinate of the initial point r (in barrel) and z (in forward)
 float mubarrelrad;
 float muforwardrad;
 float muoffset;
// Step and boundary on Pt
 float ptboun;
 float step;
 float ptbmax;
 float ptstep;
// size of window in phi-z.
 float phiwinb[8];
 float phiwinf[8];
 float ptwmax[8];
 float ptwmin[8];
 float zwin;
 float sigz;
 float sigf;
 float cylinderoffset;
 float diskoffset;			     
 float partrack;
};
}
#endif



