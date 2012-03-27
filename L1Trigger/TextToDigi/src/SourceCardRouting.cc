/*
  SourceCardRouting library
  Copyright Andrew Rose 2007
*/

// Prototype class definition
#include "SourceCardRouting.h"			//hh"

// File streams
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
  SourceCardRouting::SourceCardRouting(){
	//std::cout<<"Constructor"<<std::endl;
	}

  SourceCardRouting::~SourceCardRouting(){
	//std::cout<<"Destructor"<<std::endl;
	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//SFP arrays are SFP[cycle<2][sfp number<4]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

void SourceCardRouting::EMUtoSFP(	uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			uint16_t (&SFP)[2][4] ) const{

SFP[0][0]=0;
SFP[1][0]=0x8000;

for (int i=0;i<7;i++){
	for (int j=0;j<2;j++){
		SFP[0][0] = SFP[0][0]|((MIPbits[i][j]&0x01)<<((2*i)+j) );
		SFP[1][0] = SFP[1][0]|((Qbits[i][j]&0x01)<<((2*i)+j) );
	}
}

      	SFP[0][1] = (eIsoRank[0]&0x3f)|((eIsoRegionId[0]&0x01)<<6)|((eIsoCardId[0]&0x07)<<7)|((eIsoRank[1]&0x7)<<10);
      	SFP[1][1] = 0x8000|(eIsoRank[2]&0x3f)|((eIsoRegionId[2]&0x01)<<6)|((eIsoCardId[2]&0x07)<<7)|((eIsoRank[3]&0x7)<<10);
      	SFP[0][2] = (eNonIsoRank[0]&0x3f)|((eNonIsoRegionId[0]&0x01)<<6)|((eNonIsoCardId[0]&0x07)<<7)|((eIsoRank[1]&0x38)<<7)|((eIsoRegionId[1]&0x01)<<13);
      	SFP[1][2] = 0x8000|(eNonIsoRank[2]&0x3f)|((eNonIsoRegionId[2]&0x01)<<6)|((eNonIsoCardId[2]&0x07)<<7)|((eIsoRank[3]&0x38)<<7)|((eIsoRegionId[3]&0x01)<<13);
      	SFP[0][3] = (eNonIsoRank[1]&0x3f)|((eNonIsoRegionId[1]&0x01)<<6)|((eNonIsoCardId[1]&0x07)<<7)|((eIsoCardId[1]&0x07)<<10);
      	SFP[1][3] = 0x8000|(eNonIsoRank[3]&0x3f)|((eNonIsoRegionId[3]&0x01)<<6)|((eNonIsoCardId[3]&0x07)<<7)|((eIsoCardId[3]&0x07)<<10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//SFP arrays are SFP[cycle<2][sfp number<4]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

void SourceCardRouting::SFPtoEMU(	uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			uint16_t (&SFP)[2][4] ) const{


	  for (int i=0; i<7;i++){
		for (int j=0; j<2;j++){
  	    		MIPbits[i][j] = (SFP[0][0]>>((2*i)+j) )&0x1;
  	    		Qbits[i][j] = (SFP[1][0]>>((2*i)+j) )&0x1;
		}
	  }

	eIsoRank[0] = SFP[0][1]&0x3f;
	eIsoRank[1] = ((SFP[0][1]>>10)&0x7)|((SFP[0][2]>>7)&0x38);
	eIsoRank[2] = SFP[1][1]&0x3f;
	eIsoRank[3] = ((SFP[1][1]>>10)&0x7)|((SFP[1][2]>>7)&0x38);

	eNonIsoRank[0] = SFP[0][2]&0x3f;
	eNonIsoRank[1] = SFP[0][3]&0x3f;
	eNonIsoRank[2] = SFP[1][2]&0x3f;
	eNonIsoRank[3] = SFP[1][3]&0x3f;

	eIsoRegionId[0] = (SFP[0][1]>>6)&0x1;
	eIsoRegionId[1] = (SFP[0][2]>>13)&0x1;
	eIsoRegionId[2] = (SFP[1][1]>>6)&0x1;
	eIsoRegionId[3] = (SFP[1][2]>>13)&0x1;

	eNonIsoRegionId[0] = (SFP[0][2]>>6)&0x1;
	eNonIsoRegionId[1] = (SFP[0][3]>>6)&0x1;
	eNonIsoRegionId[2] = (SFP[1][2]>>6)&0x1;
	eNonIsoRegionId[3] = (SFP[1][3]>>6)&0x1;

	eIsoCardId[0] = (SFP[0][1]>>7)&0x7;
	eIsoCardId[1] = (SFP[0][3]>>10)&0x7;
	eIsoCardId[2] = (SFP[1][1]>>7)&0x7;
	eIsoCardId[3] = (SFP[1][3]>>10)&0x7;

	eNonIsoCardId[0] = (SFP[0][2]>>7)&0x7;
	eNonIsoCardId[1] = (SFP[0][3]>>7)&0x7;
	eNonIsoCardId[2] = (SFP[1][2]>>7)&0x7;
	eNonIsoCardId[3] = (SFP[1][3]>>7)&0x7;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
 
void SourceCardRouting::RC56HFtoSFP(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint16_t (&SFP)[2][4]) const{

	SFP[0][0] = (RC[5][0]&0x3ff)|((RCof[5][0]&0x1)<<10)|((RCtau[5][0]&0x1)<<11)|((HFQ[0][0]&0x1)<<12)|((HFQ[1][0]&0x01)<<13)|((HF[0][0]&0x01)<<14);
	SFP[1][0] = 0x8000|(RC[5][1]&0x3ff)|((RCof[5][1]&0x1)<<10)|((RCtau[5][1]&0x1)<<11)|((HFQ[2][0]&0x1)<<12)|((HFQ[3][0]&0x01)<<13)|((HF[2][0]&0x01)<<14);
      	SFP[0][1] = (RC[6][0]&0x3ff)|((RCof[6][0]&0x1)<<10)|((RCtau[6][0]&0x1)<<11)|((HFQ[0][1]&0x1)<<12)|((HFQ[1][1]&0x01)<<13)|((HF[0][1]&0x01)<<14);
      	SFP[1][1] = 0x8000|(RC[6][1]&0x3ff)|((RCof[6][1]&0x1)<<10)|((RCtau[6][1]&0x1)<<11)|((HFQ[2][1]&0x1)<<12)|((HFQ[3][1]&0x01)<<13)|((HF[2][1]&0x01)<<14);
      	SFP[0][2] = ((HF[0][0]>>1)&0x7f)|((HF[1][0]&0xff)<<7);
      	SFP[1][2] = 0x8000|((HF[2][0]>>1)&0x7f)|((HF[3][0]&0xff)<<7);
      	SFP[0][3] = ((HF[0][1]>>1)&0x7f)|((HF[1][1]&0xff)<<7);
      	SFP[1][3] = 0x8000|((HF[2][1]>>1)&0x7f)|((HF[3][1]&0xff)<<7);


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
 
    void SourceCardRouting::SFPtoRC56HF(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint16_t (&SFP)[2][4]) const{

	RC[5][0]=SFP[0][0]&0x3ff;
	RC[5][1]=SFP[1][0]&0x3ff;
	RC[6][0]=SFP[0][1]&0x3ff;
	RC[6][1]=SFP[1][1]&0x3ff;

	RCof[5][0]=(SFP[0][0]>>10)&0x1;
	RCof[5][1]=(SFP[1][0]>>10)&0x1;
	RCof[6][0]=(SFP[0][1]>>10)&0x1;
	RCof[6][1]=(SFP[1][1]>>10)&0x1;

	RCtau[5][0]=(SFP[0][0]>>11)&0x1;
	RCtau[5][1]=(SFP[1][0]>>11)&0x1;
	RCtau[6][0]=(SFP[0][1]>>11)&0x1;
	RCtau[6][1]=(SFP[1][1]>>11)&0x1;

	HFQ[0][0]=(SFP[0][0]>>12)&0x1;
	HFQ[1][0]=(SFP[0][0]>>13)&0x1;
	HFQ[2][0]=(SFP[1][0]>>12)&0x1;
	HFQ[3][0]=(SFP[1][0]>>13)&0x1;

	HFQ[0][1]=(SFP[0][1]>>12)&0x1;
	HFQ[1][1]=(SFP[0][1]>>13)&0x1;
	HFQ[2][1]=(SFP[1][1]>>12)&0x1;
	HFQ[3][1]=(SFP[1][1]>>13)&0x1;

	HF[0][0]=((SFP[0][2]&0x7f)<<1)|((SFP[0][0]>>14)&0x01);
	HF[1][0]=(SFP[0][2]>>7)&0xff;
	HF[2][0]=((SFP[1][2]&0x7f)<<1)|((SFP[1][0]>>14)&0x01);
	HF[3][0]=(SFP[1][2]>>7)&0xff;

	HF[0][1]=((SFP[0][3]&0x7f)<<1)|((SFP[0][1]>>14)&0x01);
	HF[1][1]=(SFP[0][3]>>7)&0xff;
	HF[2][1]=((SFP[1][3]&0x7f)<<1)|((SFP[1][1]>>14)&0x01);
	HF[3][1]=(SFP[1][3]>>7)&0xff;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
 
    void SourceCardRouting::RC012toSFP(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&SFP)[2][4]) const{

	SFP[0][0] = (RC[0][0]&0x3ff)|((RCof[0][0]&0x1)<<10)|((RCtau[0][0]&0x1)<<11)|((RC[2][0]&0x7)<<12);
	SFP[1][0] = 0x8000|(RC[0][1]&0x3ff)|((RCof[0][1]&0x1)<<10)|((RCtau[0][1]&0x1)<<11)|((RC[2][1]&0x7)<<12);

      	SFP[0][1] = (RC[1][0]&0x3ff)|((RCof[1][0]&0x1)<<10)|((RCtau[1][0]&0x1)<<11)|((RC[2][0]&0x38)<<9);
      	SFP[1][1] = 0x8000|(RC[1][1]&0x3ff)|((RCof[1][1]&0x1)<<10)|((RCtau[1][1]&0x1)<<11)|((RC[2][1]&0x38)<<9);

	SFP[0][2] = (RC[0][0]&0x3ff)|((RCof[0][0]&0x1)<<10)|((RCtau[0][0]&0x1)<<11);
	SFP[1][2] = 0x8000|(RC[0][1]&0x3ff)|((RCof[0][1]&0x1)<<10)|((RCtau[0][1]&0x1)<<11);

      	SFP[0][3] = (RC[1][0]&0x3ff)|((RCof[1][0]&0x1)<<10)|((RCtau[1][0]&0x1)<<11);
      	SFP[1][3] = 0x8000|(RC[1][1]&0x3ff)|((RCof[1][1]&0x1)<<10)|((RCtau[1][1]&0x1)<<11);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
 
    void SourceCardRouting::SFPtoRC012(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&SFP)[2][4]) const{

	RC[0][0]=SFP[0][0]&0x3ff;
	RC[0][1]=SFP[1][0]&0x3ff;
	RC[1][0]=SFP[0][1]&0x3ff;
	RC[1][1]=SFP[1][1]&0x3ff;
	RC[2][0]=(RC[2][0]&0x3c0)|((SFP[0][0]&0x7000)>>12)|((SFP[0][1]&0x7000)>>9);
	RC[2][1]=(RC[2][1]&0x3c0)|((SFP[1][0]&0x7000)>>12)|((SFP[1][1]&0x7000)>>9);

	RCof[0][0]=(SFP[0][0]>>10)&0x1;
	RCof[0][1]=(SFP[1][0]>>10)&0x1;
	RCof[1][0]=(SFP[0][1]>>10)&0x1;
	RCof[1][1]=(SFP[1][1]>>10)&0x1;

	RCtau[0][0]=(SFP[0][0]>>11)&0x1;
	RCtau[0][1]=(SFP[1][0]>>11)&0x1;
	RCtau[1][0]=(SFP[0][1]>>11)&0x1;
	RCtau[1][1]=(SFP[1][1]>>11)&0x1;


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
 
    void SourceCardRouting::RC234toSFP(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&SFP)[2][4]) const{

	SFP[0][0] = (RC[3][0]&0x3ff)|((RCof[3][0]&0x1)<<10)|((RCtau[3][0]&0x1)<<11)|((RC[2][0]&0x1c0)<<6);
	SFP[1][0] = 0x8000|(RC[3][1]&0x3ff)|((RCof[3][1]&0x1)<<10)|((RCtau[3][1]&0x1)<<11)|((RC[2][1]&0x1c0)<<6);

      	SFP[0][1] = (RC[4][0]&0x3ff)|((RCof[4][0]&0x1)<<10)|((RCtau[4][0]&0x1)<<11)|((RC[2][0]&0x200)<<3)|((RCof[2][0]&0x1)<<13)|((RCtau[2][0]&0x1)<<14);
      	SFP[1][1] = 0x8000|(RC[4][1]&0x3ff)|((RCof[4][1]&0x1)<<10)|((RCtau[4][1]&0x1)<<11)|((RC[2][1]&0x200)<<3)|((RCof[2][1]&0x1)<<13)|((RCtau[2][1]&0x1)<<14);

	SFP[0][2] = (sisterRC[3][0]&0x3ff)|((sisterRCof[3][0]&0x1)<<10)|((sisterRCtau[3][0]&0x1)<<11)|((sisterRC[2][0]&0x1c0)<<6);
	SFP[1][2] = 0x8000|(sisterRC[3][1]&0x3ff)|((sisterRCof[3][1]&0x1)<<10)|((sisterRCtau[3][1]&0x1)<<11)|((sisterRC[2][1]&0x1c0)<<6);

      	SFP[0][3] = (sisterRC[4][0]&0x3ff)|((sisterRCof[4][0]&0x1)<<10)|((sisterRCtau[4][0]&0x1)<<11)|((sisterRC[2][0]&0x200)<<3)|((sisterRCof[2][0]&0x1)<<13)|((sisterRCtau[2][0]&0x1)<<14);
      	SFP[1][3] = 0x8000|(sisterRC[4][1]&0x3ff)|((sisterRCof[4][1]&0x1)<<10)|((sisterRCtau[4][1]&0x1)<<11)|((sisterRC[2][1]&0x200)<<3)|((sisterRCof[2][1]&0x1)<<13)|((sisterRCtau[2][1]&0x1)<<14);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
 
     void SourceCardRouting::SFPtoRC234(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&SFP)[2][4]) const{

	RC[2][0]=(RC[2][0]&0x3f)|((SFP[0][0]&0x7000)>>6)|((SFP[0][1]&0x1000)>>3);
	RC[3][0]=SFP[0][0]&0x3ff;
	RC[4][0]=SFP[0][1]&0x3ff;
	RC[2][1]=(RC[2][1]&0x3f)|((SFP[1][0]&0x7000)>>6)|((SFP[1][1]&0x1000)>>3);
	RC[3][1]=SFP[1][0]&0x3ff;
	RC[4][1]=SFP[1][1]&0x3ff;

	RCof[2][0]=(SFP[0][1]>>13)&0x1;
	RCof[3][0]=(SFP[0][0]>>10)&0x1;
	RCof[4][0]=(SFP[0][1]>>10)&0x1;
	RCof[2][1]=(SFP[1][1]>>13)&0x1;
	RCof[3][1]=(SFP[1][0]>>10)&0x1;
	RCof[4][1]=(SFP[1][1]>>10)&0x1;

	RCtau[2][0]=(SFP[0][1]>>14)&0x1;
	RCtau[3][0]=(SFP[0][0]>>11)&0x1;
	RCtau[4][0]=(SFP[0][1]>>11)&0x1;
	RCtau[2][1]=(SFP[1][1]>>14)&0x1;
	RCtau[3][1]=(SFP[1][0]>>11)&0x1;
	RCtau[4][1]=(SFP[1][1]>>11)&0x1;

	sisterRC[2][0]=(sisterRC[2][0]&~0x3C0)|((SFP[0][2]&0x7000)>>6)|((SFP[0][3]&0x1000)>>3);
	sisterRC[3][0]=SFP[0][2]&0x3ff;
	sisterRC[4][0]=SFP[0][3]&0x3ff;
	sisterRC[2][1]=(sisterRC[2][1]&~0x3C0)|((SFP[1][2]&0x7000)>>6)|((SFP[1][3]&0x1000)>>3);
	sisterRC[3][1]=SFP[1][2]&0x3ff;
	sisterRC[4][1]=SFP[1][3]&0x3ff;

	sisterRCof[2][0]=(SFP[0][3]>>13)&0x1;
	sisterRCof[3][0]=(SFP[0][2]>>10)&0x1;
	sisterRCof[4][0]=(SFP[0][3]>>10)&0x1;
	sisterRCof[2][1]=(SFP[1][3]>>13)&0x1;
	sisterRCof[3][1]=(SFP[1][2]>>10)&0x1;
	sisterRCof[4][1]=(SFP[1][3]>>10)&0x1;

	sisterRCtau[2][0]=(SFP[0][3]>>14)&0x1;
	sisterRCtau[3][0]=(SFP[0][2]>>11)&0x1;
	sisterRCtau[4][0]=(SFP[0][3]>>11)&0x1;
	sisterRCtau[2][1]=(SFP[1][3]>>14)&0x1;
	sisterRCtau[3][1]=(SFP[1][2]>>11)&0x1;
	sisterRCtau[4][1]=(SFP[1][3]>>11)&0x1;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//SFP arrays are SFP[cycle<2][sfp number<4]

    void SourceCardRouting::SFPtoVHDCI(	int RoutingMode,
			uint16_t (&SFP)[2][4],
			uint32_t (&VHDCI)[2][2] ) const{

uint16_t sfp_reverse[2][4]={{0}};

for (int i=0; i<2;i++){
	for(int j=0; j<4;j++){
		for (int k=0; k<16;k++){
			sfp_reverse[i][j]=sfp_reverse[i][j]|(((SFP[i][j]>>k)&0x01)<<(15-k));
		}
		//std::cout <<hex<< SFP[i][j]<<'\t'<<sfp_reverse[i][j]<<std::endl;
	}
}

	switch (RoutingMode){
		case 0:
			VHDCI[0][0]=(SFP[0][1]&0x3ff)|((SFP[0][1]&0x1c00)<<1)|((SFP[0][2]&0x3c00)<<4)|((SFP[0][3]&0x1c00)<<8)|((SFP[0][0]&0xff)<<22);
			VHDCI[0][1]=(SFP[1][1]&0x3ff)|((SFP[1][1]&0x1c00)<<1)|((SFP[1][2]&0x3c00)<<4)|((SFP[1][3]&0x1c00)<<8)|((SFP[1][0]&0xff)<<22);
			VHDCI[1][0]=(SFP[0][2]&0x3ff)|((SFP[0][3]&0x3ff)<<11)|((SFP[0][0]&0x3f00)<<14);
			VHDCI[1][1]=(SFP[1][2]&0x3ff)|((SFP[1][3]&0x3ff)<<11)|((SFP[1][0]&0x3f00)<<14);
		break;
		case 1:
			VHDCI[0][0]=(SFP[0][0]&0xfff)|((SFP[0][1]&0x7)<<12)|((SFP[0][2]&0x80)<<10)|((SFP[0][0]&0x4000)<<4)|((sfp_reverse[0][1]&0xc)<<17)|((sfp_reverse[0][0]&0xc)<<19)|((sfp_reverse[0][1]&0x1ff0)<<19);
			VHDCI[0][1]=(SFP[1][0]&0xfff)|((SFP[1][1]&0x7)<<12)|((SFP[1][2]&0x80)<<10)|((SFP[1][0]&0x4000)<<4)|((sfp_reverse[1][1]&0xc)<<17)|((sfp_reverse[1][0]&0xc)<<19)|((sfp_reverse[1][1]&0x1ff0)<<19);

			VHDCI[1][0]=(SFP[0][1]&0x4000)|((SFP[0][3]&0x80)<<24);
			VHDCI[1][1]=(SFP[1][1]&0x4000)|((SFP[1][3]&0x80)<<24);

			for (int i=0; i<7;i++){
				VHDCI[1][0]=VHDCI[1][0]|(((SFP[0][2]>>i)&0x1)<<(2*i))|(((SFP[0][2]>>(i+8))&0x1)<<((2*i)+1))|(((sfp_reverse[0][3]>>(i+1))&0x1)<<((2*i)+17))|(((sfp_reverse[0][3]>>(i+9))&0x1)<<((2*i)+18));
				VHDCI[1][1]=VHDCI[1][1]|(((SFP[1][2]>>i)&0x1)<<(2*i))|(((SFP[1][2]>>(i+8))&0x1)<<((2*i)+1))|(((sfp_reverse[1][3]>>(i+1))&0x1)<<((2*i)+17))|(((sfp_reverse[1][3]>>(i+9))&0x1)<<((2*i)+18));		
			}
		break;
		case 2:
			VHDCI[0][0]=(SFP[0][0]&0xfff)|((SFP[0][1]&0x7)<<12)|((sfp_reverse[0][1]&0xe)<<16)|((sfp_reverse[0][0]&0xe)<<19)|((sfp_reverse[0][1]&0x1ff0)<<19);
			VHDCI[0][1]=(SFP[1][0]&0xfff)|((SFP[1][1]&0x7)<<12)|((sfp_reverse[1][1]&0xe)<<16)|((sfp_reverse[1][0]&0xe)<<19)|((sfp_reverse[1][1]&0x1ff0)<<19);
			VHDCI[1][0]=0;
			VHDCI[1][1]=0;
		break;
		case 3:
			VHDCI[0][0]=((SFP[0][0]>>12)&0x7)|((SFP[0][1]>>9)&0x38)|((SFP[0][0]&0x1ff)<<6)|((sfp_reverse[0][1]&0xfff0)<<13)|((sfp_reverse[0][0]&0x70)<<25);
			VHDCI[0][1]=((SFP[1][0]>>12)&0x7)|((SFP[1][1]>>9)&0x38)|((SFP[1][0]&0x1ff)<<6)|((sfp_reverse[1][1]&0xfff0)<<13)|((sfp_reverse[1][0]&0x70)<<25);

			VHDCI[1][0]=((SFP[0][2]>>12)&0x7)|((SFP[0][3]>>9)&0x38)|((SFP[0][2]&0x1ff)<<6)|((sfp_reverse[0][3]&0xfff0)<<13)|((sfp_reverse[0][2]&0x70)<<25);
			VHDCI[1][1]=((SFP[1][2]>>12)&0x7)|((SFP[1][3]>>9)&0x38)|((SFP[1][2]&0x1ff)<<6)|((sfp_reverse[1][3]&0xfff0)<<13)|((sfp_reverse[1][2]&0x70)<<25);
		break;
		default:
			VHDCI[0][0]=0;
			VHDCI[0][1]=0;
			VHDCI[1][0]=0;
			VHDCI[1][1]=0;
		}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SourceCardRouting::VHDCItoSFP(	int RoutingMode,
			uint16_t (&SFP)[2][4],
			uint32_t (&VHDCI)[2][2]	) const{

uint16_t VHDCI_reverse[2][4]={{0}};

for (int i=0; i<2;i++){
	for(int j=0; j<2;j++){
		for (int k=0; k<32;k++){
			VHDCI_reverse[i][j]=VHDCI_reverse[i][j]|(((VHDCI[i][j]>>k)&0x01)<<(31-k));
		}
	}
}

	switch (RoutingMode){
		case 0:
			SFP[0][0]=((VHDCI[0][0]>>22)&0xff)|((VHDCI[1][0]>>14)&0x3f00);
			SFP[1][0]=0x8000|((VHDCI[0][1]>>22)&0xff)|((VHDCI[1][1]>>14)&0x3f00);
			SFP[0][1]=(VHDCI[0][0]&0x3ff)|((VHDCI[0][0]>>1)&0x1c00);
			SFP[1][1]=0x8000|(VHDCI[0][1]&0x3ff)|((VHDCI[0][1]>>1)&0x1c00);
			SFP[0][2]=(VHDCI[1][0]&0x3ff)|((VHDCI[0][0]>>4)&0x3c00);
			SFP[1][2]=0x8000|(VHDCI[1][1]&0x3ff)|((VHDCI[0][1]>>4)&0x3c00);
			SFP[0][3]=((VHDCI[1][0]>>11)&0x3ff)|((VHDCI[0][0]>>8)&0x1c00);
			SFP[1][3]=0x8000|((VHDCI[1][1]>>11)&0x3ff)|((VHDCI[0][1]>>8)&0x1c00);
		break;
		case 1:
			SFP[0][0]=(VHDCI[0][0]&0xfff)|((VHDCI_reverse[0][0]&0x600)<<3)|((VHDCI_reverse[0][0]&0x2000)<<1);
			SFP[1][0]=0x8000|(VHDCI[0][1]&0xfff)|((VHDCI_reverse[0][1]&0x600)<<3)|((VHDCI_reverse[0][1]&0x2000)<<1);
			SFP[0][1]=((VHDCI[0][0]&0x7000)>>12)|((VHDCI_reverse[0][0]&0x1ff)<<3)|((VHDCI_reverse[0][0]&0x1800)<<1)|(VHDCI[1][0]&0x4000);
			SFP[1][1]=0x8000|((VHDCI[0][1]&0x7000)>>12)|((VHDCI_reverse[0][1]&0x1ff)<<3)|((VHDCI_reverse[0][1]&0x1800)<<1)|(VHDCI[1][1]&0x4000);

			SFP[0][2]=((VHDCI[0][0]&0x20000)>>10);
			SFP[1][2]=0x8000|((VHDCI[0][1]&0x20000)>>10);
			SFP[0][3]=((VHDCI[1][0]&0x20000)>>3);
			SFP[1][3]=0x8000|((VHDCI[1][1]&0x20000)>>3);
			for (int i=0; i<7;i++){
				SFP[0][2]=SFP[0][2]|(((VHDCI[1][0]>>(2*i))&0x1)<<i)|(((VHDCI[1][0]>>((2*i)+1))&0x1)<<(i+8));
				SFP[1][2]=SFP[1][2]|(((VHDCI[1][1]>>(2*i))&0x1)<<i)|(((VHDCI[1][1]>>((2*i)+1))&0x1)<<(i+8));
				SFP[0][3]=SFP[0][3]|(((VHDCI_reverse[1][0]>>((2*i)+1))&0x1)<<i)|(((VHDCI_reverse[1][0]>>(2*i))&0x1)<<(i+7));
				SFP[1][3]=SFP[1][3]|(((VHDCI_reverse[1][1]>>((2*i)+1))&0x1)<<i)|(((VHDCI_reverse[1][1]>>(2*i))&0x1)<<(i+7));
			}
		break;
		case 2:
			SFP[0][0]=(VHDCI[0][0]&0xfff)|((VHDCI_reverse[0][0]&0xe00)<<3);
			SFP[1][0]=0x8000|(VHDCI[0][1]&0xfff)|((VHDCI_reverse[0][1]&0xe00)<<3);
			SFP[0][1]=((VHDCI[0][0]&0x7000)>>12)|((VHDCI_reverse[0][0]&0x1ff)<<3)|(VHDCI_reverse[0][0]&0x7000);
			SFP[1][1]=0x8000|((VHDCI[0][1]&0x7000)>>12)|((VHDCI_reverse[0][1]&0x1ff)<<3)|(VHDCI_reverse[0][1]&0x7000);
			SFP[0][2]=(VHDCI[0][0]&0xfff);
			SFP[1][2]=0x8000|(VHDCI[0][1]&0xfff);
			SFP[0][3]=((VHDCI[0][0]&0x7000)>>12)|((VHDCI_reverse[0][0]&0x1ff)<<3);
			SFP[1][3]=0x8000|((VHDCI[0][1]&0x7000)>>12)|((VHDCI_reverse[0][1]&0x1ff)<<3);
		break;
		case 3:
			SFP[0][0]=((VHDCI[0][0]&0x7fc0)>>6)|((VHDCI_reverse[0][0]&0x7)<<9)|((VHDCI[0][0]&0x7)<<12);
			SFP[1][0]=0x8000|((VHDCI[0][1]&0x7fc0)>>6)|((VHDCI_reverse[0][1]&0x7)<<9)|((VHDCI[0][1]&0x7)<<12);
			SFP[0][1]=((VHDCI_reverse[0][0]&0x7ff8)>>3)|((VHDCI[0][0]&0x38)<<9);
			SFP[1][1]=0x8000|((VHDCI_reverse[0][1]&0x7ff8)>>3)|((VHDCI[0][1]&0x38)<<9);
			SFP[0][2]=((VHDCI[1][0]&0x7fc0)>>6)|((VHDCI_reverse[1][0]&0x7)<<9)|((VHDCI[1][0]&0x7)<<12);
			SFP[1][2]=0x8000|((VHDCI[1][1]&0x7fc0)>>6)|((VHDCI_reverse[1][1]&0x7)<<9)|((VHDCI[1][1]&0x7)<<12);
			SFP[0][3]=((VHDCI_reverse[1][0]&0x7ff8)>>3)|((VHDCI[1][0]&0x38)<<9);
			SFP[1][3]=0x8000|((VHDCI_reverse[1][1]&0x7ff8)>>3)|((VHDCI[1][1]&0x38)<<9);
		break;
		default:
			SFP[0][0]=0;
			SFP[1][0]=0x8000;
			SFP[0][1]=0;
			SFP[1][1]=0x8000;
			SFP[0][2]=0;
			SFP[1][2]=0x8000;
			SFP[0][3]=0;
			SFP[1][3]=0x8000;
		}
}
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

    void SourceCardRouting::EMUtoVHDCI(	uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			uint32_t (&VHDCI)[2][2] ) const{

	uint16_t SFP[2][4]={{0}};
	EMUtoSFP(eIsoRank,eIsoCardId,eIsoRegionId,eNonIsoRank,eNonIsoCardId,eNonIsoRegionId,MIPbits,Qbits,SFP);
	SFPtoVHDCI(0,SFP,VHDCI);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

    void SourceCardRouting::VHDCItoEMU(	uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			uint32_t (&VHDCI)[2][2] ) const{

	uint16_t SFP[2][4]={{0}};
	VHDCItoSFP(0,SFP,VHDCI);
	SFPtoEMU(eIsoRank,eIsoCardId,eIsoRegionId,eNonIsoRank,eNonIsoCardId,eNonIsoRegionId,MIPbits,Qbits,SFP);

}





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
 
    void SourceCardRouting::RC56HFtoVHDCI(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint32_t (&VHDCI)[2][2] ) const{

	uint16_t SFP[2][4]={{0}};
  	RC56HFtoSFP(RC,RCof,RCtau,HF,HFQ,SFP);
	SFPtoVHDCI(1,SFP,VHDCI);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
 
    void SourceCardRouting::VHDCItoRC56HF(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint32_t (&VHDCI)[2][2] ) const{

	uint16_t SFP[2][4]={{0}};
	VHDCItoSFP(1,SFP,VHDCI);
  	SFPtoRC56HF(RC,RCof,RCtau,HF,HFQ,SFP);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
 
    void SourceCardRouting::RC012toVHDCI(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint32_t (&VHDCI)[2][2] ) const{

	uint16_t SFP[2][4]={{0}};
 	RC012toSFP(RC,RCof,RCtau,SFP);
	SFPtoVHDCI(2,SFP,VHDCI);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
 
    void SourceCardRouting::VHDCItoRC012(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint32_t (&VHDCI)[2][2] ) const{

	uint16_t SFP[2][4]={{0}};
	VHDCItoSFP(2,SFP,VHDCI);
 	SFPtoRC012(RC,RCof,RCtau,SFP);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
 
    void SourceCardRouting::RC234toVHDCI(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint32_t (&VHDCI)[2][2] ) const{

	uint16_t SFP[2][4]={{0}};
  	RC234toSFP(RC,RCof,RCtau,sisterRC,sisterRCof,sisterRCtau,SFP);
	SFPtoVHDCI(3,SFP,VHDCI);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
 
     void SourceCardRouting::VHDCItoRC234(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint32_t (&VHDCI)[2][2] ) const{

	uint16_t SFP[2][4]={{0}};
	VHDCItoSFP(3,SFP,VHDCI);
  	SFPtoRC234(RC,RCof,RCtau,sisterRC,sisterRCof,sisterRCtau,SFP);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

    void SourceCardRouting::EMUtoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			std::string &dataString ) const{

		uint32_t VHDCI[2][2]={{0}};
		EMUtoVHDCI(eIsoRank,eIsoCardId,eIsoRegionId,eNonIsoRank,eNonIsoCardId,eNonIsoRegionId,MIPbits,Qbits,VHDCI);
		VHDCItoSTRING (logicalCardID, eventNumber, dataString, VHDCI);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]

 
    void SourceCardRouting::RC56HFtoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			std::string &dataString ) const{

		uint32_t VHDCI[2][2]={{0}};
  		RC56HFtoVHDCI(RC,RCof,RCtau,HF,HFQ,VHDCI);
		VHDCItoSTRING (logicalCardID, eventNumber, dataString, VHDCI);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]

 
    void SourceCardRouting::RC012toSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			std::string &dataString ) const{

		uint32_t VHDCI[2][2]={{0}};
  		RC012toVHDCI(RC,RCof,RCtau,VHDCI);
		VHDCItoSTRING (logicalCardID, eventNumber, dataString, VHDCI);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]

 
    void SourceCardRouting::RC234toSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			std::string &dataString ) const{

		uint32_t VHDCI[2][2]={{0}};
  		RC234toVHDCI(RC,RCof,RCtau,sisterRC,sisterRCof,sisterRCtau,VHDCI);
		VHDCItoSTRING (logicalCardID, eventNumber, dataString, VHDCI);

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SourceCardRouting::SFPtoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			int RoutingMode,
			uint16_t (&SFP)[2][4],
			std::string &dataString	) const{

		uint32_t VHDCI[2][2]={{0}};
		SFPtoVHDCI(RoutingMode,SFP,VHDCI);
		VHDCItoSTRING (logicalCardID, eventNumber, dataString, VHDCI);

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void SourceCardRouting::STRINGtoVHDCI(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			std::string &dataString,
			uint32_t (&VHDCI)[2][2]	) const{

		stringstream temp;
		
		if (dataString!=""){
	        	temp << dataString << std::endl;
	        	temp >> dec >> eventNumber;
	       		temp >> dec >> logicalCardID;
			temp >> hex >> VHDCI[0][0];
			temp >> hex >> VHDCI[0][1];
			temp >> hex >> VHDCI[1][0];
			temp >> hex >> VHDCI[1][1];
		}else{
	        	eventNumber=65535;
	       		logicalCardID=65535;
			VHDCI[0][0]=0;
			VHDCI[0][1]=0;
			VHDCI[1][0]=0;
			VHDCI[1][1]=0;
		}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void SourceCardRouting::VHDCItoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			std::string &dataString,
			uint32_t (&VHDCI)[2][2]	) const{

		stringstream temp;

		temp << dec << eventNumber << '\t';
		temp << dec << logicalCardID << '\t';
		temp << hex << setw(8) << setfill('0') << VHDCI[0][0] << '\t';
		temp << hex << setw(8) << setfill('0') << VHDCI[0][1] << '\t';
		temp << hex << setw(8) << setfill('0') << VHDCI[1][0] << '\t';
		temp << hex << setw(8) << setfill('0') << VHDCI[1][1] << std::endl;
		dataString = temp.str();
}
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void SourceCardRouting::LogicalCardIDtoRoutingMode( uint16_t &logicalCardID,
				     int &RoutingMode,
				     int &RCTCrateNumber	) const{
		
		RCTCrateNumber = (logicalCardID>>3);
		if ( (logicalCardID&0x4) != 0)RCTCrateNumber+=9;
		RoutingMode = (logicalCardID&0x3); 

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   void SourceCardRouting::RoutingModetoLogicalCardID( uint16_t &logicalCardID,
				     int &RoutingMode,
				     int &RCTCrateNumber	) const{

		logicalCardID = ((RCTCrateNumber%9)<<3)|(RCTCrateNumber>8?0x4:0x0)|(RoutingMode&0x3);

}





























//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//These were going to be implimented but made things a lot more complicated than necessary

/*
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SourceCardRouting::RCtoSFP(	int &RoutingMode,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint16_t (&SFP)[2][4] ){

	switch(RoutingMode){
		case 1:
  			RC56HFtoSFP(RC,RCof,RCtau,HF,HFQ,SFP);
			break;
		case 2:
 			RC012toSFP(RC,RCof,RCtau,SFP);		
			break;
		case 3:
		 	RC234toSFP(RC,RCof,RCtau,sisterRC,sisterRCof,sisterRCtau,SFP);
			break;
		default:
			break;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SourceCardRouting::SFPtoRC(	int &RoutingMode,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint16_t (&SFP)[2][4] ){

	switch(RoutingMode){
		case 1:
  			SFPtoRC56HF(RC,RCof,RCtau,HF,HFQ,SFP);
			break;
		case 2:
 			SFPtoRC012(RC,RCof,RCtau,SFP);		
			break;
		case 3:
		 	SFPtoRC234(RC,RCof,RCtau,sisterRC,sisterRCof,sisterRCtau,SFP);
			break;
		default:
			break;
	}
}
*/

/*
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void SourceCardRouting::RCtoVHDCI(	int &RoutingMode,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint32_t (&VHDCI)[2][2]	){

	switch(RoutingMode){
		case 1:
  			RC56HFtoVHDCI(RC,RCof,RCtau,HF,HFQ,VHDCI);
			break;
		case 2:
 			RC012toVHDCI(RC,RCof,RCtau,VHDCI);		
			break;
		case 3:
		 	RC234toVHDCI(RC,RCof,RCtau,sisterRC,sisterRCof,sisterRCtau,VHDCI);
			break;
		default:
			break;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void SourceCardRouting::VHDCItoRC(	int &RoutingMode,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint32_t (&VHDCI)[2][2]	){

	switch(RoutingMode){
		case 1:
  			VHDCItoRC56HF(RC,RCof,RCtau,HF,HFQ,VHDCI);
			break;
		case 2:
 			VHDCItoRC012(RC,RCof,RCtau,VHDCI);		
			break;
		case 3:
		 	VHDCItoRC234(RC,RCof,RCtau,sisterRC,sisterRCof,sisterRCtau,VHDCI);
			break;
		default:
			break;
	}
}
*/

