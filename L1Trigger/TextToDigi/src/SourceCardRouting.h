/*
  SourceCardRouting library
  Copyright Andrew Rose 2007
*/

#ifndef SOURCECARDROUTING_H
#define SOURCECARDROUTING_H

// The string and stream definitions
#include <iostream>
#include <string>

class SourceCardRouting {

 public:

  SourceCardRouting();
  ~SourceCardRouting();

//SFP arrays are SFP[cycle<2][sfp number<4]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7]
    void EMUtoSFP(	unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7],
			unsigned short (&Qbits)[7],
			unsigned short (&SFP)[2][4] 	);

//SFP arrays are SFP[cycle<2][sfp number<4]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7]
    void SFPtoEMU(	unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7],
			unsigned short (&Qbits)[7],
			unsigned short (&SFP)[2][4]	);

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC56HFtoSFP(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned short (&SFP)[2][4]	);

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC56HF(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned short (&SFP)[2][4]	);

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC012toSFP(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&SFP)[2][4]	);

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC012(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&SFP)[2][4]	);

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC234toSFP(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&SFP)[2][4]	);

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC234(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&SFP)[2][4]	);

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//SFP arrays are SFP[cycle<2][sfp number<4]

    void SFPtoVHDCI(	int RoutingMode,
			unsigned short (&SFP)[2][4],
			unsigned long (&VHDCI)[2][2] );


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void VHDCItoSFP(	int RoutingMode,
			unsigned short (&SFP)[2][4],
			unsigned long (&VHDCI)[2][2]	);

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7]

    void EMUtoVHDCI(	unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7],
			unsigned short (&Qbits)[7],
			unsigned long (&VHDCI)[2][2] 	);


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7]
    void VHDCItoEMU(	unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7],
			unsigned short (&Qbits)[7],
			unsigned long (&VHDCI)[2][2]	);

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC56HFtoVHDCI(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned long (&VHDCI)[2][2]	);

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC56HF(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned long (&VHDCI)[2][2]	);

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC012toVHDCI(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned long (&VHDCI)[2][2]);

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC012(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned long (&VHDCI)[2][2]	);

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC234toVHDCI(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned long (&VHDCI)[2][2]	);

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC234(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned long (&VHDCI)[2][2]	);

/***********************************************************************************************************************/

//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7]

    void EMUtoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7],
			unsigned short (&Qbits)[7],
			std::string &dataString	);

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
    void RC56HFtoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			std::string &dataString	);

//RC arrays are RC[receiver card number<7][region<2]
    void RC012toSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			std::string &dataString	);

//RC arrays are RC[receiver card number<7][region<2]
    void RC234toSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			std::string &dataString	);

/***********************************************************************************************************************/

//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			int RoutingMode,
			unsigned short (&SFP)[2][4],
			std::string &dataString	);

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void STRINGtoVHDCI(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			std::string &dataString,
			unsigned long (&VHDCI)[2][2]	);


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			std::string &dataString,
			unsigned long (&VHDCI)[2][2]	);

/***********************************************************************************************************************/

    void LogicalCardIDtoRoutingMode( unsigned short &logicalCardID,
				     int &RoutingMode,
				     int &RCTCrateNumber	);

   void RoutingModetoLogicalCardID( unsigned short &logicalCardID,
				     int &RoutingMode,
				     int &RCTCrateNumber	);



}; //class SourceCardRouting

#endif //~SOURCECARDMANAGER_H



















