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

  /// Struct of all data needed for running the emulator to SFP (sourcecard optical output) conversion.
  struct EmuToSfpData
  {
    // Input data.
    unsigned short eIsoRank[4];
    unsigned short eIsoCardId[4];
    unsigned short eIsoRegionId[4];
    unsigned short eNonIsoRank[4];
    unsigned short eNonIsoCardId[4];
    unsigned short eNonIsoRegionId[4];
    unsigned short mipBits[7][2];
    unsigned short qBits[7][2];
    // Output data.
    unsigned short sfp[2][4]; // [ cycle ] [ output number ]
  };

//SFP arrays are SFP[cycle<2][sfp number<4]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]
    void EMUtoSFP(	unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7][2],
			unsigned short (&Qbits)[7][2],
			unsigned short (&SFP)[2][4] 	) const;

//SFP arrays are SFP[cycle<2][sfp number<4]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]
    void SFPtoEMU(	unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7][2],
			unsigned short (&Qbits)[7][2],
			unsigned short (&SFP)[2][4]	) const;

/***********************************************************************************************************************/
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC56HFtoSFP(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned short (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC56HF(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned short (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC012toSFP(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC012(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC234toSFP(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC234(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&SFP)[2][4]	) const;

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//SFP arrays are SFP[cycle<2][sfp number<4]

    void SFPtoVHDCI(	int RoutingMode,
			unsigned short (&SFP)[2][4],
			unsigned long (&VHDCI)[2][2] ) const;


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void VHDCItoSFP(	int RoutingMode,
			unsigned short (&SFP)[2][4],
			unsigned long (&VHDCI)[2][2]	) const;

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

    void EMUtoVHDCI(	unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7][2],
			unsigned short (&Qbits)[7][2],
			unsigned long (&VHDCI)[2][2] 	) const;


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]
    void VHDCItoEMU(	unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7][2],
			unsigned short (&Qbits)[7][2],
			unsigned long (&VHDCI)[2][2]	) const;



/***********************************************************************************************************************/

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC56HFtoVHDCI(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned long (&VHDCI)[2][2]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC56HF(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned long (&VHDCI)[2][2]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC012toVHDCI(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned long (&VHDCI)[2][2]) const;

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC012(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned long (&VHDCI)[2][2]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC234toVHDCI(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned long (&VHDCI)[2][2]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC234(	unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned long (&VHDCI)[2][2]	) const;

/***********************************************************************************************************************/

//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

    void EMUtoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&eIsoRank)[4],
			unsigned short (&eIsoCardId)[4],
			unsigned short (&eIsoRegionId)[4],
			unsigned short (&eNonIsoRank)[4],
			unsigned short (&eNonIsoCardId)[4],
			unsigned short (&eNonIsoRegionId)[4],
			unsigned short (&MIPbits)[7][2],
			unsigned short (&Qbits)[7][2],
			std::string &dataString	) const;

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
    void RC56HFtoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			std::string &dataString	) const;

//RC arrays are RC[receiver card number<7][region<2]
    void RC012toSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			std::string &dataString	) const;

//RC arrays are RC[receiver card number<7][region<2]
    void RC234toSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			std::string &dataString	) const;

/***********************************************************************************************************************/

//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			int RoutingMode,
			unsigned short (&SFP)[2][4],
			std::string &dataString	) const;

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void STRINGtoVHDCI(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			std::string &dataString,
			unsigned long (&VHDCI)[2][2]	) const;


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			std::string &dataString,
			unsigned long (&VHDCI)[2][2]	) const;

/***********************************************************************************************************************/

    void LogicalCardIDtoRoutingMode( unsigned short &logicalCardID,
				     int &RoutingMode,
				     int &RCTCrateNumber	) const;

   void RoutingModetoLogicalCardID( unsigned short &logicalCardID,
				     int &RoutingMode,
				     int &RCTCrateNumber	) const;



}; //class SourceCardRouting

#endif //~SOURCECARDMANAGER_H













//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//These were going to be implimented but made things a lot more complicated than necessary
/*
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RCtoSFP(	int &RoutingMode,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned short (&SFP)[2][4] );


//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC(	int &RoutingMode,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned short (&SFP)[2][4] );*/

/*
//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RCtoSTRING(	unsigned short &logicalCardID,
			unsigned short &eventNumber,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			std::string &dataString	);
*/	

/*
//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RCtoVHDCI(	int &RoutingMode,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned long (&VHDCI)[2][2]	);

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC(	int &RoutingMode,
			unsigned short (&RC)[7][2],
			unsigned short (&RCof)[7][2],
			unsigned short (&RCtau)[7][2],
			unsigned short (&sisterRC)[7][2],
			unsigned short (&sisterRCof)[7][2],
			unsigned short (&sisterRCtau)[7][2],
			unsigned short (&HF)[4][2],
			unsigned short (&HFQ)[4][2],
			unsigned long (&VHDCI)[2][2]	);
*/













