/*
  SourceCardRouting library
  Copyright Andrew Rose 2007
*/

#ifndef SOURCECARDROUTING_H
#define SOURCECARDROUTING_H

// standard 16-bit and 32-bit data-types
#include <stdint.h>

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
    uint16_t eIsoRank[4];
    uint16_t eIsoCardId[4];
    uint16_t eIsoRegionId[4];
    uint16_t eNonIsoRank[4];
    uint16_t eNonIsoCardId[4];
    uint16_t eNonIsoRegionId[4];
    uint16_t mipBits[7][2];
    uint16_t qBits[7][2];
    // Output data.
    uint16_t sfp[2][4]; // [ cycle ] [ output number ]
  };

//SFP arrays are SFP[cycle<2][sfp number<4]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]
    void EMUtoSFP(	uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			uint16_t (&SFP)[2][4] 	) const;

//SFP arrays are SFP[cycle<2][sfp number<4]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]
    void SFPtoEMU(	uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			uint16_t (&SFP)[2][4]	) const;

/***********************************************************************************************************************/
//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC56HFtoSFP(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint16_t (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC56HF(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint16_t (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC012toSFP(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC012(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void RC234toSFP(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&SFP)[2][4]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC234(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&SFP)[2][4]	) const;

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//SFP arrays are SFP[cycle<2][sfp number<4]

    void SFPtoVHDCI(	int RoutingMode,
			uint16_t (&SFP)[2][4],
			uint32_t (&VHDCI)[2][2] ) const;


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void VHDCItoSFP(	int RoutingMode,
			uint16_t (&SFP)[2][4],
			uint32_t (&VHDCI)[2][2]	) const;

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

    void EMUtoVHDCI(	uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			uint32_t (&VHDCI)[2][2] 	) const;


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]
    void VHDCItoEMU(	uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			uint32_t (&VHDCI)[2][2]	) const;



/***********************************************************************************************************************/

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC56HFtoVHDCI(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint32_t (&VHDCI)[2][2]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC56HF(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint32_t (&VHDCI)[2][2]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC012toVHDCI(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint32_t (&VHDCI)[2][2]) const;

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC012(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint32_t (&VHDCI)[2][2]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RC234toVHDCI(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint32_t (&VHDCI)[2][2]	) const;

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC234(	uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint32_t (&VHDCI)[2][2]	) const;

/***********************************************************************************************************************/

//electron arrays are eIsoRank[candidate number<4]
//muon arrays are MIPbits[rec card number<7][region<2]

    void EMUtoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&eIsoRank)[4],
			uint16_t (&eIsoCardId)[4],
			uint16_t (&eIsoRegionId)[4],
			uint16_t (&eNonIsoRank)[4],
			uint16_t (&eNonIsoCardId)[4],
			uint16_t (&eNonIsoRegionId)[4],
			uint16_t (&MIPbits)[7][2],
			uint16_t (&Qbits)[7][2],
			std::string &dataString	) const;

//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
    void RC56HFtoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			std::string &dataString	) const;

//RC arrays are RC[receiver card number<7][region<2]
    void RC012toSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			std::string &dataString	) const;

//RC arrays are RC[receiver card number<7][region<2]
    void RC234toSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			std::string &dataString	) const;

/***********************************************************************************************************************/

//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			int RoutingMode,
			uint16_t (&SFP)[2][4],
			std::string &dataString	) const;

/***********************************************************************************************************************/

//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void STRINGtoVHDCI(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			std::string &dataString,
			uint32_t (&VHDCI)[2][2]	) const;


//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			std::string &dataString,
			uint32_t (&VHDCI)[2][2]	) const;

/***********************************************************************************************************************/

    void LogicalCardIDtoRoutingMode( uint16_t &logicalCardID,
				     int &RoutingMode,
				     int &RCTCrateNumber	) const;

   void RoutingModetoLogicalCardID( uint16_t &logicalCardID,
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
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint16_t (&SFP)[2][4] );


//RC arrays are RC[receiver card number<7][region<2]
//HF arrays are HF[eta<4][HF region<2]
//SFP arrays are SFP[cycle<2][sfp number<4]
    void SFPtoRC(	int &RoutingMode,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint16_t (&SFP)[2][4] );*/

/*
//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RCtoSTRING(	uint16_t &logicalCardID,
			uint16_t &eventNumber,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			std::string &dataString	);
*/	

/*
//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void RCtoVHDCI(	int &RoutingMode,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint32_t (&VHDCI)[2][2]	);

//RC arrays are RC[receiver card number<7][region<2]
//VHDCI arrays are VHDCI[VHDCI<2][cycle<2]
    void VHDCItoRC(	int &RoutingMode,
			uint16_t (&RC)[7][2],
			uint16_t (&RCof)[7][2],
			uint16_t (&RCtau)[7][2],
			uint16_t (&sisterRC)[7][2],
			uint16_t (&sisterRCof)[7][2],
			uint16_t (&sisterRCtau)[7][2],
			uint16_t (&HF)[4][2],
			uint16_t (&HFQ)[4][2],
			uint32_t (&VHDCI)[2][2]	);
*/













