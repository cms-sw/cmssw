#ifndef DataFormats_HGCalDigi_interface_HGCalDigiTriggerSoA_h
#define DataFormats_HGCalDigi_interface_HGCalDigiTriggerSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcaldigi {

  // // Generate structure of arrays (SoA) layout with Digi dataformat
  // GENERATE_SOA_LAYOUT(HGCalDigiTriggerSoALayout,
  //                     SOA_COLUMN(uint8_t, algo),
  //                     SOA_COLUMN(bool, valid),
  //                     SOA_COLUMN(uint8_t, BXm3_location),
  //                     SOA_COLUMN(uint8_t, BXm2_location),
  //                     SOA_COLUMN(uint8_t, BXm1_location),
  //                     SOA_COLUMN(uint8_t, BX0_location),
  //                     SOA_COLUMN(uint8_t, BXp1_location),
  //                     SOA_COLUMN(uint8_t, BXp2_location),
  //                     SOA_COLUMN(uint8_t, BXp3_location),
  //                     SOA_COLUMN(uint16_t, BXm3_energy),
  //                     SOA_COLUMN(uint16_t, BXm2_energy),
  //                     SOA_COLUMN(uint16_t, BXm1_energy),
  //                     SOA_COLUMN(uint16_t, BX0_energy),
  //                     SOA_COLUMN(uint16_t, BXp1_energy),
  //                     SOA_COLUMN(uint16_t, BXp2_energy),
  //                     SOA_COLUMN(uint16_t, BXp3_energy),
  //                     SOA_COLUMN(uint16_t, flags),
  //                     SOA_COLUMN(uint16_t, layer),
  //                     SOA_COLUMN(uint16_t, moduleIdx))
  // using HGCalDigiTriggerSoA = HGCalDigiTriggerSoALayout<>;

  // GENERATE_SOA_LAYOUT(HGCalDigiTriggerSoALayout,
  //                     SOA_COLUMN(uint8_t, algo),        //0:BC, 1:STC4A(4E3M), 2:STC4B(5E4M), 3:STC16
  // 		      SOA_COLUMN(bool, valid),          //valid bit
  //                     SOA_COLUMN(uint8_t, nBxs),        //nof Bxs        //"TODO: REMOVE THIS (When Configuration is ready)" [Read nBxs from run111138_board160_configuration.yaml CommonReadout-->RxChannels-->ReadoutWindow]
  //                     SOA_COLUMN(uint8_t, nTCs),        //nof TCs per Bx //"TODO: REMOVE THIS (When Configuration is ready)" [Interpret from nof elinks]
  // 		      SOA_COLUMN(uint8_t, iBx),         //ith Bx   //We need this for DQM
  // 		      SOA_COLUMN(uint32_t, econTId),    //econTId  //"TODO: REMOVE THIS (When Configuration is ready)"
  // 		      SOA_COLUMN(uint8_t, bxId),        //Bx index as read fom ECONT
  // 		      SOA_COLUMN(uint32_t, TotE),          //Module sum for BC and total enegy for STC
  // 		      SOA_COLUMN(uint32_t, TCEnergy),      //TC energies
  // 		      SOA_COLUMN(uint8_t, TCAddress)      //TC addresses
  // 		      )

  using Mtrxu32t71 = Eigen ::Matrix<uint32_t, 7, 1>;
  using Mtrxu8t71 = Eigen ::Matrix<uint8_t, 7, 1>;
  using Mtrxbt71 = Eigen ::Matrix<bool, 7, 1>;

  GENERATE_SOA_LAYOUT(
      HGCalDigiTriggerSoALayout,
      SOA_COLUMN(uint8_t, algo),          //0:BC, 1:STC4A(4E3M), 2:STC4B(5E4M), 3:STC16
      SOA_COLUMN(uint8_t, sumType),       //0: MS summed over 48-N TCs, 1: MS summed over 48 TCs
      SOA_EIGEN_COLUMN(Mtrxbt71, valid),  //valid bit
      SOA_COLUMN(
          uint8_t,
          nBxs),  //nof Bxs   //"TODO: REMOVE THIS (When Configuration is ready)" [Read nBxs from run111138_board160_configuration.yaml CommonReadout-->RxChannels-->ReadoutWindow]
      SOA_COLUMN(uint8_t,
                 nTCs),  //nof TCs per Bx //"TODO: REMOVE THIS (When Configuration is ready)" [Interpret from nof elinks]
      SOA_COLUMN(uint32_t, econTId),                  //econTId  //"TODO: REMOVE THIS (When Configuration is ready)"
      SOA_EIGEN_COLUMN(Mtrxu8t71, econtHeader),       //header read fom ECONT
      SOA_EIGEN_COLUMN(Mtrxu8t71, expEcontHeader),    //header expected fom Slink trailer bx counter
      SOA_EIGEN_COLUMN(Mtrxu32t71, TotE),             //Module sum for BC and total enegy for STC
      SOA_EIGEN_COLUMN(Mtrxu32t71, encodedTotE),      //Module sum for BC and total enegy for STC, encoded
      SOA_EIGEN_COLUMN(Mtrxu32t71, TCEnergy),         //TC energies
      SOA_EIGEN_COLUMN(Mtrxu32t71, encodedTCEnergy),  //TC energies - encoded
      SOA_EIGEN_COLUMN(Mtrxu8t71, TCAddress)          //TC addresses
  )

  using HGCalDigiTriggerSoA = HGCalDigiTriggerSoALayout<>;

}  // namespace hgcaldigi

#endif  // DataFormats_HGCalDigi_interface_HGCalDigiTriggerSoA_h
