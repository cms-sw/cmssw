#ifndef FIRMWARE_LINPUPPI_BITS_H
#define FIRMWARE_LINPUPPI_BITS_H

#define LINPUPPI_ptLSB 0.25
#define LINPUPPI_DR2LSB 1.9e-5
#define LINPUPPI_dzLSB 0.05
#define LINPUPPI_pt2LSB LINPUPPI_ptLSB* LINPUPPI_ptLSB
#define LINPUPPI_pt2DR2_scale LINPUPPI_ptLSB* LINPUPPI_ptLSB / LINPUPPI_DR2LSB

#define LINPUPPI_sum_bitShift 15
#define LINPUPPI_x2_bits 6          // decimal bits the discriminator values
#define LINPUPPI_alpha_bits 5       // decimal bits of the alpha values
#define LINPUPPI_alphaSlope_bits 5  // decimal bits of the alphaSlope values
#define LINPUPPI_ptSlope_bits 6     // decimal bits of the ptSlope values
#define LINPUPPI_weight_bits 8

#endif
