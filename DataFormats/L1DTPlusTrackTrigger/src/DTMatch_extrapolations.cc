/*! \class DTMatch
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Pierluigi Zotto
 *  \author Nicola Pozzobon
 *  \brief DT local triggers matched together.
 *         Objects of this class do correspond to DT muons that are then extrapolated
 *         to the stubs on the tracker layers, including a virtual one just enclosing
 *         the magnetic field volume. The main methods do aim at getting a tracker
 *         precision Pt.
 *         The matching stubs, mapped by tracker layer id, and tracker tracks are
 *         set as data members of the virtual base class DTMatchBase.
 *         Several objects of class DTMatchPt are built by methods defined in the
 *         base class DTMatchBasePtMethods.
 *  \date 2009, Feb 2
 */

#include <math.h>

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatch.h"

/// Extrapolation to a given Tk layer
/// parameter is the DetId-like index of the layer (form 1 to N)
void DTMatch::extrapolateToTrackerLayer( unsigned int aLayer )
{
  if ( aLayer > numberOfTriggerLayers ||
       aLayer == 0 )
  {
    return;
  }

  unsigned int iTk = aLayer - 1;

  /// Compute DT predicted phi and theta on tracker layers (each wheel) from Ext
  /// NOTE: needs to be extrapolated because of bending...
  /// Ext = A*|Bending|*Bending + B*Bending = Bti - Predicted, where Bti = Trigger + sector
  /// Predicted = Trigger - (A*|Bending|*Bending + B*Bending) + sector
  /// The chosen function is an antisymmetric 2nd degree polynomial
  /// For theta, it is a symmetric 2nd degree polynomial

  /// Correlated
  float A_DeltaPhi_C[6][10] = { /// Layer, (2*Wheel + Station)
    {0.015, 0.041, 0.013, 0.032, 0.014, 0.031, 0.013, 0.032, 0.015, 0.041},
    {0.015, 0.041, 0.013, 0.032, 0.014, 0.031, 0.013, 0.032, 0.015, 0.041},
    {0.015, 0.041, 0.013, 0.032, 0.014, 0.031, 0.013, 0.032, 0.015, 0.041},
    {0.015, 0.041, 0.013, 0.032, 0.014, 0.031, 0.013, 0.032, 0.015, 0.041},
    {0.015, 0.041, 0.013, 0.032, 0.014, 0.031, 0.013, 0.032, 0.015, 0.041},
    {0.016, 0.042, 0.013, 0.032, 0.014, 0.031, 0.013, 0.032, 0.016, 0.042} };
  float B_DeltaPhi_C[6][10] = { /// Layer, (2*Wheel + Station)
    {13.0, 21.8, 11.7, 18.6, 12.0, 18.2, 11.8, 18.5, 13.1, 21.8},
    {12.6, 21.1, 11.4, 18.0, 11.6, 17.6, 11.4, 18.0, 12.6, 21.1},
    {12.0, 20.3, 10.9, 17.3, 11.1, 17.0, 10.9, 17.3, 12.1, 20.3},
    {11.4, 19.4, 10.3, 16.5, 10.5, 16.2, 10.4, 16.5, 11.5, 19.3},
    {10.7, 18.3, 9.7, 15.6, 9.9, 15.4, 9.7, 15.6, 10.8, 18.2},
    {10.1, 17.3, 9.1, 14.8, 9.3, 14.5, 9.1, 14.8, 10.1, 17.3} };
 
  /// Single
  float A_DeltaPhi_S[6][10] = { /// Layer, (2*Wheel + Station)
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.031, 0.013, 0.033, 0.016, 0.042},
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.032, 0.014, 0.032, 0.016, 0.042},
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.032, 0.013, 0.033, 0.016, 0.042},
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.031, 0.013, 0.033, 0.016, 0.042},
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.032, 0.014, 0.032, 0.016, 0.042},
    {0.016, 0.043, 0.014, 0.032, 0.014, 0.031, 0.013, 0.032, 0.016, 0.042} };
  float B_DeltaPhi_S[6][10] = { /// Layer, (2*Wheel + Station)
    {13.3, 22.1, 11.8, 18.5, 11.9, 18.1, 11.8, 18.5, 13.3, 22.0},
    {12.8, 21.4, 11.4, 18.0, 11.5, 17.6, 11.4, 18.0, 12.9, 21.3},
    {12.3, 20.6, 10.9, 17.3, 11.0, 16.9, 10.9, 17.3, 12.3, 20.5},
    {11.7, 19.6, 10.4, 16.5, 10.5, 16.2, 10.4, 16.5, 11.7, 19.6},
    {11.0, 18.5, 9.7, 15.6, 9.8, 15.3, 9.7, 15.6, 11.0, 18.5},
    {10.3, 17.6, 9.1, 14.8, 9.3, 14.5, 9.1, 14.8, 10.4, 17.5} };

  /// Correlated
  float A_DeltaTheta_C[6][10] = { /// Layer, (2*Wheel + Station)
    {0.007, 0.017, 0.004, 0.011, -0.000, -0.000, -0.004, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.004, 0.011, 0.000, 0.000, -0.004, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.004, 0.010, -0.000, -0.000, -0.004, -0.010, -0.006, -0.017},
    {0.005, 0.016, 0.004, 0.011, -0.000, 0.000, -0.004, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.004, 0.009, 0.000, 0.000, -0.004, -0.009, -0.007, -0.017},
    {0.007, 0.017, 0.004, 0.010, -0.000, -0.000, -0.004, -0.009, -0.006, -0.015} };
  float C_DeltaTheta_C[6][10] = { /// Layer, (2*Wheel + Station)
    {-0.8, 12.6, 23.1, 27.3, 3.2, 2.1, -23.8, -26.5, -1.6, -10.1},
    {8.5, 16.6, 21.6, 24.1, 1.1, 0.2, -21.8, -22.1, -10.1, -13.5},
    {11.3, 18.0, 21.1, 23.7, 0.9, 0.7, -21.3, -21.5, -13.2, -14.6},
    {14.4, 20.0, 26.6, 9.6, -27.5, -17.9, 16.8, 12.2, -28.1, -9.6},
    {22.2, 16.7, 14.0, 18.0, -5.1, 2.4, -20.3, -26.6, -1.5, -13.8},
    {26.8, 28.6, 15.7, 17.5, -2.9, -5.1, -14.7, -19.9, -38.2, -27.4} };

  /// Single
  float A_DeltaTheta_S[6][10] = { /// Layer, (2*Wheel + Station)
    {0.006, 0.018, 0.005, 0.011, -0.000, -0.000, -0.004, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.005, 0.011, 0.000, -0.000, -0.004, -0.010, -0.006, -0.017},
    {0.006, 0.017, 0.005, 0.010, 0.000, 0.000, -0.004, -0.010, -0.006, -0.017},
    {0.005, 0.015, 0.005, 0.011, -0.000, 0.000, -0.004, -0.012, -0.006, -0.016},
    {0.007, 0.019, 0.005, 0.010, 0.000, 0.000, -0.005, -0.010, -0.005, -0.016},
    {0.006, 0.017, 0.004, 0.010, 0.000, -0.000, -0.004, -0.010, -0.006, -0.015} };
  float C_DeltaTheta_S[6][10] = { /// Layer, (2*Wheel + Station)
    {44.7, 32.5, 40.6, 40.0, 3.3, -1.0, -38.9, -43.3, -49.5, -33.0},
    {48.5, 34.2, 37.5, 37.0, 1.2, -0.8, -34.7, -38.0, -51.1, -33.4},
    {47.9, 36.1, 37.1, 35.0, 1.7, -1.7, -34.6, -34.8, -50.9, -36.0},
    {52.1, 56.9, 38.6, 39.4, -27.8, -15.4, 3.9, -0.2, -33.2, -25.4},
    {54.7, 30.0, 30.0, 33.4, -8.3, 5.3, -40.3, -46.0, -53.1, -37.3},
    {59.4, 49.6, 32.1, 30.9, -6.0, -6.1, -26.6, -34.4, -75.7, -50.8} };

  /// Functions to compute extrapolation windows
  float A_SigmaDeltaPhi_C[6][10] = { /// Layer, (2*Wheel + Station)
    {0.014, 0.078, 0.008, 0.038, 0.011, 0.040, 0.009, 0.038, 0.014, 0.077},
    {0.013, 0.075, 0.008, 0.038, 0.011, 0.039, 0.008, 0.037, 0.013, 0.075},
    {0.013, 0.072, 0.008, 0.036, 0.010, 0.038, 0.008, 0.036, 0.013, 0.072},
    {0.012, 0.069, 0.007, 0.035, 0.010, 0.036, 0.007, 0.035, 0.012, 0.069},
    {0.011, 0.066, 0.007, 0.033, 0.009, 0.035, 0.007, 0.033, 0.011, 0.065},
    {0.010, 0.061, 0.007, 0.031, 0.009, 0.033, 0.007, 0.031, 0.010, 0.061} };
  float C_SigmaDeltaPhi_C[6][10] = { /// Layer, (2*Wheel + Station)
    {21.5, 39.7, 19.4, 33.1, 20.4, 32.2, 19.6, 32.8, 21.2, 39.6},
    {20.9, 38.2, 18.7, 32.0, 19.7, 31.2, 18.9, 31.8, 20.6, 38.1},
    {19.9, 36.9, 18.2, 31.0, 19.0, 30.3, 18.2, 30.8, 19.8, 36.8},
    {19.1, 35.1, 17.3, 29.8, 18.0, 29.0, 17.2, 29.4, 18.8, 34.9},
    {17.8, 33.0, 16.3, 28.4, 17.1, 27.4, 16.3, 27.9, 17.5, 32.7},
    {16.7, 31.0, 15.3, 26.9, 16.0, 25.9, 15.3, 26.6, 16.6, 30.9} };

  float A_SigmaDeltaPhi_S[6][10] = { /// Layer, (2*Wheel + Station)
    {0.008, 0.048, 0.006, 0.025, 0.008, 0.026, 0.006, 0.024, 0.008, 0.047},
    {0.008, 0.047, 0.006, 0.025, 0.008, 0.025, 0.006, 0.023, 0.008, 0.046},
    {0.007, 0.046, 0.005, 0.024, 0.008, 0.025, 0.006, 0.023, 0.008, 0.045},
    {0.008, 0.044, 0.005, 0.023, 0.008, 0.024, 0.006, 0.022, 0.007, 0.045},
    {0.007, 0.043, 0.005, 0.022, 0.008, 0.023, 0.005, 0.022, 0.007, 0.042},
    {0.007, 0.041, 0.005, 0.022, 0.007, 0.022, 0.005, 0.021, 0.007, 0.040} };
  float C_SigmaDeltaPhi_S[6][10] = { /// Layer, (2*Wheel + Station)
    {162.7, 288.3, 159.5, 260.5, 168.9, 283.5, 137.6, 263.3, 146.3, 293.4},
    {156.5, 280.3, 156.3, 253.7, 164.2, 275.9, 133.4, 257.3, 140.5, 283.6},
    {152.2, 269.6, 149.1, 243.0, 157.2, 265.4, 127.2, 247.2, 135.2, 272.6},
    {143.8, 257.6, 141.3, 233.2, 150.4, 254.3, 121.3, 235.4, 129.2, 260.3},
    {136.2, 243.3, 132.6, 221.5, 141.5, 240.2, 113.8, 223.2, 121.3, 247.8},
    {127.5, 229.0, 124.9, 209.5, 134.2, 230.0, 107.5, 213.1, 114.4, 230.7} };


  float A_SigmaDeltaTheta_C[6][10] = { /// Layer, (2*Wheel + Station)
    {0.0098, 0.0412, 0.0037, 0.0185, 0.0004, 0.0023, 0.0036, 0.0195, 0.0090, 0.0401},
    {0.0078, 0.0409, 0.0040, 0.0176, 0.0006, 0.0026, 0.0042, 0.0180, 0.0079, 0.0419},
    {0.0098, 0.0432, 0.0040, 0.0173, 0.0008, 0.0027, 0.0042, 0.0177, 0.0079, 0.0434},
    {0.0099, 0.0442, 0.0063, 0.0184, 0.0003, 0.0063, 0.0059, 0.0223, 0.0087, 0.0455},
    {0.0096, 0.0469, 0.0061, 0.0191, 0.0005, 0.0032, 0.0053, 0.0178, 0.0089, 0.0519},
    {0.0124, 0.0451, 0.0054, 0.0175, 0.0009, 0.0032, 0.0056, 0.0211, 0.0100, 0.0434} };

  float C_SigmaDeltaTheta_C[6][10] = { /// Layer, (2*Wheel + Station)
    {319.0, 351.6, 524.7, 571.8, 730.7, 739.3, 526.0, 569.6, 322.1, 355.0},
    {212.0, 226.9, 331.5, 363.0, 461.0, 464.1, 332.5, 360.9, 212.5, 227.3},
    {144.1, 153.9, 231.6, 249.4, 318.1, 322.6, 232.3, 247.7, 148.1, 155.2},
    {118.5, 122.4, 176.2, 209.0, 211.6, 233.2, 178.0, 186.8, 123.1, 124.3},
    {93.5, 91.5, 140.1, 142.3, 203.2, 198.2, 134.8, 148.0, 97.0, 91.5},
    {73.4, 80.8, 118.9, 124.7, 170.2, 165.9, 118.3, 117.8, 79.2, 84.4} };

  float A_SigmaDeltaTheta_S[6][10] = { /// Layer, (2*Wheel + Station)
    {0.0109, 0.0345, 0.0059, 0.0186, 0.0011, 0.0016, 0.0054, 0.0205, 0.0047, 0.0360},
    {0.0118, 0.0416, 0.0065, 0.0201, 0.0009, 0.0021, 0.0047, 0.0187, 0.0067, 0.0447},
    {0.0098, 0.0430, 0.0075, 0.0207, 0.0010, 0.0029, 0.0047, 0.0203, 0.0089, 0.0576},
    {0.0098, 0.0351, 0.0100, 0.0259, 0.0001, 0.0027, 0.0090, 0.0257, 0.0093, 0.0506},
    {0.0126, 0.0564, 0.0084, 0.0222, 0.0001, 0.0023, 0.0055, 0.0211, 0.0102, 0.0497},
    {0.0136, 0.0675, 0.0072, 0.0251, 0.0009, 0.0033, 0.0086, 0.0237, 0.0125, 0.0441} };

  float C_SigmaDeltaTheta_S[6][10] = { /// Layer, (2*Wheel + Station)
    {349.9, 381.4, 527.4, 581.6, 712.4, 725.4, 551.5, 585.6, 350.7, 392.2},
    {226.9, 250.8, 343.2, 373.4, 458.1, 460.3, 358.4, 375.3, 230.5, 258.0},
    {157.4, 169.9, 236.1, 255.5, 320.0, 315.9, 247.0, 260.3, 157.0, 174.2},
    {133.2, 146.5, 181.1, 226.3, 220.3, 237.4, 196.8, 205.4, 132.2, 149.3},
    {103.9, 108.0, 142.1, 162.0, 202.2, 191.0, 156.4, 161.3, 93.4, 116.1},
    {77.8, 90.2, 119.5, 128.7, 168.7, 167.2, 119.5, 134.1, 78.2, 85.1} };

  int bitShift = 8;

  /// Compute predicted positions and search windows sizes
  /// NOTE: phib > 0 negative muons; phib < 0 positive muons
  ///       code >= 16 High quality ; code < 16 low quality

  int iSt = this->getDTStation() - 1;
  int iWh = this->getDTWheel() + 2;

  int thisPhiB = this->getDTTSPhiB();

  /// Calculate deviation
  int extPhi = 0;
  int deltaTheta = 0;

  if ( this->getDTCode() >= 16 ) /// High quality muons
  {
    extPhi = static_cast< int >( A_DeltaPhi_C[iTk][2*iWh+iSt] * bitShift * fabs(static_cast< float >(thisPhiB)) * static_cast< float >(thisPhiB) )
           + static_cast< int >( B_DeltaPhi_C[iTk][2*iWh+iSt] * bitShift * static_cast< float >(thisPhiB) );

    deltaTheta = static_cast< int >( A_DeltaTheta_C[iTk][2*iWh+iSt] * bitShift * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
               + static_cast< int >( C_DeltaTheta_C[iTk][2*iWh+iSt] * bitShift );
  }
  else /// Low quality muons
  {
    extPhi = static_cast< int >( A_DeltaPhi_S[iTk][2*iWh+iSt] * bitShift * fabs(static_cast< float >(thisPhiB)) * static_cast< float >(thisPhiB) )
           + static_cast< int >( B_DeltaPhi_S[iTk][2*iWh+iSt] * bitShift * static_cast< float >(thisPhiB) );

    deltaTheta = static_cast< int >( A_DeltaTheta_S[iTk][2*iWh+iSt] * bitShift * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
               + static_cast< int >( C_DeltaTheta_S[iTk][2*iWh+iSt] * bitShift );
  }

  /// The projected angles
  int predPhi = this->getDTTSPhi() - extPhi/bitShift
              + static_cast< int >( (this->getDTSector() - 1) * M_PI/6. * 4096. ); /// This brings local phi (getDTTSPhi) to global phi

  int predTheta = this->getDTTSTheta() - deltaTheta/bitShift;

  if ( predPhi < 0 )
  {
    predPhi += static_cast< int >( 2. * M_PI * 4096. );
  }

  /// Calculate errors
  int windowPhi = 0;
  int windowTheta = 0;

  if ( this->getDTCode() >= 16 ) /// High quality muons
  {
    windowPhi = static_cast< int >( A_SigmaDeltaPhi_C[iTk][2*iWh+iSt] * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
              + static_cast< int >( C_SigmaDeltaPhi_C[iTk][2*iWh+iSt] );

    windowTheta = static_cast< int >( A_SigmaDeltaTheta_C[iTk][2*iWh+iSt] * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
                + static_cast< int >( C_SigmaDeltaTheta_C[iTk][2*iWh+iSt] );
  }
  else /// Low quality muons
  {
    windowPhi = static_cast< int >( A_SigmaDeltaPhi_S[iTk][2*iWh+iSt] * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
              + static_cast< int >( C_SigmaDeltaPhi_S[iTk][2*iWh+iSt] );

    windowTheta = static_cast< int >( A_SigmaDeltaTheta_S[iTk][2*iWh+iSt] * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
                + static_cast< int >( C_SigmaDeltaTheta_S[iTk][2*iWh+iSt] );
  }

  this->setPredStubPhi( aLayer, predPhi, windowPhi );
  this->setPredStubTheta( aLayer, predTheta, windowTheta );

  return;
}

/// Extrapolation to the vertex
void DTMatch::extrapolateToVertex()
{
  /// Compute DT predicted phi and theta on tracker layers (each wheel) from Ext
  /// NOTE: needs to be extrapolated because of bending...
  /// Ext = A*|Bending|*Bending + B*Bending = Bti - Predicted, where Bti = Trigger + sector
  /// Predicted = Trigger - (A*|Bending|*Bending + B*Bending) + sector
  /// The chosen function is an antisymmetric 2nd degree polynomial
  /// For theta, it is a symmetric 2nd degree polynomial

  /// By 2*Wheel+Station
  float A_DeltaPhi_C[10] = {0.015, 0.041, 0.013, 0.032, 0.014, 0.031, 0.013, 0.032, 0.015, 0.041};
  float B_DeltaPhi_C[10] = {13.8, 23.1, 12.5, 19.6, 12.7, 19.2, 12.5, 19.6, 13.9, 23.0};
  float A_DeltaPhi_S[10] = {0.016, 0.042, 0.014, 0.033, 0.014, 0.032, 0.014, 0.032, 0.016, 0.042};
  float B_DeltaPhi_S[10] = {14.1, 23.3, 12.5, 19.6, 12.6, 19.1, 12.5, 19.6, 14.1, 23.2};
  float A_DeltaTheta_C[10] = {0.006, 0.015, 0.004, 0.009, -0.000, -0.000, -0.004, -0.009, -0.006, -0.015};
  float C_DeltaTheta_C[10] = {23.3, 24.0, 22.0, 24.5, 0.8, 0.1, -21.9, -23.0, -25.1, -20.3};
  float A_DeltaTheta_S[10] = {0.006, 0.016, 0.005, 0.010, -0.000, -0.000, -0.004, -0.009, -0.006, -0.016};
  float C_DeltaTheta_S[10] = {51.3, 42.7, 42.9, 36.7, 1.5, -0.2, -38.9, -36.6, -52.4, -40.8};

  /// Extrapolation windows
  float A_SigmaDeltaPhi_C[10] = {0.015, 0.081, 0.009, 0.040, 0.012, 0.042, 0.009, 0.040, 0.015, 0.081};
  float C_SigmaDeltaPhi_C[10] = {23.0, 41.1, 20.7, 35.3, 21.8, 34.0, 20.9, 34.9, 22.7, 41.1};
  float A_SigmaDeltaPhi_S[10] = {0.008, 0.050, 0.006, 0.025, 0.009, 0.026, 0.006, 0.024, 0.009, 0.049};
  float C_SigmaDeltaPhi_S[10] = {172.0, 303.3, 168.2, 274.3, 177.8, 296.9, 144.8, 275.0, 153.2, 306.0};
  float A_SigmaDeltaTheta_C[10] = {0.0098, 0.0437, 0.0052, 0.0185, 0.0017, 0.0049, 0.0054, 0.0187, 0.0098, 0.0432};
  float C_SigmaDeltaTheta_C[10] = {55.4, 46.2, 79.8, 71.3, 103.4, 87.9, 79.5, 71.0, 55.2, 46.4};
  float A_SigmaDeltaTheta_S[10] = {0.0113, 0.0546, 0.0078, 0.0228, 0.0015, 0.0048, 0.0061, 0.0224, 0.0103, 0.0523};
  float C_SigmaDeltaTheta_S[10] = {55.0, 58.0, 92.2, 83.4, 110.6, 95.3, 93.5, 82.8, 53.8, 55.4};

  /// Error on PhiB vs. PhiB
  float A_SigmaPhiB_C[10] = {0.00069, 0.00219, 0.00050, 0.00132, 0.00058, 0.00146, 0.00050, 0.00130, 0.00068, 0.00219};
  float B_SigmaPhiB_C[10] = {0.046, 0.069, 0.034, 0.044, 0.035, 0.041, 0.034, 0.045, 0.046, 0.068};
  float C_SigmaPhiB_C[10] = {1.3, 1.5, 1.4, 1.5, 1.4, 1.5, 1.4, 1.5, 1.3, 1.5};
  float A_SigmaPhiB_S[10] = {0.00047, 0.00163, 0.00030, 0.00092, 0.00042, 0.00091, 0.00060, 0.00082, 0.00055, 0.00158};
  float B_SigmaPhiB_S[10] = {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
  float C_SigmaPhiB_S[10] = {12.2, 12.7, 13.6, 13.5, 13.9, 14.7, 10.1, 13.6, 10.7, 12.9};

  /// Integer calculations: shift by 3 bits to preserve resolution
  int bitShift = 8;

  /// Calculate the error on PhiB
  int iWh = this->getDTWheel() + 2;
  int iSt = this->getDTStation() - 1;
  int thisPhiB = this->getDTTSPhiB();
  float predSigmaPhiB = 0.; 

  /// Parameterisation with a 2nd degree polynomial
  if ( this->getDTCode() >= 16 )
  {
    predSigmaPhiB = A_SigmaPhiB_C[iSt+2*iWh] * thisPhiB * thisPhiB
                  + B_SigmaPhiB_C[iSt+2*iWh] * abs(thisPhiB)
                  + C_SigmaPhiB_C[iSt+2*iWh];
  }
  else
  {
    predSigmaPhiB = A_SigmaPhiB_S[iSt+2*iWh] * thisPhiB * thisPhiB 
                  + B_SigmaPhiB_S[iSt+2*iWh] * abs(thisPhiB)
                  + C_SigmaPhiB_S[iSt+2*iWh];
  }

  this->setPredSigmaPhiB( predSigmaPhiB );

  /// Calculate deviation
  int extPhi = 0;
  int deltaTheta = 0;

  if ( this->getDTCode() >= 16 ) /// High quality muons
  {
    extPhi = static_cast< int >( A_DeltaPhi_C[2*iWh+iSt] * bitShift * fabs(static_cast< float >(thisPhiB)) * static_cast< float >(thisPhiB) )
           + static_cast< int >( B_DeltaPhi_C[2*iWh+iSt] * bitShift * static_cast< float >(thisPhiB) );

    deltaTheta = static_cast< int >( A_DeltaTheta_C[2*iWh+iSt] * bitShift * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
               + static_cast< int >( C_DeltaTheta_C[2*iWh+iSt] * bitShift );
  }
  else /// Low quality muons
  {
    extPhi = static_cast< int >( A_DeltaPhi_S[2*iWh+iSt] * bitShift * fabs(static_cast< float >(thisPhiB)) * static_cast< float >(thisPhiB) )
           + static_cast< int >( B_DeltaPhi_S[2*iWh+iSt] * bitShift * static_cast< float >(thisPhiB) );

    deltaTheta = static_cast< int >( A_DeltaTheta_S[2*iWh+iSt] * bitShift * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
               + static_cast< int >( C_DeltaTheta_S[2*iWh+iSt] * bitShift );
  }

  /// The projected angles
  int predPhi = this->getDTTSPhi() - extPhi/bitShift
              + static_cast< int >( (this->getDTSector() - 1) * M_PI/6. * 4096. ); /// This brings local phi (getDTTSPhi) to global phi

  int predTheta = this->getDTTSTheta() - deltaTheta/bitShift;

  if ( predPhi < 0 )
  {
    predPhi += static_cast< int >( 2. * M_PI * 4096. );
  }

  int windowPhi = 0;
  int windowTheta = 0;

  if ( this->getDTCode() >= 16 ) /// High quality muons
  {
    windowPhi = static_cast< int >( A_SigmaDeltaPhi_C[2*iWh+iSt] * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
              + static_cast< int >( C_SigmaDeltaPhi_C[2*iWh+iSt] );

    windowTheta = static_cast< int >( A_SigmaDeltaTheta_C[2*iWh+iSt] * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
                + static_cast< int >( C_SigmaDeltaTheta_C[2*iWh+iSt] );
  }
  else /// Low quality muons
  {
    windowPhi = static_cast< int >( A_SigmaDeltaPhi_S[2*iWh+iSt] * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
              + static_cast< int >( C_SigmaDeltaPhi_S[2*iWh+iSt] );

    windowTheta = static_cast< int >( A_SigmaDeltaTheta_S[2*iWh+iSt] * static_cast< float >(thisPhiB) * static_cast< float >(thisPhiB) )
                + static_cast< int >( C_SigmaDeltaTheta_S[2*iWh+iSt] );
  }

  this->setPredVtxPhi( predPhi, windowPhi );
  this->setPredVtxTheta( predTheta, windowTheta );

  return;
}

