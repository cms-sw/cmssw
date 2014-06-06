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
    {0.016, 0.042, 0.013, 0.032, 0.014, 0.031, 0.013, 0.032, 0.016, 0.041} };
  float B_DeltaPhi_C[6][10] = { /// Layer, (2*Wheel + Station)
    {13.0, 21.8, 11.7, 18.6, 11.9, 18.2, 11.8, 18.5, 13.1, 21.8},
    {12.6, 21.2, 11.3, 18.0, 11.6, 17.7, 11.4, 18.0, 12.6, 21.1},
    {12.0, 20.3, 10.9, 17.3, 11.1, 17.0, 10.9, 17.3, 12.1, 20.3},
    {11.4, 19.4, 10.3, 16.5, 10.5, 16.2, 10.3, 16.5, 11.5, 19.3},
    {10.7, 18.3, 9.7, 15.6, 9.9, 15.4, 9.7, 15.6, 10.8, 18.2},
    {10.1, 17.4, 9.1, 14.8, 9.3, 14.5, 9.1, 14.8, 10.1, 17.3} };

  /// Single
  float A_DeltaPhi_S[6][10] = { /// Layer, (2*Wheel + Station)
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.032, 0.014, 0.032, 0.016, 0.042},
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.031, 0.014, 0.033, 0.016, 0.042},
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.032, 0.013, 0.033, 0.016, 0.042},
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.031, 0.013, 0.032, 0.016, 0.042},
    {0.016, 0.042, 0.014, 0.033, 0.014, 0.031, 0.013, 0.032, 0.016, 0.042},
    {0.016, 0.043, 0.014, 0.033, 0.014, 0.032, 0.013, 0.032, 0.016, 0.042} };
  float B_DeltaPhi_S[6][10] = { /// Layer, (2*Wheel + Station)
    {13.3, 22.1, 11.7, 18.5, 11.9, 18.1, 11.7, 18.5, 13.3, 22.0},
    {12.8, 21.4, 11.3, 18.0, 11.5, 17.6, 11.4, 17.9, 12.8, 21.3},
    {12.3, 20.6, 10.9, 17.3, 11.0, 16.9, 10.9, 17.3, 12.3, 20.5},
    {11.7, 19.6, 10.3, 16.5, 10.5, 16.2, 10.4, 16.5, 11.7, 19.6},
    {11.0, 18.5, 9.7, 15.6, 9.8, 15.3, 9.7, 15.6, 11.0, 18.5},
    {10.3, 17.6, 9.1, 14.8, 9.2, 14.4, 9.1, 14.7, 10.4, 17.5} };

  /// Correlated
  float A_DeltaTheta_C[6][10] = { /// Layer, (2*Wheel + Station)
    {0.007, 0.018, 0.004, 0.011, 0.000, 0.000, -0.004, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.004, 0.011, 0.000, 0.000, -0.004, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.004, 0.010, 0.000, -0.000, -0.004, -0.010, -0.006, -0.017},
    {0.006, 0.016, 0.004, 0.011, -0.000, 0.000, -0.004, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.004, 0.009, 0.000, 0.000, -0.004, -0.009, -0.007, -0.017},
    {0.007, 0.017, 0.004, 0.010, 0.000, -0.000, -0.004, -0.010, -0.006, -0.016} };
  float C_DeltaTheta_C[6][10] = { /// Layer, (2*Wheel + Station)
    {-3.0, 11.9, 22.0, 27.1, 2.4, 2.6, -23.6, -25.6, -1.0, -9.0},
    {6.8, 15.9, 20.5, 23.8, 0.5, 0.6, -21.6, -21.2, -9.6, -12.5},
    {9.6, 17.4, 20.1, 23.5, 0.4, 1.0, -21.2, -20.6, -12.7, -13.6},
    {12.8, 19.3, 26.0, 8.8, -27.6, -17.7, 17.0, 13.4, -27.4, -9.0},
    {20.4, 16.1, 12.9, 17.7, -5.0, 2.8, -20.1, -25.8, -1.4, -13.2},
    {24.7, 28.0, 14.7, 17.2, -3.2, -4.9, -14.6, -19.1, -37.6, -26.7} };

  /// Single
  float A_DeltaTheta_S[6][10] = { /// Layer, (2*Wheel + Station)
    {0.006, 0.017, 0.005, 0.012, -0.000, 0.000, -0.005, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.005, 0.011, 0.000, 0.000, -0.005, -0.011, -0.006, -0.017},
    {0.006, 0.017, 0.005, 0.011, 0.000, 0.000, -0.005, -0.010, -0.006, -0.017},
    {0.005, 0.015, 0.005, 0.012, -0.000, 0.000, -0.004, -0.012, -0.006, -0.016},
    {0.007, 0.019, 0.005, 0.010, 0.000, 0.000, -0.005, -0.010, -0.005, -0.016},
    {0.006, 0.017, 0.004, 0.010, 0.000, -0.000, -0.005, -0.010, -0.006, -0.015} };
  float C_DeltaTheta_S[6][10] = { /// Layer, (2*Wheel + Station)
    {43.7, 35.7, 38.0, 36.0, 2.2, 1.3, -38.5, -39.7, -49.0, -33.6},
    {48.1, 36.5, 34.4, 34.4, 0.6, 0.1, -35.0, -34.9, -52.9, -34.0},
    {47.7, 38.0, 34.4, 33.0, 1.0, -0.3, -34.7, -32.3, -52.9, -36.4},
    {51.8, 58.8, 36.1, 38.0, -27.7, -14.0, 3.8, 2.3, -35.3, -25.7},
    {55.2, 31.4, 27.4, 32.2, -8.4, 6.9, -41.0, -44.8, -55.3, -37.8},
    {61.0, 52.1, 29.8, 29.9, -6.4, -6.1, -26.4, -32.6, -80.5, -51.0} };

  /// Functions to compute extrapolation windows
  /// Correlated
  float A_SigmaDeltaPhi_C[6][10] = { /// Layer, (2*Wheel + Station)
    {0.014, 0.078, 0.008, 0.039, 0.011, 0.040, 0.009, 0.038, 0.014, 0.077},
    {0.013, 0.076, 0.008, 0.038, 0.011, 0.039, 0.008, 0.037, 0.013, 0.075},
    {0.013, 0.073, 0.008, 0.036, 0.010, 0.038, 0.008, 0.036, 0.013, 0.072},
    {0.012, 0.070, 0.007, 0.035, 0.010, 0.036, 0.007, 0.034, 0.012, 0.069},
    {0.011, 0.066, 0.007, 0.033, 0.009, 0.035, 0.007, 0.033, 0.011, 0.066},
    {0.010, 0.062, 0.007, 0.031, 0.009, 0.033, 0.007, 0.031, 0.010, 0.061} };
  float C_SigmaDeltaPhi_C[6][10] = { /// Layer, (2*Wheel + Station)
    {21.6, 39.8, 19.6, 33.3, 20.5, 32.2, 19.8, 32.9, 21.6, 39.7},
    {20.9, 38.6, 18.8, 32.2, 19.9, 31.4, 19.1, 31.9, 20.7, 38.2},
    {19.9, 37.1, 18.4, 31.2, 19.1, 30.3, 18.5, 30.9, 20.0, 36.9},
    {19.1, 35.3, 17.4, 29.9, 18.1, 29.0, 17.4, 29.5, 19.0, 35.1},
    {17.8, 33.1, 16.4, 28.5, 17.3, 27.7, 16.5, 27.9, 17.9, 32.9},
    {16.8, 31.2, 15.4, 27.1, 16.1, 26.0, 15.5, 26.7, 16.7, 31.0} };

  /// Single
  float A_SigmaDeltaPhi_S[6][10] = { /// Layer, (2*Wheel + Station)
    {0.008, 0.047, 0.006, 0.023, 0.007, 0.025, 0.005, 0.022, 0.008, 0.047},
    {0.008, 0.046, 0.005, 0.023, 0.007, 0.024, 0.005, 0.022, 0.008, 0.047},
    {0.008, 0.045, 0.005, 0.022, 0.007, 0.024, 0.005, 0.022, 0.007, 0.046},
    {0.007, 0.043, 0.005, 0.022, 0.007, 0.023, 0.004, 0.021, 0.007, 0.044},
    {0.007, 0.042, 0.005, 0.021, 0.006, 0.022, 0.004, 0.020, 0.007, 0.043},
    {0.007, 0.041, 0.005, 0.021, 0.006, 0.021, 0.004, 0.020, 0.006, 0.040} };
  float C_SigmaDeltaPhi_S[6][10] = { /// Layer, (2*Wheel + Station)
    {165.1, 327.0, 172.5, 306.0, 196.2, 317.3, 171.3, 301.0, 163.4, 324.3},
    {159.3, 315.9, 167.8, 298.5, 191.0, 309.1, 166.8, 293.7, 158.2, 312.6},
    {153.0, 304.3, 161.4, 287.7, 183.2, 299.4, 159.5, 283.6, 152.9, 300.6},
    {146.5, 291.0, 153.2, 273.6, 175.0, 285.8, 152.6, 269.6, 144.9, 290.3},
    {137.3, 275.3, 144.1, 260.4, 163.9, 269.7, 142.7, 256.6, 136.2, 272.9},
    {130.2, 259.1, 135.7, 247.8, 154.6, 257.3, 134.6, 242.4, 130.5, 255.6} };

  /// Correlated
  float A_SigmaDeltaTheta_C[6][10] = { /// Layer, (2*Wheel + Station)
    {0.0016, 0.0060, 0.0011, 0.0029, 0.0004, 0.0009, 0.0011, 0.0028, 0.0014, 0.0056},
    {0.0013, 0.0053, 0.0008, 0.0025, 0.0005, 0.0013, 0.0009, 0.0025, 0.0012, 0.0054},
    {0.0015, 0.0061, 0.0009, 0.0027, 0.0007, 0.0018, 0.0010, 0.0027, 0.0014, 0.0059},
    {0.0016, 0.0072, 0.0013, 0.0021, 0.0007, 0.0013, 0.0010, 0.0021, 0.0010, 0.0064},
    {0.0013, 0.0062, 0.0009, 0.0029, 0.0008, 0.0012, 0.0012, 0.0036, 0.0012, 0.0065},
    {0.0019, 0.0077, 0.0011, 0.0037, 0.0012, 0.0030, 0.0011, 0.0037, 0.0020, 0.0073} };
  float C_SigmaDeltaTheta_C[6][10] = { /// Layer, (2*Wheel + Station)
    {331.6, 370.2, 537.0, 588.8, 729.3, 741.6, 537.3, 590.4, 333.1, 369.8},
    {216.6, 237.8, 343.9, 377.1, 462.2, 468.5, 343.8, 376.9, 217.7, 237.6},
    {154.2, 166.9, 240.4, 264.1, 321.5, 326.1, 240.2, 263.9, 155.0, 166.6},
    {117.9, 125.4, 189.7, 225.4, 250.7, 268.4, 195.0, 220.7, 127.7, 129.3},
    {101.8, 106.1, 151.2, 158.3, 195.4, 201.9, 147.0, 149.6, 103.1, 104.1},
    {84.1, 87.7, 124.9, 133.3, 165.7, 163.9, 125.7, 133.0, 80.5, 87.6} };

  /// Single
  float A_SigmaDeltaTheta_S[6][10] = { /// Layer, (2*Wheel + Station)
    {0.0020, 0.0041, 0.0013, 0.0036, 0.0004, 0.0016, 0.0013, 0.0029, 0.0015, 0.0043},
    {0.0009, 0.0034, 0.0007, 0.0022, 0.0003, 0.0011, 0.0007, 0.0017, 0.0011, 0.0032},
    {0.0013, 0.0047, 0.0008, 0.0026, 0.0005, 0.0017, 0.0008, 0.0022, 0.0013, 0.0046},
    {0.0013, 0.0067, 0.0014, 0.0005, 0.0005, 0.0013, 0.0015, 0.0007, 0.0009, 0.0075},
    {0.0012, 0.0078, 0.0011, 0.0029, 0.0007, 0.0012, 0.0009, 0.0044, 0.0009, 0.0068},
    {0.0015, 0.0083, 0.0012, 0.0042, 0.0009, 0.0026, 0.0010, 0.0036, 0.0020, 0.0074} };
  float C_SigmaDeltaTheta_S[6][10] = { /// Layer, (2*Wheel + Station)
    {375.5, 417.5, 545.7, 600.7, 720.8, 734.0, 551.0, 605.1, 383.2, 420.0},
    {243.0, 270.2, 353.8, 389.7, 459.0, 467.7, 359.1, 392.0, 247.5, 271.2},
    {171.7, 189.8, 253.7, 275.3, 322.6, 326.6, 258.3, 276.6, 176.4, 190.9},
    {132.0, 136.6, 195.3, 237.8, 248.3, 271.4, 198.6, 232.0, 138.4, 133.4},
    {109.7, 108.6, 155.5, 167.3, 198.6, 207.9, 164.4, 156.0, 113.1, 110.5},
    {86.8, 90.7, 132.6, 135.6, 172.3, 165.0, 142.0, 137.4, 85.6, 92.1} };

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
  float B_DeltaPhi_C[10] = {13.8, 23.1, 12.4, 19.6, 12.7, 19.2, 12.5, 19.5, 13.9, 23.0};
  float A_DeltaPhi_S[10] = {0.016, 0.042, 0.014, 0.033, 0.014, 0.032, 0.013, 0.032, 0.016, 0.042};
  float B_DeltaPhi_S[10] = {14.0, 23.3, 12.4, 19.6, 12.6, 19.1, 12.5, 19.5, 14.1, 23.2};
  float A_DeltaTheta_C[10] = {0.006, 0.016, 0.004, 0.009, -0.000, -0.000, -0.004, -0.009, -0.006, -0.016};
  float C_DeltaTheta_C[10] = {21.8, 23.3, 20.9, 24.4, 0.7, 0.4, -22.0, -22.1, -24.9, -19.4};
  float A_DeltaTheta_S[10] = {0.006, 0.016, 0.005, 0.010, -0.000, -0.000, -0.005, -0.010, -0.006, -0.016};
  float C_DeltaTheta_S[10] = {51.9, 44.3, 40.0, 35.8, 1.8, 0.1, -39.9, -34.3, -56.2, -41.0};

  /// Extrapolation windows
  float A_SigmaDeltaPhi_C[10] = {0.015, 0.082, 0.009, 0.040, 0.012, 0.042, 0.009, 0.040, 0.015, 0.081};
  float C_SigmaDeltaPhi_C[10] = {23.0, 41.5, 20.9, 35.5, 21.9, 34.0, 21.1, 34.9, 23.1, 41.3};
  float A_SigmaDeltaPhi_S[10] = {0.008, 0.049, 0.006, 0.024, 0.007, 0.025, 0.005, 0.023, 0.008, 0.049};
  float C_SigmaDeltaPhi_S[10] = {173.6, 341.1, 181.9, 321.9, 206.6, 334.1, 180.5, 315.0, 173.5, 338.8};
  float A_SigmaDeltaTheta_C[10] = {0.0021, 0.0084, 0.0014, 0.0050, 0.0014, 0.0041, 0.0014, 0.0050, 0.0021, 0.0083};
  float C_SigmaDeltaTheta_C[10] = {55.5, 50.3, 81.0, 72.4, 105.4, 89.1, 81.3, 72.1, 56.7, 49.8};
  float A_SigmaDeltaTheta_S[10] = {0.0020, 0.0082, 0.0012, 0.0045, 0.0011, 0.0040, 0.0012, 0.0046, 0.0020, 0.0083};
  float C_SigmaDeltaTheta_S[10] = {54.0, 53.1, 97.4, 82.7, 112.8, 92.1, 98.8, 82.7, 55.2, 52.6};

  /// Error on PhiB vs. PhiB
  float A_SigmaPhiB_C[10] = {0.00070, 0.00218, 0.00050, 0.00131, 0.00059, 0.00146, 0.00050, 0.00129, 0.00069, 0.00219};
  float B_SigmaPhiB_C[10] = {0.044, 0.068, 0.032, 0.043, 0.033, 0.040, 0.032, 0.044, 0.044, 0.067};
  float C_SigmaPhiB_C[10] = {1.3, 1.5, 1.4, 1.5, 1.4, 1.5, 1.4, 1.5, 1.3, 1.5};
  float A_SigmaPhiB_S[10] = {0.00048, 0.00147, 0.00028, 0.00074, 0.00032, 0.00079, 0.00029, 0.00073, 0.00047, 0.00149};
  float B_SigmaPhiB_S[10] = {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
  float C_SigmaPhiB_S[10] = {12.0, 14.2, 14.3, 15.7, 16.0, 16.6, 13.9, 15.5, 11.9, 14.1};

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

