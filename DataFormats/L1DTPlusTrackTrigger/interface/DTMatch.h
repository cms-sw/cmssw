#ifndef DTMatch_h
#define DTMatch_h

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

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatchBase.h"

/// Pt bins
float const binPt[25] = {  0.,                          /// Lower limit
                           4.,  5.,  6.,  7.,  8.,      /// 1 GeV steps
                          10., 12., 14., 16., 18., 20., /// 2 GeV steps
                          25., 30., 35., 40., 45., 50., /// 5 GeV steps
                          60., 70., 80., 90., 100.,     /// 10 GeV steps
                         120., 140. };                  /// 20 GeV steps

/// Curvature thresholds to accept 90% of the muons with Pt larger than a given value
/// Let the Pt be 10 GeV. This is entry 7 of the binPt array.
/// The seventh value in the table below, i.e. entry 6, corresponds to the curvature
/// threshold that accepts 90% of the 10 GeV muons
/// There are two tables: one for DT seeds in station MB1, one for DT seeds
/// in station MB2
float const cutPtInvMB1[15][24] = {
  {0.3212, 0.2390, 0.1932, 0.1630, 0.1410, 0.1115, 0.0923, 0.0787, 0.0686, 0.0608, 0.0546, 0.0436, 0.0363, 0.0312, 0.0273, 0.0243, 0.0220, 0.0184, 0.0159, 0.0140, 0.0126, 0.0114, 0.0097, 0.0084},
  {0.3209, 0.2395, 0.1938, 0.1636, 0.1416, 0.1119, 0.0926, 0.0791, 0.0689, 0.0611, 0.0549, 0.0438, 0.0365, 0.0313, 0.0274, 0.0245, 0.0221, 0.0185, 0.0160, 0.0141, 0.0126, 0.0115, 0.0097, 0.0085},
  {0.3231, 0.2409, 0.1951, 0.1646, 0.1423, 0.1125, 0.0930, 0.0794, 0.0692, 0.0613, 0.0551, 0.0440, 0.0366, 0.0314, 0.0276, 0.0246, 0.0222, 0.0186, 0.0161, 0.0142, 0.0127, 0.0115, 0.0098, 0.0086},
  {0.3208, 0.2403, 0.1950, 0.1646, 0.1424, 0.1126, 0.0931, 0.0795, 0.0693, 0.0614, 0.0552, 0.0441, 0.0367, 0.0315, 0.0276, 0.0246, 0.0222, 0.0186, 0.0161, 0.0142, 0.0127, 0.0116, 0.0098, 0.0086},
  {0.3230, 0.2420, 0.1960, 0.1655, 0.1432, 0.1131, 0.0936, 0.0798, 0.0696, 0.0617, 0.0554, 0.0442, 0.0369, 0.0316, 0.0277, 0.0247, 0.0223, 0.0187, 0.0162, 0.0143, 0.0128, 0.0116, 0.0099, 0.0087},
  {0.3265, 0.2439, 0.1976, 0.1665, 0.1442, 0.1139, 0.0941, 0.0803, 0.0700, 0.0620, 0.0557, 0.0445, 0.0370, 0.0318, 0.0279, 0.0249, 0.0224, 0.0188, 0.0163, 0.0144, 0.0129, 0.0117, 0.0100, 0.0087},
  {0.3212, 0.2418, 0.1962, 0.1658, 0.1436, 0.1135, 0.0939, 0.0802, 0.0698, 0.0619, 0.0556, 0.0444, 0.0370, 0.0317, 0.0278, 0.0248, 0.0224, 0.0188, 0.0163, 0.0144, 0.0129, 0.0117, 0.0100, 0.0087},
  {0.3240, 0.2433, 0.1976, 0.1667, 0.1444, 0.1141, 0.0943, 0.0805, 0.0702, 0.0622, 0.0559, 0.0446, 0.0372, 0.0319, 0.0280, 0.0249, 0.0225, 0.0189, 0.0164, 0.0144, 0.0130, 0.0118, 0.0100, 0.0088},
  {0.3268, 0.2455, 0.1990, 0.1680, 0.1455, 0.1148, 0.0950, 0.0809, 0.0706, 0.0626, 0.0562, 0.0448, 0.0374, 0.0321, 0.0281, 0.0251, 0.0226, 0.0190, 0.0165, 0.0146, 0.0131, 0.0119, 0.0101, 0.0089},
  {0.3312, 0.2480, 0.2011, 0.1698, 0.1468, 0.1158, 0.0957, 0.0817, 0.0712, 0.0631, 0.0566, 0.0452, 0.0376, 0.0323, 0.0283, 0.0253, 0.0228, 0.0192, 0.0166, 0.0147, 0.0132, 0.0120, 0.0103, 0.0090},
  {0.3222, 0.2436, 0.1981, 0.1675, 0.1449, 0.1146, 0.0948, 0.0808, 0.0705, 0.0625, 0.0561, 0.0448, 0.0373, 0.0320, 0.0281, 0.0250, 0.0226, 0.0190, 0.0164, 0.0145, 0.0131, 0.0119, 0.0101, 0.0089},
  {0.3250, 0.2453, 0.1993, 0.1684, 0.1458, 0.1153, 0.0953, 0.0813, 0.0708, 0.0628, 0.0564, 0.0450, 0.0375, 0.0322, 0.0282, 0.0252, 0.0227, 0.0191, 0.0165, 0.0146, 0.0131, 0.0120, 0.0102, 0.0089},
  {0.3277, 0.2473, 0.2010, 0.1698, 0.1469, 0.1160, 0.0960, 0.0818, 0.0713, 0.0632, 0.0568, 0.0453, 0.0378, 0.0324, 0.0284, 0.0253, 0.0229, 0.0193, 0.0167, 0.0147, 0.0133, 0.0121, 0.0103, 0.0090},
  {0.3321, 0.2502, 0.2032, 0.1714, 0.1484, 0.1171, 0.0968, 0.0826, 0.0719, 0.0637, 0.0572, 0.0457, 0.0381, 0.0327, 0.0287, 0.0256, 0.0231, 0.0194, 0.0168, 0.0149, 0.0134, 0.0122, 0.0105, 0.0092},
  {0.3374, 0.2539, 0.2060, 0.1738, 0.1503, 0.1185, 0.0979, 0.0834, 0.0727, 0.0644, 0.0579, 0.0462, 0.0385, 0.0330, 0.0290, 0.0259, 0.0234, 0.0197, 0.0171, 0.0151, 0.0136, 0.0124, 0.0106, 0.0094} };

float const cutPtInvMB2[15][24] = {
  {0.3882, 0.2744, 0.2174, 0.1815, 0.1560, 0.1223, 0.1007, 0.0857, 0.0746, 0.0661, 0.0593, 0.0474, 0.0395, 0.0340, 0.0299, 0.0268, 0.0243, 0.0205, 0.0178, 0.0158, 0.0143, 0.0130, 0.0112, 0.0099},
  {0.3894, 0.2765, 0.2193, 0.1829, 0.1574, 0.1234, 0.1015, 0.0864, 0.0752, 0.0666, 0.0598, 0.0478, 0.0399, 0.0343, 0.0302, 0.0270, 0.0245, 0.0207, 0.0180, 0.0160, 0.0144, 0.0132, 0.0114, 0.0101},
  {0.3952, 0.2798, 0.2216, 0.1848, 0.1588, 0.1245, 0.1024, 0.0871, 0.0757, 0.0671, 0.0602, 0.0481, 0.0402, 0.0346, 0.0304, 0.0273, 0.0247, 0.0209, 0.0182, 0.0162, 0.0146, 0.0134, 0.0115, 0.0102},
  {0.3922, 0.2790, 0.2217, 0.1851, 0.1593, 0.1248, 0.1027, 0.0874, 0.0761, 0.0673, 0.0605, 0.0483, 0.0403, 0.0347, 0.0306, 0.0274, 0.0248, 0.0210, 0.0183, 0.0163, 0.0147, 0.0135, 0.0116, 0.0103},
  {0.3979, 0.2823, 0.2238, 0.1867, 0.1608, 0.1260, 0.1036, 0.0880, 0.0767, 0.0679, 0.0610, 0.0487, 0.0406, 0.0350, 0.0308, 0.0276, 0.0251, 0.0212, 0.0185, 0.0164, 0.0149, 0.0136, 0.0117, 0.0104},
  {0.4051, 0.2865, 0.2269, 0.1892, 0.1627, 0.1274, 0.1048, 0.0890, 0.0775, 0.0686, 0.0616, 0.0492, 0.0411, 0.0353, 0.0311, 0.0279, 0.0253, 0.0215, 0.0187, 0.0167, 0.0151, 0.0138, 0.0119, 0.0106},
  {0.3961, 0.2829, 0.2251, 0.1880, 0.1617, 0.1267, 0.1042, 0.0887, 0.0772, 0.0684, 0.0614, 0.0490, 0.0409, 0.0352, 0.0310, 0.0278, 0.0252, 0.0214, 0.0187, 0.0166, 0.0150, 0.0138, 0.0119, 0.0105},
  {0.4013, 0.2864, 0.2276, 0.1897, 0.1633, 0.1279, 0.1053, 0.0895, 0.0778, 0.0690, 0.0619, 0.0495, 0.0413, 0.0355, 0.0313, 0.0281, 0.0255, 0.0216, 0.0189, 0.0168, 0.0152, 0.0139, 0.0121, 0.0107},
  {0.4091, 0.2911, 0.2308, 0.1923, 0.1655, 0.1294, 0.1065, 0.0905, 0.0787, 0.0697, 0.0626, 0.0500, 0.0417, 0.0359, 0.0317, 0.0284, 0.0258, 0.0219, 0.0191, 0.0171, 0.0155, 0.0142, 0.0123, 0.0109},
  {0.4185, 0.2968, 0.2349, 0.1956, 0.1682, 0.1316, 0.1081, 0.0918, 0.0798, 0.0707, 0.0635, 0.0507, 0.0423, 0.0365, 0.0322, 0.0288, 0.0262, 0.0223, 0.0195, 0.0174, 0.0158, 0.0145, 0.0125, 0.0112},
  {0.4005, 0.2876, 0.2285, 0.1909, 0.1642, 0.1287, 0.1058, 0.0901, 0.0783, 0.0694, 0.0623, 0.0498, 0.0415, 0.0358, 0.0315, 0.0283, 0.0257, 0.0218, 0.0190, 0.0170, 0.0154, 0.0141, 0.0122, 0.0109},
  {0.4064, 0.2911, 0.2312, 0.1928, 0.1659, 0.1300, 0.1069, 0.0908, 0.0791, 0.0700, 0.0629, 0.0502, 0.0419, 0.0361, 0.0319, 0.0286, 0.0259, 0.0220, 0.0193, 0.0172, 0.0156, 0.0143, 0.0124, 0.0110},
  {0.4137, 0.2957, 0.2346, 0.1955, 0.1682, 0.1317, 0.1082, 0.0920, 0.0800, 0.0708, 0.0636, 0.0508, 0.0424, 0.0365, 0.0322, 0.0289, 0.0263, 0.0223, 0.0195, 0.0175, 0.0158, 0.0145, 0.0126, 0.0113},
  {0.4238, 0.3015, 0.2388, 0.1990, 0.1712, 0.1338, 0.1099, 0.0934, 0.0812, 0.0719, 0.0645, 0.0516, 0.0431, 0.0371, 0.0328, 0.0294, 0.0267, 0.0228, 0.0199, 0.0178, 0.0162, 0.0149, 0.0130, 0.0116},
  {0.4366, 0.3095, 0.2448, 0.2038, 0.1750, 0.1367, 0.1123, 0.0954, 0.0828, 0.0733, 0.0658, 0.0526, 0.0439, 0.0379, 0.0334, 0.0300, 0.0273, 0.0233, 0.0205, 0.0183, 0.0167, 0.0153, 0.0134, 0.0119} };

/// Class implementation
class DTMatch : public DTMatchBase
{
  public:

    /// Trivial default constructor
    DTMatch();

    /// Constructors
    DTMatch( int aDTWheel, int aDTStation, int aDTSector,
             int aDTBX, int aDTCode, int aTSPhi, int aTSPhiB, int aTSTheta,
             bool aFlagBXOK );

    DTMatch( int aDTWheel, int aDTStation, int aDTSector,
             int aDTBX, int aDTCode, int aTSPhi, int aTSPhiB, int aTSTheta,
             GlobalPoint aDTPosition, GlobalVector aDTDirection,
             bool aFlagBXOK );

    DTMatch( const DTMatch& aDTM );

    /// Assignment operator
    DTMatch& operator = ( const DTMatch& aDTM );

    /// Destructor
    ~DTMatch(){}

    /// Method with the only purpose of making this class non-abstract
//    void toMakeThisAbstract(){} // NOTE: this overrides the method inherited from the parent class
                                // As the parent class needs to be an abstract class, this method
                                // is declared as purely virtual in the parent class, JUST for this
                                // purpose of making it abstract.
                                // On the other side, we want to create objects of this class, and
                                // this is the reason to make THIS class not abstract.

    /*** DT TRIGGER BASIC INFORMATION ***/
    /// Return functions for trigger order
    inline int getDTTTrigOrder() const { return theDTTrigOrder; }

    /// Set phi-eta matching order flag
    inline void setDTTrigOrder( unsigned int aDTTrigOrder )
    {
      theDTTrigOrder = aDTTrigOrder;
      return;
    }

    /// Theta flag if position comes from BTI
    inline void setThetaCorrection( float aDeltaTheta )
    {
      theThetaFlag = false; /// Default value is true in constructor
      theDeltaTheta = static_cast< int >( aDeltaTheta * 4096. / 3. );
      return;
    }

    inline bool getThetaFlag() const { return theThetaFlag; }
    inline int getDeltaTheta() const { return theDeltaTheta; }

    /// Flag if redundant and to be rejected
    inline void setRejectionFlag( bool aFlag )
    {
      theRejectionFlag = aFlag;
      return;
    }

    inline bool getRejectionFlag() const { return theRejectionFlag; }

    /*** DT TRIGGER MOMENTUM PARAMETERISATION ***/
    /// Return functions
    int getDTPt() const;
    int getDTPtMin( float nSigmas ) const;
    int getDTPtMax( float nSigmas ) const;

    /// Extrapolation of DT object to TK and Vtx
    void extrapolateToTrackerLayer( unsigned int aLayer ); /// This runs from 1 to N, as the DetId index
    void extrapolateToVertex();

    /*** PROJECTION TO STUBS ***/
    /// Return functions for projected positions at specific layers (12 bits precision)
    /// NOTE: also these run from 1 to N, as the DetId index
    inline int getPredStubPhi( unsigned int aLayer ) const
    {
      if ( aLayer > 0 && aLayer <= numberOfTriggerLayers )
      {
        return thePredPhi[ aLayer ];
      }
      return -999999;
    }

    inline int getPredStubSigmaPhi( unsigned int aLayer ) const
    {
      if ( aLayer > 0 && aLayer <= numberOfTriggerLayers )
      {
        return thePredSigmaPhi[ aLayer ];
      }
      return -999999;
    }

    inline int getPredStubTheta( unsigned int aLayer ) const
    {
      if ( aLayer > 0 && aLayer <= numberOfTriggerLayers )
      {
        return thePredTheta[ aLayer ];
      }
      return -999999;
    }

    inline int getPredStubSigmaTheta( unsigned int aLayer ) const
    {
      if ( aLayer > 0 && aLayer <= numberOfTriggerLayers )
      {
        return thePredSigmaTheta[ aLayer ];
      }
      return -999999;
    }

    /// Set predicted tracker phi and theta in each layer
    inline void setPredStubPhi( unsigned int aLayer, int aPhi, int aSigmaPhi )
    {
      if ( aLayer > 0 && aLayer <= numberOfTriggerLayers )
      {
        thePredPhi[ aLayer ] = aPhi;
        thePredSigmaPhi[ aLayer ] = aSigmaPhi;
      }
      return;
    }

    inline void setPredStubTheta( unsigned int aLayer, int aTheta, int aSigmaTheta )
    {
      if ( aLayer > 0 && aLayer <= numberOfTriggerLayers )
      {
        thePredTheta[ aLayer ] = aTheta;
        thePredSigmaTheta[ aLayer ] = aSigmaTheta;
      }
      return;
    }

    /*** PROJECTION TO VERTEX ***/
    /// Return functions at vertex
    inline int getPredVtxPhi() const { return thePredPhi[ 0 ]; }
    inline int getPredVtxSigmaPhi() const { return thePredSigmaPhi[ 0 ]; }
    inline int getPredVtxTheta() const { return thePredTheta[ 0 ]; }
    inline int getPredVtxSigmaTheta() const { return thePredSigmaTheta[ 0 ]; }

    /// Set predicted tracker phi and theta at vertex
    inline void setPredVtxPhi( int aPhi, int aSigmaPhi )
    {
      thePredPhi[ 0 ] = aPhi;
      thePredSigmaPhi[ 0 ] = aSigmaPhi;
      return;
    }

    inline void setPredVtxTheta( int aTheta, int aSigmaTheta )
    {
      thePredTheta[ 0 ] = aTheta;
      thePredSigmaTheta[ 0 ] = aSigmaTheta;
      return;
    }

    /*** BENDING ANGLE ***/
    /// Return predicted error on bending angle inside the tracker
    inline float getPredSigmaPhiB() const { return thePredSigmaPhiB; }

    /// Set predicted error on bending angle inside the tracker
    inline void setPredSigmaPhiB( float aPredSigmaPhib )
    {
      thePredSigmaPhiB = aPredSigmaPhib;
      return;
    }

    /*** CHECK THE MATCHES ***/
    /// Check the match with a stub
    bool checkStubPhiMatch( int anotherPhi, unsigned int aLayer, float nSigmas ) const;
    bool checkStubThetaMatch( int anotherTheta, unsigned int aLayer, float nSigmas ) const;
    int findStubDeltaPhi( int anotherPhi, unsigned int aLayer ) const;

    /// Check the match with a track
    bool checkVtxPhiMatch( int anotherPhi, float nSigmas ) const;
    bool checkVtxThetaMatch( int anotherTheta, float nSigmas ) const;
    int findVtxDeltaPhi( int anotherPhi ) const;

    /*** DIFFERENT PT ASSIGNMENT ***/
    /// Return functions for Pt information
    inline float getPtPriority() const { return thePtPriority; }
    inline float getPtAverage() const { return thePtAverage; }
    inline float getPtPriorityFlag() const { return thePtPriorityFlag; }
    inline float getPtAverageFlag() const { return thePtAverageFlag; }
    inline float getPtPriorityBin() const { return thePtPriorityBin; }
    inline float getPtAverageBin() const { return thePtAverageBin; }
    inline float getPtTTTrackBin() const { return thePtTTTrackBin; }
    inline float getPtMajorityFullTkBin() const { return thePtMajorityFullTkBin; }
    inline float getPtMajorityBin() const { return thePtMajorityBin; }
    inline float getPtMixedModeBin() const { return thePtMixedModeBin; }

    /// Operations with the huge tables
    void findPtPriority();
    void findPtAverage();
    void findPtPriorityBin();
    void findPtAverageBin();
    void findPtTTTrackBin();
    void findPtMajorityFullTkBin();
    void findPtMajorityBin();
    void findPtMixedModeBin();

    unsigned int findPtBin( float aPtInv, unsigned int aMethod );

    /// These are to store the floating point values
    inline void setPtPriority( float aPtInv )
    {
      thePtPriority = 1./aPtInv;
      thePtPriorityFlag = true;
      return;
    }

    inline void setPtAverage( float aPtInv )
    {
      thePtAverage = 1./aPtInv;
      thePtAverageFlag = true;
      return;
    }

    /// These are to store the Pt bins
    inline void setPtPriorityBin( float aPtBin )
    {
      thePtPriorityBin = aPtBin;
      return;
    }

    inline void setPtAverageBin( float aPtBin )
    {
      thePtAverageBin = aPtBin;
      return;
    }

    inline void setPtTTTrackBin( float aPtBin )
    {
      thePtTTTrackBin = aPtBin;
      return;
    }

    inline void setPtMajorityFullTkBin( float aPtBin )
    {
      thePtMajorityFullTkBin = aPtBin;
      return;
    }

    inline void setPtMajorityBin( float aPtBin )
    {
      thePtMajorityBin = aPtBin;
      return;
    }

    inline void setPtMixedModeBin( float aPtBin )
    {
      thePtMixedModeBin = aPtBin;
      return;
    }

    /// Debug function
    std::string print() const
    {
      std::stringstream output;
      output << "DTMatch : wh " << this->getDTWheel() << ", st " << this->getDTStation() << ", se " << this->getDTSector()
             << ", bx " << this->getDTBX() << ", code " << this->getDTCode() << " rejection " << this->getRejectionFlag()
             << std::endl;
      return output.str();
    }

  private :

    void init();

    /// Data members

    /// DT Trigger order
    unsigned int theDTTrigOrder;

    /// Flag if theta missing and BTI is used instead
    bool theThetaFlag;
    int theDeltaTheta; /// its correction

    /// Flag if redundant and to be rejected
    bool theRejectionFlag;

    /// Predicted positions
    /// NOTE: using arrays, overdimensioned to account for
    /// DetId layer index ranging from 1 to numberOfTriggerLayers
    /// Element [ 0 ] used to store prediction at vertex
    int thePredPhi[ numberOfTriggerLayers + 1 ];
    int thePredSigmaPhi[ numberOfTriggerLayers + 1 ];
    int thePredTheta[ numberOfTriggerLayers + 1 ];
    int thePredSigmaTheta[ numberOfTriggerLayers + 1 ];

    /// Predicted direction at vertex is stored in elements [0] of the previous arrays

    /// Predicted error on bending angle inside the tracker
    float thePredSigmaPhiB;

    /// Pt information
    float thePtPriority; /// The values
    float thePtAverage;
    bool thePtPriorityFlag; /// Flags to check that the values are actually set
    bool thePtAverageFlag;
    float thePtPriorityBin; /// The binned Pt
    float thePtAverageBin;
    float thePtTTTrackBin;
    float thePtMajorityFullTkBin;
    float thePtMajorityBin;
    float thePtMixedModeBin;
};

typedef std::vector< DTMatch > DTMatchesCollection;

#endif

