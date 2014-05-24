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
          {0.3752,0.2424,0.1967,0.1660,0.1436,0.1136,0.0941,0.0802,0.0699,0.0621,0.0559,0.0446,0.0372,0.0319,0.0280,0.0250,0.0226,0.0189,0.0164,0.0145,0.0130,0.0118,0.0101,0.0088},
          {0.3853,0.2429,0.1974,0.1669,0.1443,0.1140,0.0945,0.0806,0.0702,0.0624,0.0561,0.0448,0.0373,0.0320,0.0280,0.0250,0.0226,0.0189,0.0164,0.0145,0.0129,0.0118,0.0100,0.0088},
          {0.4082,0.2436,0.1977,0.1672,0.1445,0.1142,0.0947,0.0807,0.0704,0.0625,0.0562,0.0449,0.0374,0.0320,0.0281,0.0251,0.0226,0.0190,0.0164,0.0145,0.0130,0.0118,0.0100,0.0088},
          {0.4074,0.2433,0.1976,0.1671,0.1444,0.1142,0.0946,0.0807,0.0703,0.0625,0.0562,0.0448,0.0373,0.0320,0.0281,0.0251,0.0226,0.0189,0.0164,0.0145,0.0129,0.0118,0.0100,0.0088},
          {0.4016,0.2444,0.1982,0.1674,0.1447,0.1144,0.0948,0.0808,0.0705,0.0625,0.0563,0.0449,0.0374,0.0321,0.0281,0.0251,0.0227,0.0190,0.0164,0.0145,0.0130,0.0118,0.0101,0.0088},
          {0.4523,0.2460,0.1996,0.1686,0.1457,0.1152,0.0954,0.0813,0.0709,0.0630,0.0567,0.0452,0.0377,0.0323,0.0283,0.0254,0.0229,0.0192,0.0167,0.0148,0.0132,0.0120,0.0103,0.0090},
          {0.4743,0.2464,0.2006,0.1699,0.1470,0.1162,0.0963,0.0821,0.0716,0.0636,0.0572,0.0457,0.0380,0.0326,0.0286,0.0256,0.0231,0.0193,0.0168,0.0148,0.0133,0.0121,0.0103,0.0091},
          {0.4410,0.2475,0.2010,0.1703,0.1473,0.1165,0.0966,0.0823,0.0717,0.0637,0.0573,0.0458,0.0381,0.0327,0.0286,0.0256,0.0231,0.0194,0.0168,0.0149,0.0133,0.0121,0.0104,0.0091},
          {0.7146,0.2499,0.2028,0.1717,0.1483,0.1173,0.0972,0.0828,0.0722,0.0642,0.0577,0.0461,0.0383,0.0329,0.0288,0.0258,0.0233,0.0195,0.0169,0.0150,0.0134,0.0122,0.0105,0.0092},
          {0.5429,0.2502,0.2032,0.1720,0.1485,0.1176,0.0974,0.0830,0.0724,0.0643,0.0578,0.0461,0.0384,0.0329,0.0289,0.0259,0.0234,0.0195,0.0170,0.0150,0.0135,0.0123,0.0105,0.0092},
          {0.5257,0.2464,0.2007,0.1705,0.1473,0.1166,0.0966,0.0823,0.0717,0.0638,0.0574,0.0457,0.0381,0.0327,0.0286,0.0256,0.0231,0.0194,0.0168,0.0149,0.0133,0.0121,0.0104,0.0091},
          {0.8706,0.2471,0.2014,0.1707,0.1476,0.1168,0.0968,0.0825,0.0719,0.0639,0.0574,0.0459,0.0382,0.0327,0.0287,0.0257,0.0232,0.0194,0.0169,0.0149,0.0134,0.0122,0.0104,0.0091},
          {1.0308,0.2499,0.2032,0.1721,0.1486,0.1176,0.0974,0.0831,0.0724,0.0643,0.0579,0.0462,0.0384,0.0329,0.0289,0.0259,0.0233,0.0196,0.0170,0.0150,0.0135,0.0123,0.0105,0.0092},
          {0.4280,0.2504,0.2037,0.1725,0.1489,0.1179,0.0976,0.0832,0.0725,0.0644,0.0580,0.0462,0.0385,0.0330,0.0289,0.0259,0.0234,0.0196,0.0170,0.0151,0.0135,0.0123,0.0106,0.0093},
          {0.3503,0.2596,0.2091,0.1769,0.1528,0.1209,0.1001,0.0852,0.0742,0.0660,0.0593,0.0474,0.0395,0.0339,0.0297,0.0267,0.0241,0.0202,0.0176,0.0157,0.0141,0.0129,0.0111,0.0098} };


float const cutPtInvMB2[15][24] = {
          {0.5051,0.2637,0.2073,0.1728,0.1488,0.1162,0.0955,0.0818,0.0712,0.0627,0.0567,0.0454,0.0379,0.0328,0.0288,0.0259,0.0235,0.0200,0.0175,0.0156,0.0141,0.0130,0.0113,0.0101},
          {0.4895,0.2653,0.2095,0.1735,0.1500,0.1168,0.0960,0.0822,0.0715,0.0631,0.0570,0.0456,0.0381,0.0330,0.0289,0.0260,0.0236,0.0201,0.0176,0.0157,0.0142,0.0131,0.0114,0.0102},
          {0.5166,0.2649,0.2100,0.1739,0.1499,0.1171,0.0960,0.0823,0.0715,0.0632,0.0571,0.0456,0.0381,0.0331,0.0290,0.0261,0.0237,0.0201,0.0176,0.0158,0.0142,0.0132,0.0114,0.0102},
          {0.5505,0.2656,0.2094,0.1739,0.1498,0.1169,0.0961,0.0823,0.0717,0.0632,0.0572,0.0457,0.0382,0.0330,0.0290,0.0261,0.0236,0.0202,0.0176,0.0158,0.0142,0.0132,0.0114,0.0103},
          {0.4948,0.2661,0.2097,0.1748,0.1504,0.1172,0.0963,0.0824,0.0717,0.0633,0.0572,0.0457,0.0382,0.0331,0.0291,0.0262,0.0237,0.0202,0.0177,0.0158,0.0143,0.0132,0.0115,0.0103},
          {0.8226,0.2697,0.2117,0.1758,0.1517,0.1178,0.0971,0.0831,0.0722,0.0637,0.0576,0.0461,0.0386,0.0333,0.0293,0.0264,0.0240,0.0205,0.0179,0.0161,0.0145,0.0135,0.0118,0.0105},
          {0.6968,0.2717,0.2147,0.1778,0.1533,0.1194,0.0983,0.0838,0.0731,0.0644,0.0583,0.0467,0.0389,0.0338,0.0296,0.0268,0.0243,0.0208,0.0182,0.0164,0.0147,0.0137,0.0119,0.0108},
          {0.4571,0.2766,0.2147,0.1784,0.1535,0.1199,0.0981,0.0841,0.0731,0.0645,0.0583,0.0468,0.0390,0.0339,0.0298,0.0268,0.0243,0.0209,0.0183,0.0164,0.0148,0.0138,0.0120,0.0108},
          {0.3605,0.2750,0.2168,0.1799,0.1550,0.1207,0.0989,0.0849,0.0736,0.0650,0.0588,0.0471,0.0394,0.0341,0.0300,0.0271,0.0246,0.0211,0.0185,0.0167,0.0150,0.0140,0.0122,0.0110},
          {0.9221,0.2797,0.2169,0.1810,0.1551,0.1208,0.0992,0.0850,0.0739,0.0652,0.0588,0.0472,0.0394,0.0342,0.0301,0.0271,0.0247,0.0211,0.0186,0.0167,0.0151,0.0140,0.0123,0.0111},
          {0.8316,0.2724,0.2148,0.1786,0.1535,0.1195,0.0985,0.0842,0.0733,0.0647,0.0585,0.0468,0.0390,0.0339,0.0298,0.0268,0.0244,0.0209,0.0183,0.0164,0.0148,0.0138,0.0120,0.0108},
          {0.9734,0.2732,0.2154,0.1792,0.1542,0.1203,0.0985,0.0845,0.0733,0.0648,0.0586,0.0469,0.0391,0.0340,0.0298,0.0269,0.0244,0.0210,0.0184,0.0165,0.0149,0.0138,0.0121,0.0109},
          {0.8544,0.2810,0.2171,0.1804,0.1555,0.1207,0.0993,0.0851,0.0738,0.0652,0.0590,0.0472,0.0394,0.0343,0.0301,0.0272,0.0247,0.0212,0.0186,0.0168,0.0151,0.0140,0.0123,0.0111},
          {0.8574,0.2799,0.2174,0.1813,0.1557,0.1212,0.0993,0.0851,0.0741,0.0653,0.0590,0.0473,0.0396,0.0344,0.0302,0.0273,0.0248,0.0213,0.0187,0.0168,0.0152,0.0141,0.0124,0.0112},
          {0.8574,0.6947,0.5410,0.4498,0.3875,0.3013,0.2474,0.2122,0.1840,0.1626,0.1470,0.1177,0.0983,0.0854,0.0749,0.0676,0.0614,0.0527,0.0463,0.0417,0.0376,0.0349,0.0305,0.0275} };

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
    int getDTPt();
    int getDTPtMin( float nSigmas );
    int getDTPtMax( float nSigmas );

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

