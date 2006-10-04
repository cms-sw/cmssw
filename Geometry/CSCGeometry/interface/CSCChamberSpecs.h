#ifndef Geometry_CSCGeometry_CSCChamberSpecs_H
#define Geometry_CSCGeometry_CSCChamberSpecs_H

/** \class CSCChamberSpecs
 *  Each endcap muon CSCChamberSpecs object is a set of parameters
 *  for one of several types of chamber. 
 * 
 * \author Tim Cox
 *
 * There are only a small finite number (9) of distinct chamber types 
 * in the hardware, according to physical dimensions and properties. 
 * The individual types currently correspond to each (Station,Ring) pair.
 *
 * RESPONSIBILITIES: <br>
 * Supply chamber type-dependent parameters on demand.
 * Geometrical size is not a responsibility of the Specs class.
 *
 * COLLABORATORS: <br>
 * A Chamber knows its associated Specs.
 *
 * \warning Disclaimer:
 * The mess of methods is only a temporary hack until we decide
 * how the old MuEndOrcaSpec values should be packaged. <BR>
 *
 */

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include <cmath>
#include <map>
#include <string>
#include <vector>

class CSCLayerGeometry;
class CSCWireGroupPackage;
class Topology; //@@ Can go once GeomDetType has it removed.
class TrapezoidalPlaneBounds;

class CSCChamberSpecs : public GeomDetType { 

public:

  typedef std::vector<float> CSCSpecsParcel;

  /// Default ctor
  CSCChamberSpecs();

  /// Usual ctor from supplied params
  CSCChamberSpecs( int iChamberType , 
		   const TrapezoidalPlaneBounds & mediaShape, 
                   const CSCSpecsParcel& fupar,
                   const CSCWireGroupPackage& wg 
		  );

  /// Destructor
  ~CSCChamberSpecs();

  /// Allow comparison of Specs objects
  bool operator!=( const CSCChamberSpecs& specs ) const;
  bool operator==( const CSCChamberSpecs& specs ) const;

  //@@ Will be moved from GeomDetType interface (good, so we're no longer forced to pick something)
  /// Returns StripTopology of the odd-layer, positive-z geometry
  virtual const Topology& topology() const;

  /// Accessors for LayerGeometry's
  const CSCLayerGeometry* const oddLayerGeometry( int iendcap ) const 
   { return (iendcap==1? poszOddLayerGeometry:negzOddLayerGeometry);}
  const CSCLayerGeometry* const evenLayerGeometry( int iendcap ) const 
   { return (iendcap==1? poszEvenLayerGeometry:negzEvenLayerGeometry);}

   /**
   *  The 'type' of the chamber associated with this ChamberSpecs. <br>
   *  The chamber type is a unique integer 1-10 for a station, ring pair
   *  MEs/r.
   *  The type value is defined as <br>
   *        1           for S = 1  and R=A=4 split strips in ME1/1 <br>
   *      2,3,4 =  R+1  for S = 1  and R = 1,2,3 <br>
   *      5-10  = 2*S+R for S = 2,3,4 and R = 1,2 <br>
   *   It is conventional usage in the Endcap Muon community/
   */
  int chamberType() const; 

  /** The (hardware) name for the chamber type associated with
   * this ChamberSpecs. <BR>
   * It is a string "MEs/r" where integer s = station, r = ring.<BR>
   */
  std::string chamberTypeName() const;




  // CHAMBER 'PARAMETER' FUNCTIONS
  //@@ FIXME these must all be sorted out in a world of real conditions & calibration

  float stripResolution()     const {return specsValue(0);}
  float wireResolution()      const {return specsValue(1);}
  float efficiency()          const {return specsValue(2);}
  float timeWindow()          const {return specsValue(3);}

  /**
   * neutron hit rate per CSC layer per event.
   */
  float neutronRate()         const {return specsValue(4);}

  /**
   * number of strips in one chamber.
   */
  int nStrips()               const {return int( specsValue(5)*2. );}

  /**
   * number of strips 2*nnodes+1 around hit.
   */
  int   nNodes()              const {return int( specsValue(6) );}
  int   nWiresPerGroup()      const {return int( specsValue(7) );}
  int   nFloatingStrips()     const {return int( specsValue(8) );}

  /**
   * strip pitch in phi, in radians (the strips are fan-shaped)
   */
  float stripPhiPitch() const {return 1.E-03 * stripDeltaPhi[theChamberType-1];}

  /**
   * offset to centreToIntersection, in cm (the backed-out corrections 
   * for positioning the strips)
   */
  float ctiOffset() const {return centreToIntersectionOffset[theChamberType-1];}

  /**
   * wire spacing, in cm. All layers in a chamber have the same spacing.
   */
  // specValue is superseded by LayerGeometry value(s)...
  //  float wireSpacing()         const {return specsValue(9);}
  float wireSpacing() const;
 
  /**
   * distance from anode to cathode, in cm.
   */
  float anodeCathodeSpacing() const {return specsValue(10);}

  float gasGain()             const;

  float voltage()             const {return specsValue(12);}

  //  float phiTilt()             const {return specsValue(13);}

  float calibrationError()    const {return specsValue(14);}

  float electronAttraction()  const {return specsValue(15);}

  /**
  * the fraction of the charge that survives to reach the cathode.
  */
  float fractionQS()          const {return specsValue(16);}

  /**
   * ADC calibration, in fC.
   */
  //@@  float chargePerCount()      const {return specsValue(17);}
  float chargePerCount() const;

  /**
   * anode wire radius, in cm.
   */
  float wireRadius()          const {return specsValue(18);}

  /**
   * Fast shaper peaking time (ns).
   */
  float shaperPeakingTime()   const {return specsValue(19);}
  
  /**
   * the constant term in the electronics noise, in # of electrons.
   */
  float constantNoise() const {return specsValue(85);}

  /**
   * the # of noise electrons per picofarad of capacitance.
   */
  float e_pF() const {return specsValue(86);}

  /**
   * the number of noise electrons.
   * the time interval is in ns
   *  we show the noise for 100 ns, then scale as 1/sqrt(time)
   */
  //@@ Why 1/sqrt(time)?!?
  float wireNoise(float timeInterval) const {
    const float WIRE_CAPACITANCE = 8.; // in pF
    return (constantNoise() + nWiresPerGroup()*WIRE_CAPACITANCE)
           / sqrt(timeInterval/100.);
  }
 
  float stripNoise(float timeInterval) const;

  //@@ The following is nonsense to be fixed at some stage
  //  float adcThreshold()        const {return 9.99;}  





  // STATIC FUNCTIONS

  /**
   * The usual integer label for 'chamber type' of this ring and station
   */
  static int whatChamberType( int istation, int iring );

  /**
   * Return the CSCChamberSpecs* for given chamber type
   * if it exists, or 0 if it has not been created.
   */
  static CSCChamberSpecs* lookUp( int iChamberType );

  /**
   * Build CSCChamberSpecs for given chamber type.
   *
   * @@ This method is probably a good candidate to be replaced
   * by a factory.
   */
  static CSCChamberSpecs* build( int iChamberType,
          const std::vector<float>& fpar, const std::vector<float>& fupar,
			       const CSCWireGroupPackage& wg );

  static void setGangedStripsInME1a(bool gs)  { gangedstripsME1a = gs; }
  static void setOnlyWiresInME1a(bool ow)     { onlywiresME1a = ow; }
  static void setUseRadialStrips(bool rs)     { useRadialStrips = rs; }
  static void setUseRealWireGeometry(bool wg) { useRealWireGeometry = wg; }
  static void setUseCentreTIOffsets(bool cti) { useCentreTIOffsets = cti; }

  /**
   * Ganged strips in ME1a
   */
  static bool gangedStrips() { return gangedstripsME1a; }

  /**
   * Wires only in ME1a
   */
  static bool wiresOnly() { return onlywiresME1a; }

  /**
   * Strips modelled as radial rather than trapezoidal
   */
  static bool radialStrips() { return useRadialStrips; }

  /**
   * Wire geometry modelled as real hardware (complex
   * groupings of wires and dead regions) or as a 'pseudo'
   * geometry with just one wire grouping per chamber type
   * (as was done in ORCA versions up to and including ORCA_8_8_1.)
   */
  static bool realWireGeometry() { return useRealWireGeometry; }

  /**
   * Use the backed-out offsets for theCentreToIntersection in
   * CSCLayerGeometry
   */
  static bool centreTIOffsets() { return useCentreTIOffsets; }

 private:

  /// Accessor to chamber specs values
  float specsValue( int index ) const;

  /// Dump parameters for overall strip and wire modelling
  static void whatModelling();


// A ChamberSpecs has 4 associated LayerGeometry's
  CSCLayerGeometry* poszOddLayerGeometry;
  CSCLayerGeometry* poszEvenLayerGeometry;
  CSCLayerGeometry* negzOddLayerGeometry;
  CSCLayerGeometry* negzEvenLayerGeometry;

//  theChamberType is a unique integer 1-10 for a station, ring pair.

  //  The type value is defined as <br>
  //        1           for S = 1  and R=A=4 split strips in ME11 <br>
  //      2,3,4 =  R+1  for S = 1  and R = 1,2,3 <br>
  //      5-10  = 2*S+R for S = 2,3,4 and R = 1,2 <br>
  
  int theChamberType;

  // Store for specs parameter values
  CSCSpecsParcel theSpecsValues;

// Store pointers to Specs objects as we build them.
  static std::map<int, CSCChamberSpecs*, std::less<int> > specsMap;

  // Delta-phi width of strip in each chamber type (in mrad)
  static const float stripDeltaPhi[10];

  // Backed-out offsets for the whereStripsMeet calculation
  static const float centreToIntersectionOffset[10];

  // Names of chamber types
  static const std::string theName[10];

  // Name of this class 
  static const std::string myName;

  // Parameters controlling modelling of geometry _within_ all DetUnit's
  static bool theFirstCall;
  static bool gangedstripsME1a;
  static bool onlywiresME1a;
  static bool useRadialStrips;
  static bool useRealWireGeometry;
  static bool useCentreTIOffsets;

};

#endif // Geometry_CSCGeometry_CSCChamberSpecs_H
