#ifndef Geometry_CSCGeometry_CSCChamberSpecs_H
#define Geometry_CSCGeometry_CSCChamberSpecs_H

/** \class CSCChamberSpecs
 *  Each endcap muon CSCChamberSpecs object is a set of parameters
 *  for one of several types of chamber. 
 * 
 * \author Tim Cox
 *
 * There are only a small finite number (9, or 10 incl. ME1A as separate) of distinct chamber types 
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
 * The mess of methods was supposed to be a temporary hack until it was decided
 * how to handle such spec pars... but there's still nothing better. <BR>
 *
 */

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include <cmath>
#include <string>
#include <vector>

class CSCGeometry;
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
  CSCChamberSpecs( const CSCGeometry* geom, int iChamberType, 
		   const TrapezoidalPlaneBounds& mediaShape,
                   const CSCSpecsParcel& fupar,
                   const CSCWireGroupPackage& wg 
		  );

  /// Destructor
  ~CSCChamberSpecs();

  /// Allow comparison of Specs objects
  bool operator!=( const CSCChamberSpecs& specs ) const;
  bool operator==( const CSCChamberSpecs& specs ) const;

  //@@ Topology() will be removed from GeomDetType interface (good, so we're no longer forced to pick something)
  //@@ But still there as of Aug-2007. So much for design.

  /// Returns StripTopology of the odd-layer, positive-z geometry
  virtual const Topology& topology() const;

  /// Accessors for LayerGeometry's
  const CSCLayerGeometry* oddLayerGeometry( int iendcap ) const 
   { return (iendcap==1? poszOddLayerGeometry:negzOddLayerGeometry);}
  const CSCLayerGeometry* evenLayerGeometry( int iendcap ) const 
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
  int nStrips()               const {return nstrips;}

  /**
   * number of strips 2*nnodes+1 around hit.
   */
  int   nNodes()              const {return int( specsValue(6) );}
  int   nWiresPerGroup()      const {return int( specsValue(7) );}
  int   nFloatingStrips()     const {return int( specsValue(8) );}

  /**
   * strip pitch in phi, in radians (the strips are fan-shaped)
   */
  float stripPhiPitch() const {return 1.E-03 * stripDeltaPhi;}

  /**
   * offset to centreToIntersection, in cm (the backed-out corrections 
   * for positioning the strips)
   */
  float ctiOffset() const {return centreToIntersectionOffset;}

  /**
   * wire spacing, in cm. All layers in a chamber have the same spacing.
   */
  float wireSpacing() const;
 
  /**
   * distance from anode to cathode, in cm.
   */
  float anodeCathodeSpacing() const {return specsValue(9);}

  float gasGain()             const;

  float voltage()             const {return specsValue(11);}

  float calibrationError()    const {return specsValue(13);}

  float electronAttraction()  const {return specsValue(14);}

  /**
  * the fraction of the charge that survives to reach the cathode.
  */
  float fractionQS()          const {return specsValue(15);}

  /**
   * ADC calibration, in fC.
   */
  float chargePerCount() const;

  /**
   * anode wire radius, in cm.
   */
  float wireRadius()          const {return specsValue(17);}

  /**
   * Fast shaper peaking time (ns).
   */
  float shaperPeakingTime()   const {return specsValue(18);}
  
  /**
   * the constant term in the electronics noise, in # of electrons.
   */
  float constantNoise() const {return specsValue(22);}

  /**
   * the # of noise electrons per picofarad of capacitance.
   */
  float e_pF() const {return specsValue(23);}

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

  /**
   * Are strips ganged?
   */
   bool gangedStrips() const { return gangedStrips_; }

  // STATIC FUNCTION

  /**
   * The usual integer label for 'chamber type' of this ring and station
   */
  static int whatChamberType( int istation, int iring );

 private:

  /// Accessor to chamber specs values
  float specsValue( int index ) const {
    return theSpecsValues[ index ];  
  }

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

  // Careful that the following order of data members is required to ensure
  // correct initialization in constructor initialization list!

  // Store for specs parameter values
  CSCSpecsParcel theSpecsValues;

  int nstrips; // no. of strips per layer
  float stripDeltaPhi;   // Delta-phi width of strip in this chamber type (in mrad)
  float centreToIntersectionOffset; // Possible correction to whereStripsMeet

  bool gangedStrips_;

  // Names of chamber types
  static const std::string theName[10];

  // Name of this class 
  static const std::string myName;

};

#endif // Geometry_CSCGeometry_CSCChamberSpecs_H
