#ifndef CSCGeomety_CSCGattiFunction_h
#define CSCGeomety_CSCGattiFunction_h

/** \class CSCGattiFunction
 *
 * Represent functional form of charge distribution over strips
 * in Endcap Muon CSC's.
 *
 * \author Rick Wilkinson
 *
 * This is required in building RecHits from strips in CSCRecHit
 * and for distributing charge over strips in CSCDigitizer.
 *
 * It was ported from FORTRAN in CMSISM to C++ in ORCA and then CMSSW. <BR>
 *
 *  Function: describes the cathode signal using                       <BR>
 *                the single-parameter Gatti formula:                  <BR>
@code
 *                              1 - tanh(K_2 * lambda)**2              
 *     Gamma(lambda) = K_1 * -------------------------------           
 *                           1 + K_3 * tanh (K_2 *lambda)**2           
 *     lambda = x/h, h is anode cathode spacing                        
 *                                                                     
 *     K_2 = pi/2*(1 - 0.5*sqrt(K_3))                                  
 *                                                                     
 *              K_2*sqrt(K_3)                                          
 *      K_1 = -------------------                                      
 *            4 * atan(sqrt(K_3))                                      
 *                                                                     
 *  References  : E.Gatti, A.Longoni, NIM 163 (1979) 82-93.            
 *                                                                     
 *  For K_3, "It is used parametrization from Fig.2 from E.Mathieson   
 *            J.S.Gordon, "Cathode charge distributions in multi-      
 *            wire chambers", NIM 227 (1984) 277-282"                  
 *  (comment from GATTI3 in cmsim/src/mc_uty/.)                        
@endcode
 
 */

class CSCChamberSpecs;

class CSCGattiFunction {
public:
  CSCGattiFunction();
  /// Calculates k1, k2, k3, h per chamber type, if necessary
  void initChamberSpecs(const CSCChamberSpecs &);

  ///  Returns the fraction of charge on a strip centered
  ///  a distance of x away from the center of the shower,
  ///  at zero.  Note that the user is responsible for making
  ///  sure the constants have been initialized using the chamber specs.
  double binValue(double x, double stripWidth) const;

private:
  // geometry constants for the detector
  double k1, k2, k3, h;
  double norm, sqrtk3;

  const CSCChamberSpecs *thePreviousSpecs;
};

#endif
