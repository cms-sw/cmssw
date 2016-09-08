#ifndef HeavyFlavorAnalysis_RecoDecay_BPHVertexSelect_h
#define HeavyFlavorAnalysis_RecoDecay_BPHVertexSelect_h
/** \class BPHVertexSelect
 *
 *  Description: 
 *     Base class for candidate selection at vertex reconstruction level
 *
 *
 *  $Date: 2015-07-06 18:34:25 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class BPHDecayVertex;

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHVertexSelect {

 public:

  /** Constructor
   */
  BPHVertexSelect();

  /** Destructor
   */
  virtual ~BPHVertexSelect();

  /** Operations
   */
  /// accept function
  virtual bool accept( const BPHDecayVertex& cand ) const = 0;

 private:

  // private copy and assigment constructors
  BPHVertexSelect           ( const BPHVertexSelect& x );
  BPHVertexSelect& operator=( const BPHVertexSelect& x );

};


#endif

