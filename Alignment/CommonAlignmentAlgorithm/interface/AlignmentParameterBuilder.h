#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h

#include <string>
#include <vector>

/** \class AlignmentParameterBuilder
 *
 *  Build Alignment Parameter Structure 
 *
 *  $Date: 2006/10/20 13:24:55 $
 *  $Revision: 1.3 $
 *  (last update by $Author: flucke $)
 */

namespace edm {
  class ParameterSet;
}
class Alignabletracker;

class AlignmentParameterBuilder 
{
public:

  /// Constructor
  explicit AlignmentParameterBuilder(AlignableTracker *alignableTracker);
  /// Constructor adding selection using int addSelections(const edm::ParameterSet &pSet)
  AlignmentParameterBuilder(AlignableTracker *alignableTracker, const edm::ParameterSet &pSet);

  /// destructor 
  virtual ~AlignmentParameterBuilder() {};

  /// Add several selections defined by the PSet which must contain a vstring like e.g.
  /// vstring alignableParamSelector = { "PixelHalfBarrelLadders,111000,pixelSelection",
  ///                                    "BarrelDSRods,111000",
  ///                                    "BarrelSSRods,101000"}
  /// The '11100' part defines which d.o.f. to be aligned (x,y,z,alpha,beta,gamma)
  /// returns number of added selections or -1 if problems (then also an error is logged)
  /// If a string contains a third, comma separated part (e.g. ',pixelSelection'),
  /// a further PSet of that name is expected to select eta/z/phi/r-ranges
  unsigned int addSelections(const edm::ParameterSet &pset);

  /// Add arbitrary selection of Alignables 
  void add (const std::vector<Alignable*>& alignables, const std::vector<bool> &sel);

  /// Get list of alignables for which AlignmentParameters are built 
  std::vector<Alignable*> alignables() { return theAlignables; };

  /// Remove n Alignables from list 
  void fixAlignables( int n );

  /// Decoding string to select local rigid body parameters into vector<bool>,
  /// "101001" will mean x,z,gamma, but not y,alpha,beta
  /// cms::Exception if problems while decoding (e.g. length of sring)
  std::vector<bool> decodeParamSel(const std::string &selString) const;
  /// Decomposing input string 's' into parts separated by 'delimiter'
  std::vector<std::string> decompose(const std::string &s, std::string::value_type delimiter) const;
private:

  // data members

  /// Vector of alignables 
  std::vector<Alignable*> theAlignables;

  /// Alignable tracker   
  AlignableTracker* theAlignableTracker;

};

#endif
