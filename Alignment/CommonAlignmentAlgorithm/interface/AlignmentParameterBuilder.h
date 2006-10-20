#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include <string>
#include <vector>

/** \class AlignmentParameterBuilder
 *
 *  Build Alignment Parameter Structure 
 *
 *  $Date: 2006/10/17 11:02:42 $
 *  $Revision: 1.11 $
 *  (last update by $Author: flucke $)
 */



class AlignmentParameterBuilder 
{
public:

  /// Constructor
  AlignmentParameterBuilder(AlignableTracker* alignableTracker);

  /// destructor 
  ~AlignmentParameterBuilder() {};

  /// Add several selections defined by the PSet which must contain a vstring like e.g.
  /// vstring alignableParamSelector = { "PixelHalfBarrelLadders,111000,pixelSelection",
  ///                                    "BarrelDSRods,111000",
  ///                                    "BarrelSSRods,101000"}
  /// The '11100' part defines which d.o.f. to be aligned (x,y,z,alpha,beta,gamma)
  /// returns number of added selections or -1 if problems (then also an error is logged)
  /// If a string contains a third, comma separated part (e.g. ',pixelSelection'),
  /// a further PSet of that name is expected to select eta/z/phi/r-ranges
  int addSelections(const edm::ParameterSet &pset);

  /// Add predefined selection of alignables defined by a string 
  void addSelection(const std::string &name, const std::vector<bool> &sel);

  /// Add arbitrary selection of Alignables 
  void add (const std::vector<Alignable*>& alignables, const std::vector<bool> &sel);

  /// Add all level 1 objects (Dets) 
  void addAllDets(const std::vector<bool> &sel);

  /// Add all level 2 (Rod or equivalent) objects 
  void addAllRods(const std::vector<bool> &sel);

  /// Add all level 3 (Layer or equivalent) objects 
  void addAllLayers(const std::vector<bool> &sel);

  /// Add all level 4 (Halfbarrel etc) objects 
  void addAllComponents(const std::vector<bool> &sel);

  /// Add all alignables 
  void addAllAlignables(const std::vector<bool> &sel);

  /// Get list of alignables for which AlignmentParameters are built 
  std::vector<Alignable*> alignables() { return theAlignables; };

  /// Remove n Alignables from list 
  void fixAlignables( int n );

  /// Decoding string to select local rigid body parameters into vector<bool>,
  /// "101001" will mean x,z,gamma, but not y,alpha,beta
  /// LogError if problems while decoding (e.g. length of sring)
  std::vector<bool> decodeParamSel(const std::string &selString) const;
  std::vector<std::string> decompose(const std::string &s, std::string::value_type delimiter) const;
private:

  // data members

  /// Vector of alignables 
  std::vector<Alignable*> theAlignables;

  /// Alignable tracker   
  AlignableTracker* theAlignableTracker;

  /// Alignable id converter
  TrackerAlignableId* theTrackerAlignableId;

  bool theOnlyDS;
  bool theOnlySS;
  int  theMinLayer,theMaxLayer;
  bool theSelLayers;

};

#endif
