#ifndef Alignment_MuonAlignment_MuonScenarioBuilder_h
#define Alignment_MuonAlignment_MuonScenarioBuilder_h

/** \class MuonScenarioBuilder
 *  The misalignment scenario builder.
 *
 *  $Date: 2008/04/15 16:05:53 $
 *  $Revision: 1.4 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include <vector>


#include "Alignment/CommonAlignment/interface/AlignableModifier.h"
#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
/// Builds a scenario from configuration and applies it to the alignable Muon.

class MuonScenarioBuilder : public MisalignmentScenarioBuilder
{

public:

  /// Constructor
  explicit MuonScenarioBuilder( Alignable* alignable );

  /// Destructor
  ~MuonScenarioBuilder() {};

  /// Apply misalignment scenario to the Muon
  void applyScenario( const edm::ParameterSet& scenario );

  /// This special method allows to move a DTsector by a same amount
  void moveDTSectors( const edm::ParameterSet& scenario );

  /// this special method allows to move a CSCsector by a same amount
  void moveCSCSectors( const edm::ParameterSet& scenario );
  
  /// this special method allows to move the complete muon system by a same amount
  void moveMuon( const edm::ParameterSet& scenario );
  
  std::vector<float> extractParameters( const edm::ParameterSet& , const char* );

  void moveChamberInSector( Alignable *, std::vector<float>, std::vector<float>, std::vector<float> , std::vector<float> );
private: // Members

  AlignableMuon* theAlignableMuon;   ///< Pointer to alignable Muon object
  
  AlignableModifier theMuonModifier; 
};



#endif
