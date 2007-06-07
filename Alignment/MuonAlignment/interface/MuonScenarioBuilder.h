#ifndef Alignment_MuonAlignment_MuonScenarioBuilder_h
#define Alignment_MuonAlignment_MuonScenarioBuilder_h

/** \class MuonScenarioBuilder
 *  The misalignment scenario builder.
 *
 *  $Date: 2007/01/12 09:47:42 $
 *  $Revision: 1.1 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableModifier.h"
#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
/// Builds a scenario from configuration and applies it to the alignable Muon.

class MuonScenarioBuilder : public MisalignmentScenarioBuilder
{

public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;


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
  
  std::vector<float> extractParameters( const edm::ParameterSet& , char* );

  void moveChamberInSector( Alignable *, std::vector<float>, std::vector<float>, std::vector<float> , std::vector<float> );
private: // Members

  AlignableMuon* theAlignableMuon;   ///< Pointer to alignable Muon object
  
  AlignableModifier theMuonModifier; 
};



#endif
