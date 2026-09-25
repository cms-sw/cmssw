#ifndef Alignment_MTDAlignment_MTDScenarioBuilder_h
#define Alignment_MTDAlignment_MTDScenarioBuilder_h

/** \class MTDScenarioBuilder
 *  The misalignment scenario builder.
 *
 *  $Date: 2024/10/25 17:09:58 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include <vector>

#include "Alignment/CommonAlignment/interface/AlignableModifier.h"
#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Alignment/MTDAlignment/interface/AlignableMTD.h"
/// Builds a scenario from configuration and applies it to the alignable MTD.

class MTDScenarioBuilder : public MisalignmentScenarioBuilder {
public:
  /// Constructor
  explicit MTDScenarioBuilder(Alignable* alignable);

  /// Destructor
  ~MTDScenarioBuilder() override {}

  /// Apply misalignment scenario to the MTD
  void applyScenario(const edm::ParameterSet& scenario) override;

  /// this special method allows to move the complete MTD system by a same amount
  void moveMTD(const edm::ParameterSet& scenario);

  align::Scalars extractParameters(const edm::ParameterSet&, const char*);
  /*
  void moveChamberInSector(
      Alignable*, const align::Scalars&, const align::Scalars&, const align::Scalars&, const align::Scalars&);
  */

private:                          // Members
  AlignableMTD* theAlignableMTD;  ///< Pointer to alignable MTD object

  AlignableModifier theMTDModifier;
};

#endif
