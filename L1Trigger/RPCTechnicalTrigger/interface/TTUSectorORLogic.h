#ifndef INTERFACE_TTUSECTORORLOGIC_H
#define INTERFACE_TTUSECTORORLOGIC_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTULogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"

#include <iostream>
#include <vector>

/** @class TTUSectorORLogic TTUSectorORLogic.h interface/TTUSectorORLogic.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-06-15
 */
class TTUSectorORLogic : public TTULogic {
public:
  /// Standard constructor
  TTUSectorORLogic();

  ~TTUSectorORLogic() override;  ///< Destructor

  //... from TTULogic interface:

  bool process(const TTUInput&) override;

  void setBoardSpecs(const TTUBoardSpecs::TTUBoardConfig&) override;

  //...

protected:
private:
  bool m_debug;

  int m_maxsectors;
};
#endif  // INTERFACE_TTUSECTORORLOGIC_H
