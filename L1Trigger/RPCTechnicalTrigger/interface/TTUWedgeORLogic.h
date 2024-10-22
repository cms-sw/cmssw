#ifndef INTERFACE_TTUWEDGEORLOGIC_H
#define INTERFACE_TTUWEDGEORLOGIC_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTULogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"

#include <iostream>
#include <vector>
#include <map>

/** @class TTUWedgeORLogic TTUWedgeORLogic.h interface/TTUWedgeORLogic.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-08-09
 */
class TTUWedgeORLogic : public TTULogic {
public:
  /// Standard constructor
  TTUWedgeORLogic();

  ~TTUWedgeORLogic() override;  ///< Destructor

  bool process(const TTUInput&) override;

  void setBoardSpecs(const TTUBoardSpecs::TTUBoardConfig&) override;

protected:
private:
  bool m_debug;

  int m_maxsectors;

  int m_maxwedges;

  std::map<int, int> m_wheelMajority;

  //std::vector<int> m_wedgeSector;

  std::map<int, int> m_wedgeSector;
};
#endif  // INTERFACE_TTUWEDGEORLOGIC_H
