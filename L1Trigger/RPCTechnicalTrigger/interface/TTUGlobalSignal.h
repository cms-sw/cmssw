#ifndef TTUGLOBALSIGNAL_H
#define TTUGLOBALSIGNAL_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"

#include <map>

/** @class TTUGlobalSignal TTUGlobalSignal.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2008-11-29
 */
class TTUGlobalSignal : public RPCInputSignal {
public:
  /// Standard constructor
  TTUGlobalSignal(){};

  TTUGlobalSignal(std::map<int, TTUInput*>*);

  ~TTUGlobalSignal() override;  ///< Destructor

  void clear() override{};

  std::map<int, TTUInput*>* m_wheelmap;

protected:
private:
};
#endif  // TTUGLOBALSIGNAL_H
